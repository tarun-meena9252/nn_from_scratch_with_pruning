
import numpy as np

from nn.layers.linear import Linear


def taylor_weight_score(weights, gradients):
    """
    First-order Taylor sensitivity score for each weight.

    score_ij = abs(weight_ij * gradient_ij)
    """
    if weights.shape != gradients.shape:
        raise ValueError("Weights and gradients must have the same shape")

    return np.abs(weights * gradients)


def taylor_neuron_scores(weight_scores):
    """
    Aggregate incoming-weight Taylor scores for each output neuron.

    weight_scores shape: (in_features, out_features)
    returns shape: (out_features,)
    """
    if weight_scores.ndim != 2:
        raise ValueError("Weight scores must be a 2D array")

    return np.sum(weight_scores, axis=0)


def compute_taylor_scores(
    model,
    loss_fn,
    x,
    y,
    batch_size=64,
    max_batches=None,
):
    """
    Estimate Taylor sensitivity scores over training data.

    Returns a list of score matrices, one for each Linear layer.
    Scores are averaged across batches.

    The model's weights are not modified.
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    linear_layers = [
        layer for layer in model.layers
        if isinstance(layer, Linear)
    ]

    if not linear_layers:
        raise ValueError("Model must contain at least one Linear layer")

    accumulated_scores = [
        np.zeros_like(layer.W.data, dtype=np.float64)
        for layer in linear_layers
    ]

    batches_processed = 0

    for start in range(0, len(x), batch_size):
        if max_batches is not None and batches_processed >= max_batches:
            break

        end = min(start + batch_size, len(x))
        batch_x = x[start:end]
        batch_y = y[start:end]

        model.zero_grad()

        logits = model.forward(batch_x)
        loss_fn.forward(logits, batch_y)

        grad_logits = loss_fn.backward()
        model.backward(grad_logits)

        for index, layer in enumerate(linear_layers):
            accumulated_scores[index] += taylor_weight_score(
                layer.W.data,
                layer.W.grad,
            )

        batches_processed += 1

    if batches_processed == 0:
        raise ValueError("No batches were processed")

    model.zero_grad()

    return [
        scores / batches_processed
        for scores in accumulated_scores
    ]
