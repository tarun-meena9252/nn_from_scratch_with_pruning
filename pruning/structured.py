import copy
import numpy as np

from nn.layers.linear import Linear
from nn.models.sequential import Sequential


def neuron_importance(linear_layer):
    """
    Score each output neuron by the L2 norm of its incoming weights.

    W shape: (in_features, out_features)
    Returns: (out_features,) importance scores.
    """
    weights = linear_layer.W.data
    return np.linalg.norm(weights, axis=0)


def select_neurons_to_keep(linear_layer, prune_fraction):
    """
    Return indices of neurons to keep, preserving the most
    important neurons according to their weight magnitudes.
    """
    if not 0.0 <= prune_fraction < 1.0:
        raise ValueError("prune_fraction must be in [0, 1)")

    scores = neuron_importance(linear_layer)
    total_neurons = len(scores)
    prune_count = int(total_neurons * prune_fraction)

    # Keep at least one neuron.
    keep_count = max(1, total_neurons - prune_count)

    # Stable ordering makes ties deterministic.
    keep_indices = np.argsort(-scores, kind="stable")[:keep_count]

    # Preserve the original neuron order.
    return np.sort(keep_indices)



def structured_prune_model(
    model,
    prune_fractions,
    criterion="magnitude",
    importance_scores=None,
):
    """
    Create a smaller model by pruning hidden neurons.

    Args:
        model: Trained Sequential model.
        prune_fractions: Dict mapping hidden Linear-layer indices to
                         fractions to prune, e.g. {0: 0.25, 1: 0.25}.
        criterion: "magnitude" or "taylor".
        importance_scores: Per-weight Taylor score matrices, one per
                           Linear layer, as returned by compute_taylor_scores.

    Returns:
        A new, structurally smaller Sequential model.
    """
    import copy
    import numpy as np
    from nn.layers.linear import Linear
    from nn.models.sequential import Sequential

    if criterion not in ("magnitude", "taylor"):
        raise ValueError("criterion must be 'magnitude' or 'taylor'")

    linear_layers = [
        layer for layer in model.layers if isinstance(layer, Linear)
    ]

    if len(linear_layers) < 2:
        raise ValueError("Model must contain hidden and output Linear layers")

    hidden_count = len(linear_layers) - 1

    for index, fraction in prune_fractions.items():
        if not isinstance(index, int) or not 0 <= index < hidden_count:
            raise ValueError(f"Invalid hidden Linear-layer index: {index}")

        if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError("Prune fractions must be in [0.0, 1.0)")

    if criterion == "taylor":
        if importance_scores is None:
            raise ValueError("Taylor pruning requires importance_scores")

        if len(importance_scores) != len(linear_layers):
            raise ValueError(
                "Provide one Taylor score matrix per Linear layer"
            )

        for layer, scores in zip(linear_layers, importance_scores):
            if np.shape(scores) != layer.W.data.shape:
                raise ValueError(
                    "Each Taylor score matrix must match its layer's "
                    "weight shape"
                )

    new_layers = []
    previous_keep = None
    linear_index = 0

    for layer in model.layers:
        if not isinstance(layer, Linear):
            new_layers.append(copy.deepcopy(layer))
            continue

        # Remove incoming connections from previously pruned neurons.
        weights = layer.W.data
        biases = layer.b.data

        if previous_keep is not None:
            weights = weights[previous_keep, :]

        if linear_index < hidden_count:
            fraction = prune_fractions.get(linear_index, 0.0)

            # 'weights' has already been sliced to retain only
            # connections from surviving neurons in the previous layer.
            if criterion == "magnitude":
                scores = np.linalg.norm(weights, axis=0)

            else:
                # Start with the Taylor scores computed on the current model.
                score_matrix = importance_scores[linear_index]

                # Discard score contributions from connections whose
                # input neurons were removed at the previous stage.
                if previous_keep is not None:
                    score_matrix = score_matrix[previous_keep, :]

                # Sum importance over surviving incoming connections.
                scores = np.sum(score_matrix, axis=0)

            prune_count = int(len(scores) * fraction)
            keep_count = max(1, len(scores) - prune_count)

            previous_keep = np.sort(
                np.argsort(-scores, kind="stable")[:keep_count]
            )

            weights = weights[:, previous_keep]
            biases = biases[:, previous_keep]
        else:
            # Keep all output neurons.
            previous_keep = None

        new_linear = Linear(
            weights.shape[0],
            weights.shape[1],
            initializer="zeros",
        )
        new_linear.W.data[...] = weights
        new_linear.b.data[...] = biases
        new_layers.append(new_linear)

        linear_index += 1

    return Sequential(new_layers)
