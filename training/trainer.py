
import numpy as np

from training.batch_utils import iterate_minibatches


def train_one_epoch(
    model,
    loss_fn,
    optimizer,
    x_train,
    y_train,
    batch_size=64,
    seed=None,
):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in iterate_minibatches(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    ):
        # Forward pass.
        logits = model.forward(x_batch)

        # Compute mean loss for this batch.
        loss = loss_fn.forward(logits, y_batch)

        # Measure predictions before updating parameters.
        predictions = np.argmax(logits, axis=1)
        total_correct += np.sum(predictions == y_batch)

        # Backpropagation.
        model.zero_grad()
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)

        # Adam parameter update.
        optimizer.step()

        # Accumulate metrics weighted by batch size.
        batch_count = len(y_batch)
        total_loss += loss * batch_count
        total_samples += batch_count

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def evaluate(model, loss_fn, x, y, batch_size=256):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x_batch, y_batch in iterate_minibatches(
        x, y, batch_size=batch_size, shuffle=False
    ):
        logits = model.forward(x_batch)
        loss = loss_fn.forward(logits, y_batch)

        total_loss += loss * len(y_batch)
        total_correct += np.sum(
            np.argmax(logits, axis=1) == y_batch
        )
        total_samples += len(y_batch)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
