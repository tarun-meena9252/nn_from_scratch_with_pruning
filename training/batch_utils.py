
import numpy as np


def iterate_minibatches(
    x,
    y,
    batch_size=64,
    shuffle=True,
    seed=None,
):
    if len(x) != len(y):
        raise ValueError("Input and target counts must match")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if len(x) == 0:
        raise ValueError("Dataset must not be empty")

    indices = np.arange(len(x))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(x), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield x[batch_indices], y[batch_indices]
