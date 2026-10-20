
import numpy as np


def global_magnitude_masks(weight_parameters, sparsity):
    """
    Create binary masks for global magnitude pruning.

    Args:
        weight_parameters: Iterable of weight Parameter objects.
        sparsity: Fraction of weights to prune, in [0, 1).

    Returns:
        A list of binary NumPy masks, one per weight parameter.
        0 means pruned; 1 means retained.
    """
    parameters = list(weight_parameters)

    if not parameters:
        raise ValueError("At least one weight parameter is required")

    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Sparsity must be in the range [0, 1)")

    for parameter in parameters:
        if not np.all(np.isfinite(parameter.data)):
            raise ValueError("Weights must contain only finite values")

    total_weights = sum(parameter.data.size for parameter in parameters)
    prune_count = int(sparsity * total_weights)

    masks = [
        np.ones(parameter.data.shape, dtype=bool)
        for parameter in parameters
    ]

    if prune_count == 0:
        return masks

    # Flatten all weight magnitudes into one global array.
    magnitudes = np.concatenate([
        np.abs(parameter.data).ravel()
        for parameter in parameters
    ])

    # Select exactly the requested number of smallest-magnitude weights.
    prune_indices = np.argsort(magnitudes, kind="stable")[:prune_count]

    # Map the global indices back to each individual weight matrix.
    offset = 0

    for mask, parameter in zip(masks, parameters):
        size = parameter.data.size
        local_indices = prune_indices[
            (prune_indices >= offset) &
            (prune_indices < offset + size)
        ] - offset

        mask.ravel()[local_indices] = False
        offset += size

    return masks


def apply_masks(weight_parameters, masks):
    """Set pruned weights to zero in place."""
    parameters = list(weight_parameters)

    if len(parameters) != len(masks):
        raise ValueError("Number of parameters and masks must match")

    for parameter, mask in zip(parameters, masks):
        if parameter.data.shape != mask.shape:
            raise ValueError("Mask shape must match parameter shape")

        parameter.data[~mask] = 0.0
