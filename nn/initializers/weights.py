
import numpy as np


def zeros(shape):
    """Initialize all weights to zero."""
    return np.zeros(shape, dtype=np.float64)


def xavier_uniform(in_features, out_features, rng=None):
    """Xavier initialization using a uniform distribution."""
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Feature dimensions must be positive")

    if rng is None:
        rng = np.random.default_rng()

    limit = np.sqrt(6.0 / (in_features + out_features))

    return rng.uniform(
        low=-limit,
        high=limit,
        size=(in_features, out_features),
    )


def xavier_normal(in_features, out_features, rng=None):
    """Xavier initialization using a normal distribution."""
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Feature dimensions must be positive")

    if rng is None:
        rng = np.random.default_rng()

    std = np.sqrt(2.0 / (in_features + out_features))

    return rng.normal(
        loc=0.0,
        scale=std,
        size=(in_features, out_features),
    )


def he_uniform(in_features, out_features, rng=None):
    """He initialization using a uniform distribution."""
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Feature dimensions must be positive")

    if rng is None:
        rng = np.random.default_rng()

    limit = np.sqrt(6.0 / in_features)

    return rng.uniform(
        low=-limit,
        high=limit,
        size=(in_features, out_features),
    )


def he_normal(in_features, out_features, rng=None):
    """He initialization using a normal distribution."""
    if in_features <= 0 or out_features <= 0:
        raise ValueError("Feature dimensions must be positive")

    if rng is None:
        rng = np.random.default_rng()

    std = np.sqrt(2.0 / in_features)

    return rng.normal(
        loc=0.0,
        scale=std,
        size=(in_features, out_features),
    )
