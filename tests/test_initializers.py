
import numpy as np
import pytest

from nn.initializers.weights import (
    zeros,
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
)


def test_zeros():
    weights = zeros((4, 3))

    assert weights.shape == (4, 3)
    np.testing.assert_array_equal(weights, np.zeros((4, 3)))


@pytest.mark.parametrize(
    "initializer",
    [xavier_uniform, xavier_normal, he_uniform, he_normal],
)
def test_initializer_shape_and_finite_values(initializer):
    weights = initializer(20, 10, rng=np.random.default_rng(42))

    assert weights.shape == (20, 10)
    assert np.all(np.isfinite(weights))


def test_same_seed_produces_same_weights():
    first = he_normal(20, 10, rng=np.random.default_rng(123))
    second = he_normal(20, 10, rng=np.random.default_rng(123))

    np.testing.assert_array_equal(first, second)


def test_initializers_reject_invalid_dimensions():
    with pytest.raises(ValueError):
        he_normal(0, 10)

    with pytest.raises(ValueError):
        xavier_uniform(10, -1)


def test_he_normal_variance_is_reasonable():
    weights = he_normal(1000, 1000, rng=np.random.default_rng(42))
    expected_variance = 2.0 / 1000

    assert np.var(weights) == pytest.approx(
        expected_variance,
        rel=0.1,
    )
