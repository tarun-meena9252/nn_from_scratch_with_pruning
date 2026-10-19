
import numpy as np
import pytest

from nn.layers.linear import Linear


@pytest.mark.parametrize(
    "initializer",
    [
        "zeros",
        "xavier_uniform",
        "xavier_normal",
        "he_uniform",
        "he_normal",
    ],
)
def test_linear_supports_all_initializers(initializer):
    layer = Linear(4, 3, initializer=initializer, seed=42)

    assert layer.W.shape == (4, 3)
    assert layer.b.shape == (1, 3)
    assert np.all(np.isfinite(layer.W.data))


def test_linear_initialization_is_reproducible():
    first = Linear(20, 10, initializer="he_normal", seed=42)
    second = Linear(20, 10, initializer="he_normal", seed=42)

    np.testing.assert_array_equal(first.W.data, second.W.data)


def test_linear_rejects_unknown_initializer():
    with pytest.raises(ValueError):
        Linear(4, 3, initializer="unknown")

def test_linear_forward():
    layer = Linear(3, 2, seed=42)

    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    output = layer.forward(x)

    assert output.shape == (2, 2)

    expected = x @ layer.W.data + layer.b.data
    np.testing.assert_allclose(output, expected)


def test_linear_backward_shapes():
    layer = Linear(3, 2, seed=42)

    x = np.ones((4, 3))
    output = layer.forward(x)

    grad_output = np.ones_like(output)
    grad_input = layer.backward(grad_output)

    assert grad_input.shape == (4, 3)
    assert layer.W.grad.shape == (3, 2)
    assert layer.b.grad.shape == (1, 2)


def test_linear_backward_values():
    layer = Linear(2, 1, seed=42)

    layer.W.data[:] = [[2.0], [3.0]]
    layer.b.data[:] = [[1.0]]

    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    output = layer.forward(x)
    grad_output = np.ones_like(output)

    grad_input = layer.backward(grad_output)

    np.testing.assert_allclose(
        layer.W.grad,
        [[4.0], [6.0]]
    )
    np.testing.assert_allclose(
        layer.b.grad,
        [[2.0]]
    )
    np.testing.assert_allclose(
        grad_input,
        [[2.0, 3.0], [2.0, 3.0]]
    )
