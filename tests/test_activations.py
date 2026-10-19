import numpy as np
import pytest
from nn.activations.functions import ReLU, Sigmoid, Tanh, Softmax


def test_relu_forward_backward():
    layer = ReLU()
    x = np.array([[-2.0, 0.0, 3.0]])

    output = layer.forward(x)
    grad = layer.backward(np.ones_like(x))

    np.testing.assert_allclose(output, [[0.0, 0.0, 3.0]])
    np.testing.assert_allclose(grad, [[0.0, 0.0, 1.0]])


def test_sigmoid_forward_backward():
    layer = Sigmoid()
    x = np.array([[-2.0, 0.0, 2.0]])

    output = layer.forward(x)
    grad = layer.backward(np.ones_like(x))

    assert np.all(output > 0)
    assert np.all(output < 1)

    expected = output * (1.0 - output)
    np.testing.assert_allclose(grad, expected)


def test_sigmoid_numerical_stability():
    layer = Sigmoid()
    x = np.array([[-1000.0, 1000.0]])

    output = layer.forward(x)

    assert np.all(np.isfinite(output))
    np.testing.assert_allclose(output, [[0.0, 1.0]])


def test_tanh_forward_backward():
    layer = Tanh()
    x = np.array([[-1.0, 0.0, 1.0]])

    output = layer.forward(x)
    grad = layer.backward(np.ones_like(x))

    np.testing.assert_allclose(output, np.tanh(x))
    np.testing.assert_allclose(
        grad, 1.0 - output ** 2
    )


def test_softmax_probabilities():
    layer = Softmax()
    x = np.array([
        [1.0, 2.0, 3.0],
        [1000.0, 1001.0, 1002.0]
    ])

    output = layer.forward(x)

    assert output.shape == x.shape
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0)
    np.testing.assert_allclose(
        output.sum(axis=1),
        [1.0, 1.0]
    )


def test_softmax_backward():
    layer = Softmax()
    x = np.array([[0.2, -0.3, 0.8]])
    upstream = np.array([[0.5, -0.2, 0.7]])

    output = layer.forward(x)
    grad = layer.backward(upstream)

    # Check against the explicit softmax Jacobian.
    p = output[0]
    jacobian = np.diag(p) - np.outer(p, p)
    expected = upstream[0] @ jacobian

    np.testing.assert_allclose(grad[0], expected)


def test_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        ReLU().backward(np.ones((1, 2)))
