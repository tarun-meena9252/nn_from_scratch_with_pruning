
import numpy as np

from nn.layers.linear import Linear
from tests.gradient_check_utils import (
    numerical_gradient,
    relative_error,
)

from nn.models.sequential import Sequential
from nn.activations.functions import ReLU
from nn.losses.cross_entropy import CrossEntropyLoss


def test_linear_gradient_check():
    rng = np.random.default_rng(42)

    x = rng.normal(size=(3, 4))
    target = rng.normal(size=(3, 2))

    layer = Linear(
        4,
        2,
        initializer="xavier_uniform",
        seed=10,
    )

    def loss_function():
        prediction = layer.forward(x)
        return 0.5 * np.mean((prediction - target) ** 2)

    # Analytical gradients from backpropagation.
    prediction = layer.forward(x)
    grad_output = (prediction - target) / prediction.size

    layer.zero_grad()
    layer.backward(grad_output)

    analytic_w = layer.W.grad.copy()
    analytic_b = layer.b.grad.copy()

    # Numerical gradients for weights and biases.
    numerical_w = numerical_gradient(loss_function, layer.W)
    numerical_b = numerical_gradient(loss_function, layer.b)

    error_w = np.max(relative_error(analytic_w, numerical_w))
    error_b = np.max(relative_error(analytic_b, numerical_b))

    assert error_w < 1e-6, f"Weight gradient error: {error_w}"
    assert error_b < 1e-6, f"Bias gradient error: {error_b}"


def test_network_gradient_check():
    rng = np.random.default_rng(7)

    x = rng.normal(size=(4, 3))
    y = np.array([0, 2, 1, 0])

    model = Sequential([
        Linear(3, 4, initializer="he_normal", seed=1),
        ReLU(),
        Linear(4, 3, initializer="xavier_uniform", seed=2),
    ])

    loss_fn = CrossEntropyLoss()

    def loss_function():
        logits = model.forward(x)
        return loss_fn.forward(logits, y)

    # Analytical gradients.
    logits = model.forward(x)
    loss_fn.forward(logits, y)

    model.zero_grad()
    grad_logits = loss_fn.backward()
    model.backward(grad_logits)

    parameters = model.parameters()
    analytic_grads = [p.grad.copy() for p in parameters]

    # Numerical gradients.
    for parameter, analytic_grad in zip(parameters, analytic_grads):
        numerical_grad = numerical_gradient(
            loss_function,
            parameter,
        )

        error = np.max(
            relative_error(analytic_grad, numerical_grad)
        )

        assert error < 1e-6, (
            f"Gradient check failed for parameter shape "
            f"{parameter.shape}: relative error = {error}"
        )



def test_linear_gradients_are_finite():
    """Forward outputs, loss, and parameter gradients must be finite."""
    rng = np.random.default_rng(101)

    x = rng.normal(size=(4, 3))
    target = rng.normal(size=(4, 2))

    layer = Linear(3, 2, initializer="xavier_uniform", seed=10)

    prediction = layer.forward(x)
    loss = 0.5 * np.mean((prediction - target) ** 2)

    grad_output = (prediction - target) / prediction.size
    layer.zero_grad()
    layer.backward(grad_output)

    assert np.all(np.isfinite(prediction))
    assert np.isfinite(loss)

    assert layer.W.grad is not None
    assert layer.b.grad is not None

    assert np.all(np.isfinite(layer.W.grad))
    assert np.all(np.isfinite(layer.b.grad))


def test_network_gradients_are_finite():
    """Check numerical stability across a complete network."""
    rng = np.random.default_rng(102)

    x = rng.normal(size=(5, 3))
    y = np.array([0, 1, 2, 1, 0])

    model = Sequential([
        Linear(3, 8, initializer="he_normal", seed=1),
        ReLU(),
        Linear(8, 3, initializer="xavier_uniform", seed=2),
    ])

    loss_fn = CrossEntropyLoss()

    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)

    model.zero_grad()
    grad_logits = loss_fn.backward()
    model.backward(grad_logits)

    assert np.all(np.isfinite(logits))
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad_logits))

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert np.all(np.isfinite(parameter.grad)), (
            f"NaN or Inf gradient in parameter shape "
            f"{parameter.shape}"
        )


def test_gradient_shapes_match_parameter_shapes():
    """Every trainable parameter gradient must have the right shape."""
    rng = np.random.default_rng(103)

    x = rng.normal(size=(4, 3))
    y = np.array([0, 1, 2, 0])

    model = Sequential([
        Linear(3, 5, initializer="he_normal", seed=1),
        ReLU(),
        Linear(5, 3, initializer="xavier_uniform", seed=2),
    ])

    loss_fn = CrossEntropyLoss()

    logits = model.forward(x)
    loss_fn.forward(logits, y)

    model.zero_grad()
    model.backward(loss_fn.backward())

    for parameter in model.parameters():
        assert parameter.grad.shape == parameter.shape
        assert parameter.grad.size == parameter.data.size


def test_linear_zero_upstream_gradient_produces_zero_gradients():
    """A zero upstream gradient should produce zero W and b gradients."""
    rng = np.random.default_rng(104)

    x = rng.normal(size=(4, 3))
    layer = Linear(3, 2, initializer="xavier_uniform", seed=10)

    layer.forward(x)
    layer.zero_grad()

    grad_output = np.zeros((4, 2))
    layer.backward(grad_output)

    np.testing.assert_allclose(layer.W.grad, 0.0, atol=1e-12)
    np.testing.assert_allclose(layer.b.grad, 0.0, atol=1e-12)


def test_network_gradients_are_repeatable():
    """Identical inputs and parameters should produce identical gradients."""
    rng = np.random.default_rng(105)

    x = rng.normal(size=(4, 3))
    y = np.array([0, 1, 2, 0])

    model = Sequential([
        Linear(3, 5, initializer="he_normal", seed=1),
        ReLU(),
        Linear(5, 3, initializer="xavier_uniform", seed=2),
    ])

    loss_fn = CrossEntropyLoss()

    def calculate_gradients():
        logits = model.forward(x)
        loss_fn.forward(logits, y)

        model.zero_grad()
        model.backward(loss_fn.backward())

        return [
            parameter.grad.copy()
            for parameter in model.parameters()
        ]

    first = calculate_gradients()
    second = calculate_gradients()

    for grad1, grad2 in zip(first, second):
        np.testing.assert_allclose(
            grad1, grad2, rtol=1e-12, atol=1e-12
        )


def test_relative_error_detects_incorrect_gradient():
    """The gradient-check comparison should detect a wrong gradient."""
    analytic = np.array([0.25, -0.5, 1.0])
    incorrect = np.array([0.25, -0.1, 1.0])

    errors = relative_error(analytic, incorrect)

    assert np.max(errors) > 1e-6

