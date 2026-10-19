
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
