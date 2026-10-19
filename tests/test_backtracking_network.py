import numpy as np

from nn.models.sequential import Sequential
from nn.layers.linear import Linear
from nn.activations.functions import ReLU
from nn.losses.cross_entropy import CrossEntropyLoss
from nn.optimizers.backtracking import BacktrackingLineSearch


def test_backtracking_reduces_network_loss():
    rng = np.random.default_rng(42)

    x = rng.normal(size=(8, 4))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

    model = Sequential([
        Linear(4, 6, initializer="he_normal", seed=1),
        ReLU(),
        Linear(6, 3, initializer="xavier_uniform", seed=2),
    ])

    loss_fn = CrossEntropyLoss()

    def loss_function():
        return loss_fn.forward(model.forward(x), y)

    # Calculate the initial loss and analytical gradients.
    initial_loss = loss_function()

    model.zero_grad()
    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())

    optimizer = BacktrackingLineSearch(
        model.parameters(),
        initial_step_size=1.0,
        max_iterations=30,
    )

    accepted = optimizer.step(loss_function)
    final_loss = loss_function()

    assert accepted
    assert final_loss < initial_loss
    assert optimizer.last_step_size > 0
