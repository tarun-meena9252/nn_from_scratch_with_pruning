
import numpy as np

from nn.core.parameter import Parameter
from nn.optimizers.backtracking import BacktrackingLineSearch


def test_backtracking_accepts_a_descent_step():
    parameter = Parameter(np.array([2.0]))

    def loss():
        return float(parameter.data[0] ** 2)

    parameter.grad[:] = [4.0]

    optimizer = BacktrackingLineSearch(
        [parameter],
        initial_step_size=1.0,
    )

    accepted = optimizer.step(loss)

    assert accepted
    assert parameter.data[0] ** 2 < 4.0
    assert optimizer.last_step_size > 0


def test_backtracking_restores_parameters_if_no_step_accepted():
    parameter = Parameter(np.array([2.0]))
    original = parameter.data.copy()

    # This direction is not a descent direction for this loss.
    def loss():
        return float(parameter.data[0] ** 2)

    parameter.grad[:] = [-4.0]

    optimizer = BacktrackingLineSearch(
        [parameter],
        initial_step_size=1.0,
        max_iterations=3,
    )

    accepted = optimizer.step(loss)

    assert not accepted
    np.testing.assert_array_equal(parameter.data, original)
    assert optimizer.last_step_size == 0.0


def test_backtracking_shrinks_step_when_needed():
    parameter = Parameter(np.array([2.0]))

    def loss():
        return float(parameter.data[0] ** 2)

    parameter.grad[:] = [4.0]

    optimizer = BacktrackingLineSearch(
        [parameter],
        initial_step_size=1.0,
        shrink_factor=0.5,
    )

    assert optimizer.step(loss)
    assert optimizer.last_step_size < 1.0


def test_zero_gradient_does_not_change_parameters():
    parameter = Parameter(np.array([2.0]))
    original = parameter.data.copy()

    optimizer = BacktrackingLineSearch([parameter])

    def loss():
        return float(parameter.data[0] ** 2)

    parameter.grad[:] = [0.0]

    accepted = optimizer.step(loss)

    assert not accepted
    np.testing.assert_array_equal(parameter.data, original)
