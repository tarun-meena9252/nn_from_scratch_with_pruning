
import numpy as np
import pytest

from nn.core.parameter import Parameter
from nn.optimizers.sgd import SGD


def test_sgd_updates_parameter():
    parameter = Parameter(np.array([2.0, 4.0]))
    parameter.grad[:] = [0.5, 1.0]

    optimizer = SGD([parameter], learning_rate=0.1)
    optimizer.step()

    np.testing.assert_allclose(
        parameter.data,
        [1.95, 3.9],
    )


def test_sgd_zero_grad():
    parameter = Parameter(np.array([2.0, 4.0]))
    parameter.grad[:] = [0.5, 1.0]

    optimizer = SGD([parameter])
    optimizer.zero_grad()

    np.testing.assert_array_equal(
        parameter.grad,
        [0.0, 0.0],
    )


def test_sgd_rejects_nonpositive_learning_rate():
    parameter = Parameter(np.array([1.0]))

    with pytest.raises(ValueError):
        SGD([parameter], learning_rate=0)

    with pytest.raises(ValueError):
        SGD([parameter], learning_rate=-0.1)


def test_sgd_does_not_update_without_gradients():
    parameter = Parameter(np.array([2.0, 4.0]))
    original = parameter.data.copy()

    optimizer = SGD([parameter], learning_rate=0.1)
    optimizer.step()

    np.testing.assert_array_equal(parameter.data, original)
