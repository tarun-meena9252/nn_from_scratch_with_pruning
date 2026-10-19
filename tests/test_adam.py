
import numpy as np
import pytest

from nn.core.parameter import Parameter
from nn.optimizers.adam import Adam


def test_adam_updates_parameter():
    parameter = Parameter(np.array([2.0, 4.0]))
    parameter.grad[:] = [0.5, 1.0]

    optimizer = Adam([parameter], learning_rate=0.001)
    optimizer.step()

    # At the first step, bias correction makes m_hat = grad
    # and v_hat = grad ** 2.
    expected = np.array([1.999, 3.999])

    np.testing.assert_allclose(
        parameter.data,
        expected,
        atol=1e-7,
    )


def test_adam_tracks_steps():
    parameter = Parameter(np.array([1.0]))
    optimizer = Adam([parameter])

    parameter.grad[:] = [0.5]
    optimizer.step()

    assert optimizer.t == 1

    parameter.grad[:] = [0.2]
    optimizer.step()

    assert optimizer.t == 2



def test_adam_preserves_and_updates_moments_between_steps():
    parameter = Parameter(np.array([1.0]))
    optimizer = Adam([parameter])

    parameter.grad[:] = [1.0]
    optimizer.step()

    first_moment = optimizer.m[0].copy()
    first_second_moment = optimizer.v[0].copy()

    assert optimizer.t == 1

    # A zero gradient should decay the moments, not erase them.
    parameter.zero_grad()
    optimizer.step()

    assert optimizer.t == 2

    np.testing.assert_allclose(
        optimizer.m[0],
        0.9 * first_moment,
    )

    np.testing.assert_allclose(
        optimizer.v[0],
        0.999 * first_second_moment,
    )



def test_adam_zero_grad():
    parameter = Parameter(np.array([1.0, 2.0]))
    parameter.grad[:] = [0.5, 0.7]

    optimizer = Adam([parameter])
    optimizer.zero_grad()

    np.testing.assert_array_equal(
        parameter.grad,
        np.zeros(2),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"beta1": 1.0},
        {"beta1": -0.1},
        {"beta2": 1.0},
        {"beta2": -0.1},
        {"epsilon": 0},
    ],
)
def test_adam_rejects_invalid_hyperparameters(kwargs):
    parameter = Parameter(np.array([1.0]))

    with pytest.raises(ValueError):
        Adam([parameter], **kwargs)
