
import numpy as np
import pytest

from nn.core.parameter import Parameter
from nn.core.module import Module


def test_parameter_initialization():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    p = Parameter(data)

    np.testing.assert_array_equal(p.data, data)
    np.testing.assert_array_equal(p.grad, np.zeros(3))
    assert p.shape == (3,)
    assert p.size == 3


def test_zero_grad():
    p = Parameter(np.array([1.0, 2.0]))
    p.grad[:] = [0.5, -0.2]

    p.zero_grad()

    np.testing.assert_array_equal(p.grad, [0.0, 0.0])
    np.testing.assert_array_equal(p.data, [1.0, 2.0])


def test_integer_parameters_rejected():
    with pytest.raises(TypeError):
        Parameter(np.array([1, 2, 3]))


def test_base_module_requires_forward_implementation():
    module = Module()

    with pytest.raises(NotImplementedError):
        module.forward(np.array([1.0]))
