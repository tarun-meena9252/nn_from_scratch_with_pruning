
import numpy as np

from nn.models.sequential import Sequential
from nn.layers.linear import Linear
from nn.activations.functions import ReLU


def make_model():
    return Sequential([
        Linear(4, 5, seed=1),
        ReLU(),
        Linear(5, 3, seed=2),
    ])


def test_forward_shape():
    model = make_model()

    x = np.random.default_rng(10).normal(size=(8, 4))
    output = model.forward(x)

    assert output.shape == (8, 3)


def test_parameters_are_collected():
    model = make_model()
    params = model.parameters()

    # First Linear: 4*5 weights + 5 biases.
    # Second Linear: 5*3 weights + 3 biases.
    assert len(params) == 4
    assert sum(p.size for p in params) == 43


def test_backward_shape():
    model = make_model()

    x = np.random.default_rng(10).normal(size=(8, 4))
    logits = model.forward(x)

    grad_output = np.ones_like(logits)
    grad_input = model.backward(grad_output)

    assert grad_input.shape == x.shape

    for parameter in model.parameters():
        assert parameter.grad.shape == parameter.shape


def test_zero_grad():
    model = make_model()

    x = np.ones((2, 4))
    logits = model.forward(x)
    model.backward(np.ones_like(logits))

    model.zero_grad()

    for parameter in model.parameters():
        np.testing.assert_array_equal(
            parameter.grad,
            np.zeros_like(parameter.grad)
        )


def test_predict():
    model = make_model()

    x = np.ones((6, 4))
    predictions = model.predict(x)

    assert predictions.shape == (6,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)


def test_empty_model_rejected():
    try:
        Sequential([])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
