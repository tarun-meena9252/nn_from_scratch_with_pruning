
import numpy as np
import pytest

from nn.losses.cross_entropy import CrossEntropyLoss


def test_loss_is_positive():
    loss_fn = CrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 0.3, 1.5]
    ])
    targets = np.array([0, 2])

    loss = loss_fn.forward(logits, targets)

    assert np.isfinite(loss)
    assert loss > 0


def test_gradient_shape_and_values():
    loss_fn = CrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.2, 0.3, 1.5]
    ])
    targets = np.array([0, 2])

    loss_fn.forward(logits, targets)
    grad = loss_fn.backward()

    assert grad.shape == logits.shape
    np.testing.assert_allclose(
        grad.sum(axis=1), [0.0, 0.0], atol=1e-12
    )


def test_confident_correct_predictions_have_low_loss():
    loss_fn = CrossEntropyLoss()

    logits = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0]
    ])
    targets = np.array([0, 1])

    loss = loss_fn.forward(logits, targets)

    assert loss < 0.001


def test_extreme_logits_are_stable():
    loss_fn = CrossEntropyLoss()

    logits = np.array([
        [1000.0, 0.0, -1000.0]
    ])
    targets = np.array([0])

    loss = loss_fn.forward(logits, targets)
    grad = loss_fn.backward()

    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


def test_invalid_targets_raise():
    loss_fn = CrossEntropyLoss()

    with pytest.raises(ValueError):
        loss_fn.forward(
            np.array([[1.0, 2.0]]),
            np.array([2])
        )


def test_backward_before_forward_raises():
    with pytest.raises(RuntimeError):
        CrossEntropyLoss().backward()


def test_numerical_gradient_check():
    logits = np.array([
        [0.2, -0.3, 0.8],
        [0.5, 0.1, -0.2]
    ])
    targets = np.array([2, 0])

    loss_fn = CrossEntropyLoss()
    loss_fn.forward(logits, targets)
    analytical = loss_fn.backward()

    numerical = np.zeros_like(logits)
    eps = 1e-6

    for index in np.ndindex(logits.shape):
        original = logits[index]

        logits[index] = original + eps
        plus = CrossEntropyLoss().forward(logits, targets)

        logits[index] = original - eps
        minus = CrossEntropyLoss().forward(logits, targets)

        logits[index] = original

        numerical[index] = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(
        analytical, numerical, rtol=1e-5, atol=1e-7
    )
