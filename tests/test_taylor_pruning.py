
import numpy as np
import pytest

from pruning.taylor import (
    taylor_weight_score,
    taylor_neuron_scores,
)

from nn.layers.linear import Linear
from nn.models.sequential import Sequential
from pruning.structured import structured_prune_model


def test_taylor_weight_score():
    weights = np.array([
        [2.0, -3.0],
        [0.5, 4.0],
    ])

    gradients = np.array([
        [0.1, -0.2],
        [-0.4, 0.0],
    ])

    scores = taylor_weight_score(weights, gradients)

    expected = np.array([
        [0.2, 0.6],
        [0.2, 0.0],
    ])

    np.testing.assert_allclose(scores, expected)


def test_taylor_neuron_scores_sums_incoming_weight_scores():
    scores = np.array([
        [0.2, 0.6, 0.1],
        [0.3, 0.0, 0.4],
    ])

    neuron_scores = taylor_neuron_scores(scores)

    np.testing.assert_allclose(
        neuron_scores,
        [0.5, 0.6, 0.5],
    )


def test_taylor_weight_score_rejects_mismatched_shapes():
    weights = np.zeros((2, 3))
    gradients = np.zeros((2, 2))

    with pytest.raises(ValueError):
        taylor_weight_score(weights, gradients)


def test_taylor_neuron_scores_requires_2d_input():
    with pytest.raises(ValueError):
        taylor_neuron_scores(np.array([1.0, 2.0]))


def test_taylor_structured_pruning_uses_taylor_scores():
    hidden = Linear(2, 3, initializer="zeros")
    output = Linear(3, 2, initializer="zeros")

    # Neuron 0 has the largest magnitude, but neuron 1 has
    # the highest Taylor importance.
    hidden.W.data[...] = np.array([
        [10.0, 1.0, 2.0],
        [10.0, 1.0, 2.0],
    ])
    hidden.b.data[...] = np.array([[0.1, 0.2, 0.3]])

    model = Sequential([hidden, output])

    taylor_scores = [
        np.array([
            [0.1, 3.0, 0.2],
            [0.1, 2.0, 0.2],
        ]),
        np.zeros_like(output.W.data),
    ]

    pruned = structured_prune_model(
        model,
        prune_fractions={0: 1 / 3},
        criterion="taylor",
        importance_scores=taylor_scores,
    )

    pruned_hidden = [
        layer for layer in pruned.layers if isinstance(layer, Linear)
    ][0]

    # Taylor neuron scores are [0.2, 5.0, 0.4].
    # Neurons 1 and 2 survive.
    assert pruned_hidden.W.data.shape == (2, 2)
    np.testing.assert_array_equal(
        pruned_hidden.W.data,
        hidden.W.data[:, [1, 2]],
    )


def test_taylor_pruning_requires_scores():
    model = Sequential([
        Linear(2, 3, initializer="zeros"),
        Linear(3, 2, initializer="zeros"),
    ])

    with pytest.raises(ValueError):
        structured_prune_model(
            model,
            {0: 0.25},
            criterion="taylor",
        )


def test_structured_pruning_rejects_invalid_criterion():
    model = Sequential([
        Linear(2, 3, initializer="zeros"),
        Linear(3, 2, initializer="zeros"),
    ])

    with pytest.raises(ValueError):
        structured_prune_model(
            model,
            {0: 0.25},
            criterion="unknown",
        )


def test_taylor_scores_ignore_removed_input_connections():
    first = Linear(2, 2, initializer="zeros")
    second = Linear(2, 2, initializer="zeros")
    output = Linear(2, 1, initializer="zeros")

    first.W.data[...] = np.array([
        [10.0, 1.0],
        [10.0, 1.0],
    ])

    second.W.data[...] = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
    ])

    model = Sequential([first, second, output])

    # First hidden layer: neuron 0 has higher Taylor importance.
    first_scores = np.array([
        [10.0, 0.1],
        [10.0, 0.1],
    ])

    # Second hidden layer: neuron 0 looks more important when
    # both inputs count, but neuron 1 wins using only input 0.
    second_scores = np.array([
        [0.1, 5.0],
        [100.0, 0.1],
    ])

    output_scores = np.zeros_like(output.W.data)

    pruned = structured_prune_model(
        model,
        prune_fractions={0: 0.5, 1: 0.5},
        criterion="taylor",
        importance_scores=[
            first_scores,
            second_scores,
            output_scores,
        ],
    )

    linear_layers = [
        layer for layer in pruned.layers
        if isinstance(layer, Linear)
    ]

    # First layer keeps original neuron 0.
    np.testing.assert_array_equal(
        linear_layers[0].W.data,
        first.W.data[:, [0]],
    )

    # Second layer's scores after slicing are:
    # neuron 0 = 0.1, neuron 1 = 5.0.
    # Therefore, neuron 1 should survive.
    np.testing.assert_array_equal(
        linear_layers[1].W.data,
        second.W.data[[0], :][:, [1]],
    )