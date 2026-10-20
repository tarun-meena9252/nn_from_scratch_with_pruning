
import numpy as np
import pytest

from nn.layers.linear import Linear
from pruning.structured import (
    neuron_importance,
    select_neurons_to_keep,
)


def test_neuron_importance_uses_column_norms():
    layer = Linear(2, 3, initializer="zeros")
    layer.W.data[:] = np.array([
        [3.0, 0.0, 1.0],
        [4.0, 0.0, 2.0],
    ])

    scores = neuron_importance(layer)

    np.testing.assert_allclose(
        scores,
        [5.0, 0.0, np.sqrt(5.0)],
    )


def test_select_neurons_keeps_highest_scores():
    layer = Linear(2, 3, initializer="zeros")
    layer.W.data[:] = np.array([
        [3.0, 0.0, 1.0],
        [4.0, 0.0, 2.0],
    ])

    kept = select_neurons_to_keep(layer, 1 / 3)

    np.testing.assert_array_equal(kept, [0, 2])


def test_zero_pruning_keeps_all_neurons():
    layer = Linear(2, 3, initializer="zeros")

    kept = select_neurons_to_keep(layer, 0.0)

    np.testing.assert_array_equal(kept, [0, 1, 2])


def test_invalid_prune_fraction_raises():
    layer = Linear(2, 3, initializer="zeros")

    with pytest.raises(ValueError):
        select_neurons_to_keep(layer, 1.0)
