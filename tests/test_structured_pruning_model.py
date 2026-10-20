
import numpy as np

from nn.layers.linear import Linear
from nn.models.sequential import Sequential
from pruning.structured import structured_prune_model
from nn.activations.functions import ReLU


def build_model():
    return Sequential([
        Linear(4, 6, initializer="he_normal", seed=1),
        ReLU(),
        Linear(6, 5, initializer="he_normal", seed=2),
        ReLU(),
        Linear(5, 3, initializer="xavier_uniform", seed=3),
    ])


def test_structural_pruning_reduces_layer_dimensions():
    model = build_model()
    pruned = structured_prune_model(
        model, {0: 1 / 3, 1: 0.4}
    )

    linear_layers = [
        layer for layer in pruned.layers
        if isinstance(layer, Linear)
    ]

    assert linear_layers[0].W.shape == (4, 4)
    assert linear_layers[1].W.shape == (4, 3)
    assert linear_layers[2].W.shape == (3, 3)


def test_structural_pruning_preserves_output_shape():
    model = build_model()
    pruned = structured_prune_model(model, {0: 0.5, 1: 0.4})

    x = np.random.default_rng(42).normal(size=(8, 4))
    output = pruned.forward(x)

    assert output.shape == (8, 3)


def test_original_model_is_unchanged():
    model = build_model()
    original_shapes = [
        layer.W.shape for layer in model.layers
        if isinstance(layer, Linear)
    ]

    structured_prune_model(model, {0: 0.5})

    new_shapes = [
        layer.W.shape for layer in model.layers
        if isinstance(layer, Linear)
    ]

    assert new_shapes == original_shapes


def test_pruned_model_has_fewer_parameters():
    model = build_model()
    pruned = structured_prune_model(model, {0: 0.5, 1: 0.4})

    def count_parameters(network):
        return sum(
            parameter.data.size
            for parameter in network.parameters()
        )

    assert count_parameters(pruned) < count_parameters(model)
