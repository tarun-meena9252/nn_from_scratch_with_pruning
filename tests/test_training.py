from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nn.activations.functions import ReLU
from nn.layers.linear import Linear
from nn.losses.cross_entropy import CrossEntropyLoss
from nn.models.sequential import Sequential
from nn.optimizers.adam import Adam
from training.pruning_recovery import (
    EarlyStopping,
    ReduceLROnPlateau,
    save_checkpoint,
    load_checkpoint,
    fine_tune_with_recovery,
)


def build_model(seed=1):
    return Sequential([
        Linear(4, 5, initializer="he_normal", seed=seed),
        ReLU(),
        Linear(5, 3, initializer="xavier_uniform", seed=seed + 1),
    ])


def make_optimizer(model, learning_rate=0.01):
    return Adam(model.parameters(), learning_rate=learning_rate)


def get_parameters(model):
    params = []

    for layer in model.layers:
        if isinstance(layer, Linear):
            params.extend([
                layer.W.data.copy(),
                layer.b.data.copy(),
            ])

    return params


# -------------------------
# ReduceLROnPlateau tests
# -------------------------

def test_scheduler_tracks_improving_loss():
    model = build_model()
    optimizer = make_optimizer(model)
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=2,
        min_delta=0.01,
    )

    assert scheduler.step(1.0) is False
    assert scheduler.best_loss == 1.0
    assert scheduler.bad_epochs == 0

    assert scheduler.step(0.95) is False
    assert scheduler.best_loss == 0.95
    assert scheduler.bad_epochs == 0


def test_scheduler_reduces_learning_rate_after_patience():
    model = build_model()
    optimizer = make_optimizer(model, learning_rate=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_delta=0.001,
    )

    scheduler.step(1.0)

    assert scheduler.step(1.0) is False
    assert scheduler.step(1.0) is True

    assert optimizer.learning_rate == pytest.approx(0.005)
    assert scheduler.bad_epochs == 0


def test_scheduler_does_not_reduce_below_min_lr():
    model = build_model()
    optimizer = make_optimizer(model, learning_rate=0.00015)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=1,
        min_lr=0.0001,
    )

    scheduler.step(1.0)
    assert scheduler.step(1.0) is True
    assert optimizer.learning_rate == pytest.approx(0.0001)

    # At min_lr, the scheduler cannot reduce the learning rate further.
    assert scheduler.step(1.0) is True
    assert optimizer.learning_rate == pytest.approx(0.0001)


def test_scheduler_accepts_learning_rate_attribute():
    class LrOptimizer:
        lr = 0.02

    optimizer = LrOptimizer()
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=1,
    )

    scheduler.step(1.0)
    assert scheduler.step(1.0) is True
    assert optimizer.lr == pytest.approx(0.01)


def test_scheduler_raises_if_learning_rate_missing():
    class OptimizerWithoutLr:
        pass

    scheduler = ReduceLROnPlateau(
        OptimizerWithoutLr(),
        patience=1,
    )

    scheduler.step(1.0)

    with pytest.raises(AttributeError, match="learning-rate attribute"):
        scheduler.step(1.0)


# -------------------------
# EarlyStopping tests
# -------------------------

def test_early_stopping_tracks_improvement():
    stopper = EarlyStopping(patience=3, min_delta=0.01)

    assert stopper.step(1.0) is False
    assert stopper.best_loss == 1.0
    assert stopper.bad_epochs == 0

    assert stopper.step(0.95) is False
    assert stopper.best_loss == 0.95
    assert stopper.bad_epochs == 0


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=3, min_delta=0.001)

    assert stopper.step(1.0) is False
    assert stopper.step(1.0) is False
    assert stopper.step(1.0) is False
    assert stopper.step(1.0) is True


def test_early_stopping_resets_counter_on_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.001)

    stopper.step(1.0)
    stopper.step(1.0)

    assert stopper.bad_epochs == 1

    assert stopper.step(0.9) is False
    assert stopper.bad_epochs == 0


# -------------------------
# Checkpoint tests
# -------------------------

def test_save_checkpoint_creates_file_and_parent_directory(tmp_path):
    model = build_model()
    checkpoint_path = tmp_path / "nested" / "model.npz"

    save_checkpoint(model, checkpoint_path)

    assert checkpoint_path.exists()


def test_checkpoint_restores_weights_and_biases(tmp_path):
    model = build_model()
    checkpoint_path = tmp_path / "model.npz"

    original_params = get_parameters(model)
    save_checkpoint(model, checkpoint_path)

    # Change all parameters.
    for layer in model.layers:
        if isinstance(layer, Linear):
            layer.W.data += 10.0
            layer.b.data -= 5.0

    load_checkpoint(model, checkpoint_path)
    restored_params = get_parameters(model)

    for original, restored in zip(original_params, restored_params):
        np.testing.assert_array_equal(restored, original)


def test_checkpoint_does_not_change_non_linear_layers(tmp_path):
    model = build_model()
    checkpoint_path = tmp_path / "model.npz"

    save_checkpoint(model, checkpoint_path)

    relu_layers_before = [
        layer for layer in model.layers if isinstance(layer, ReLU)
    ]

    load_checkpoint(model, checkpoint_path)

    relu_layers_after = [
        layer for layer in model.layers if isinstance(layer, ReLU)
    ]

    assert len(relu_layers_before) == len(relu_layers_after)


def test_load_checkpoint_rejects_shape_mismatch(tmp_path):
    source_model = build_model(seed=1)
    checkpoint_path = tmp_path / "model.npz"
    save_checkpoint(source_model, checkpoint_path)

    incompatible_model = Sequential([
        Linear(4, 6, initializer="he_normal", seed=1),
        ReLU(),
        Linear(6, 3, initializer="xavier_uniform", seed=2),
    ])

    with pytest.raises(ValueError, match="Checkpoint shape mismatch"):
        load_checkpoint(incompatible_model, checkpoint_path)


def test_load_checkpoint_rejects_missing_file(tmp_path):
    model = build_model()

    with pytest.raises(FileNotFoundError):
        load_checkpoint(model, tmp_path / "missing.npz")


# -------------------------
# Recovery workflow tests
# -------------------------

def test_recovery_returns_expected_metrics_and_checkpoint(tmp_path):
    rng = np.random.default_rng(42)

    x_train = rng.normal(size=(48, 4))
    y_train = rng.integers(0, 3, size=48)

    x_val = rng.normal(size=(18, 4))
    y_val = rng.integers(0, 3, size=18)

    model = build_model()
    optimizer = make_optimizer(model, learning_rate=0.005)
    loss_fn = CrossEntropyLoss()
    checkpoint_path = tmp_path / "recovery" / "best.npz"

    returned_model, metrics = fine_tune_with_recovery(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        checkpoint_path=checkpoint_path,
        seed=42,
        max_epochs=2,
        batch_size=12,
    )

    assert returned_model is model
    assert checkpoint_path.exists()

    assert set(metrics) == {
        "best_epoch",
        "best_val_loss",
        "best_val_accuracy",
    }

    assert 1 <= metrics["best_epoch"] <= 2
    assert np.isfinite(metrics["best_val_loss"])
    assert np.isfinite(metrics["best_val_accuracy"])
    assert 0.0 <= metrics["best_val_accuracy"] <= 1.0


def test_recovery_restores_best_checkpoint(tmp_path):
    rng = np.random.default_rng(123)

    x_train = rng.normal(size=(24, 4))
    y_train = rng.integers(0, 3, size=24)
    x_val = rng.normal(size=(12, 4))
    y_val = rng.integers(0, 3, size=12)

    model = build_model()
    optimizer = make_optimizer(model)
    loss_fn = CrossEntropyLoss()
    checkpoint_path = tmp_path / "best.npz"

    epoch_parameters = []
    validation_losses = iter([0.5, 0.8, 0.9])

    def fake_train(*args, **kwargs):
        # Simulate parameter updates during training.
        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.W.data += 0.1
        return {"loss": 1.0, "accuracy": 0.5}

    def fake_evaluate(*args, **kwargs):
        try:
            loss = next(validation_losses)
        except StopIteration:
            # Final evaluation after restoring the best checkpoint.
            loss = 0.5

        if loss == 0.5 and not epoch_parameters:
            epoch_parameters.append(get_parameters(model))

        return {"loss": loss, "accuracy": 0.5}

    with (
        patch(
            "training.pruning_recovery.train_one_epoch",
            side_effect=fake_train,
        ),
        patch(
            "training.pruning_recovery.evaluate",
            side_effect=fake_evaluate,
        ),
    ):
        returned_model, metrics = fine_tune_with_recovery(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            checkpoint_path=checkpoint_path,
            seed=7,
            max_epochs=3,
            batch_size=8,
        )

    assert returned_model is model
    assert metrics["best_epoch"] == 1
    assert metrics["best_val_loss"] == pytest.approx(0.5)
    assert checkpoint_path.exists()
    assert len(epoch_parameters) == 1

    restored_parameters = get_parameters(model)
    best_parameters = epoch_parameters[0]

    # The model must match epoch 1, not the final epoch.
    for expected, restored in zip(best_parameters, restored_parameters):
        np.testing.assert_allclose(restored, expected, atol=1e-7)