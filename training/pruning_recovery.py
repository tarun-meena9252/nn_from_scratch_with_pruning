
from pathlib import Path

import numpy as np
from nn.layers.linear import Linear
from training.trainer import train_one_epoch, evaluate


class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        min_delta=1e-4,
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def _get_lr(self):
        for name in ("lr", "learning_rate"):
            if hasattr(self.optimizer, name):
                return getattr(self.optimizer, name)
        raise AttributeError(
            "Could not find optimizer learning-rate attribute. "
            "Check nn/optimizers/adam.py."
        )

    def _set_lr(self, value):
        for name in ("lr", "learning_rate"):
            if hasattr(self.optimizer, name):
                setattr(self.optimizer, name, value)
                return
        raise AttributeError("Could not update optimizer learning rate.")

    def step(self, validation_loss):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            old_lr = self._get_lr()
            new_lr = max(self.min_lr, old_lr * self.factor)

            if new_lr < old_lr:
                self._set_lr(new_lr)
                print(
                    f"Learning rate reduced: "
                    f"{old_lr:.2e} -> {new_lr:.2e}"
                )

            self.bad_epochs = 0
            return True

        return False


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, validation_loss):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(model, path):
    """Save the model's Linear layer parameters."""
    state = {}

    for index, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            state[f"layer_{index}_W"] = layer.W.data.copy()
            state[f"layer_{index}_b"] = layer.b.data.copy()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)


def load_checkpoint(model, path):
    """Restore parameters into the matching model architecture."""
    with np.load(path) as checkpoint:
        for index, layer in enumerate(model.layers):
            if not isinstance(layer, Linear):
                continue

            w_key = f"layer_{index}_W"
            b_key = f"layer_{index}_b"

            if (
                checkpoint[w_key].shape != layer.W.data.shape
                or checkpoint[b_key].shape != layer.b.data.shape
            ):
                raise ValueError(
                    f"Checkpoint shape mismatch at layer {index}"
                )

            layer.W.data[...] = checkpoint[w_key]
            layer.b.data[...] = checkpoint[b_key]


def fine_tune_with_recovery(
    model,
    loss_fn,
    optimizer,
    x_train,
    y_train,
    x_val,
    y_val,
    checkpoint_path,
    seed,
    max_epochs=15,
    batch_size=64,
):
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        min_delta=1e-4,
    )
    early_stopping = EarlyStopping(
        patience=5,
        min_delta=1e-4,
    )

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            x_train,
            y_train,
            batch_size=batch_size,
            seed=seed + epoch,
        )

        val_metrics = evaluate(model, loss_fn, x_val, y_val)
        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, checkpoint_path)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy'] * 100:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"LR: {optimizer.learning_rate:.2e}"
        )

        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    load_checkpoint(model, checkpoint_path)

    best_val_metrics = evaluate(model, loss_fn, x_val, y_val)

    print(
        f"Restored best checkpoint from epoch {best_epoch} | "
        f"Val accuracy: "
        f"{best_val_metrics['accuracy'] * 100:.2f}%"
    )

    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_metrics["accuracy"],
    }
