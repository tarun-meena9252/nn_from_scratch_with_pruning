import os
from os.path import join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataLoader import MnistDataloader
from nn.activations.functions import ReLU
from nn.layers.linear import Linear
from nn.losses.cross_entropy import CrossEntropyLoss
from nn.models.sequential import Sequential
from nn.optimizers.adam import Adam
from training.data_utils import preprocess_mnist
from training.pruning_recovery import (
    EarlyStopping,
    ReduceLROnPlateau,
    load_checkpoint,
    save_checkpoint,
)
from training.trainer import evaluate, train_one_epoch


INPUT_PATH = "./archive"
PLOT_DIR = "./plots/part2"
CHECKPOINT_PATH = "./checkpoints/part2_baseline_best.npz"
BATCH_SIZE = 64
MAX_EPOCHS = 15
SEED = 11

os.makedirs(PLOT_DIR, exist_ok=True)


def build_model(seed=SEED):
    """Same baseline architecture used in Part 3/Part 4."""
    return Sequential([
        Linear(784, 128, initializer="he_normal", seed=seed),
        ReLU(),
        Linear(128, 64, initializer="he_normal", seed=seed + 1),
        ReLU(),
        Linear(64, 10, initializer="xavier_uniform", seed=seed + 2),
    ])


def load_mnist():
    loader = MnistDataloader(
        join(INPUT_PATH, "train-images-idx3-ubyte/train-images-idx3-ubyte"),
        join(INPUT_PATH, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
        join(INPUT_PATH, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"),
        join(INPUT_PATH, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
    )
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = loader.load_data()
    x_train, y_train = preprocess_mnist(x_train_raw, y_train_raw)
    x_test, y_test = preprocess_mnist(x_test_raw, y_test_raw)
    return x_train, y_train, x_test, y_test


def plot_learning_curves(history):
    epochs = [row["epoch"] for row in history]

    # Loss curve
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Part 2: Baseline Learning Curve — Loss")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(PLOT_DIR, "part2_loss_learning_curve.png"), dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [row["train_accuracy"] * 100 for row in history], marker="o", label="Train accuracy")
    plt.plot(epochs, [row["val_accuracy"] * 100 for row in history], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Part 2: Baseline Learning Curve — Accuracy")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(PLOT_DIR, "part2_accuracy_learning_curve.png"), dpi=200)
    plt.close()


def main():
    x_train, y_train, x_test, y_test = load_mnist()

    # Deterministic 5,000-example validation split from the training set.
    rng = np.random.default_rng(2026)
    indices = rng.permutation(len(x_train))
    val_indices = indices[:5000]
    train_indices = indices[5000:]
    x_val, y_val = x_train[val_indices], y_train[val_indices]
    x_train, y_train = x_train[train_indices], y_train[train_indices]

    model = build_model(SEED)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=0.001)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-6, min_delta=1e-4
    )
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    print("========== PART 2: BASELINE TRAINING ==========")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            loss_fn,
            optimizer,
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            seed=SEED + epoch,
        )
        val_metrics = evaluate(model, loss_fn, x_val, y_val)
        val_loss = float(val_metrics["loss"])

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(model, CHECKPOINT_PATH)

        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": val_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "learning_rate": float(optimizer.learning_rate),
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train loss: {row['train_loss']:.4f} | "
            f"Train acc: {row['train_accuracy'] * 100:.2f}% | "
            f"Val loss: {row['val_loss']:.4f} | "
            f"Val acc: {row['val_accuracy'] * 100:.2f}% | "
            f"LR: {row['learning_rate']:.2e}"
        )

        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore the best validation-loss checkpoint before final evaluation.
    load_checkpoint(model, CHECKPOINT_PATH)
    best_val_metrics = evaluate(model, loss_fn, x_val, y_val)
    test_metrics = evaluate(model, loss_fn, x_test, y_test)

    plot_learning_curves(history)

    print("\n========== PART 2 SUMMARY ==========")
    print(f"Best checkpoint epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_metrics['accuracy'] * 100:.2f}%")
    print(f"Test accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Plots saved to: {PLOT_DIR}/")
    print(f"- {PLOT_DIR}/part2_loss_learning_curve.png")
    print(f"- {PLOT_DIR}/part2_accuracy_learning_curve.png")
    print(f"Best checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
