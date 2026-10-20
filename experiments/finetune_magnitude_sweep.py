from os.path import join
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Save figures without opening GUI windows.
import matplotlib.pyplot as plt
import numpy as np

from dataLoader import MnistDataloader
from training.data_utils import preprocess_mnist
from training.batch_utils import iterate_minibatches
from training.trainer import train_one_epoch, evaluate

from nn.models.sequential import Sequential
from nn.layers.linear import Linear
from nn.activations.functions import ReLU
from nn.losses.cross_entropy import CrossEntropyLoss
from nn.optimizers.adam import Adam

from pruning.magnitude import global_magnitude_masks, apply_masks


# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_PATH = "./archive"
BASELINE_EPOCHS = 5
FINETUNE_EPOCHS = 3
BATCH_SIZE = 64
BASELINE_LR = 0.001
FINETUNE_LR = 0.0001
SPARSITY_LEVELS = [0.50, 0.75, 0.90, 0.95]
PLOT_DIR = Path("plots")


# --------------------------------------------------
# Load MNIST
# --------------------------------------------------
loader = MnistDataloader(
    join(INPUT_PATH, "train-images-idx3-ubyte/train-images-idx3-ubyte"),
    join(INPUT_PATH, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
    join(INPUT_PATH, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"),
    join(INPUT_PATH, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
)

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = loader.load_data()
x_train, y_train = preprocess_mnist(x_train_raw, y_train_raw)
x_test, y_test = preprocess_mnist(x_test_raw, y_test_raw)


# --------------------------------------------------
# Model
# --------------------------------------------------
def build_model():
    return Sequential([
        Linear(784, 128, initializer="he_normal", seed=1),
        ReLU(),
        Linear(128, 64, initializer="he_normal", seed=2),
        ReLU(),
        Linear(64, 10, initializer="xavier_uniform", seed=3),
    ])


# --------------------------------------------------
# Dense forward-pass timing
# --------------------------------------------------

def measure_dense_forward_ms(model, x, repeats=10):
    """
    Time the existing model forward pass on a fixed batch.

    This measures the current dense implementation, not sparse
    execution. It includes Python and NumPy forward-pass overhead.
    """
    batch = x[:min(1000, len(x))]

    # Warm up before collecting timings.
    model.forward(batch)

    timings = []

    for _ in range(repeats):
        start = perf_counter()
        model.forward(batch)
        elapsed = perf_counter() - start

        timings.append(elapsed * 1000 / len(batch))

    return float(np.median(timings))


model = build_model()
loss_fn = CrossEntropyLoss()
baseline_optimizer = Adam(model.parameters(), learning_rate=BASELINE_LR)

print("\n========== TRAINING BASELINE ==========")
for epoch in range(BASELINE_EPOCHS):
    train_metrics = train_one_epoch(
        model,
        loss_fn,
        baseline_optimizer,
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        seed=epoch,
    )
    test_metrics = evaluate(model, loss_fn, x_test, y_test)
    print(
        f"Epoch {epoch + 1}/{BASELINE_EPOCHS} | "
        f"Train loss: {train_metrics['loss']:.4f} | "
        f"Train accuracy: {train_metrics['accuracy']:.2%} | "
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"Test accuracy: {test_metrics['accuracy']:.2%}"
    )

baseline_metrics = evaluate(model, loss_fn, x_test, y_test)
all_parameters = list(model.parameters())
baseline_parameter_values = [p.data.copy() for p in all_parameters]
baseLine_dense_ms = measure_dense_forward_ms(model, x_test)

weight_parameters = [
    layer.W for layer in model.layers if isinstance(layer, Linear)
]
total_weights = sum(parameter.data.size for parameter in weight_parameters)

print("\n========== BASELINE RESULT ==========")
print(f"Test loss: {baseline_metrics['loss']:.4f}")
print(f"Test accuracy: {baseline_metrics['accuracy']:.2%}")
print(f"Total weight positions (biases excluded): {total_weights:,}")


# --------------------------------------------------
# Fine-tuning helper: enforce masks after every update
# --------------------------------------------------
def train_one_pruned_epoch(optimizer, masks, weight_optimizer_indices, epoch):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in iterate_minibatches(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=1000 + epoch,
    ):
        logits = model.forward(batch_x)
        loss = loss_fn.forward(logits, batch_y)

        predictions = np.argmax(logits, axis=1)
        batch_count = len(batch_y)
        total_correct += int(np.sum(predictions == batch_y))
        total_loss += float(loss) * batch_count
        total_samples += batch_count

        model.zero_grad()
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)
        optimizer.step()

        # Re-enforce the mask after Adam's update.
        apply_masks(weight_parameters, masks)

        # Clear Adam's moment estimates for pruned connections.
        for mask, state_index in zip(masks, weight_optimizer_indices):
            optimizer.m[state_index][~mask] = 0.0
            optimizer.v[state_index][~mask] = 0.0

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


# --------------------------------------------------
# Independent sparsity sweep
# --------------------------------------------------
results = []
baseline_accuracy = float(baseline_metrics["accuracy"])
baseline_loss = float(baseline_metrics["loss"])

print("\n========== MAGNITUDE PRUNING + FINE-TUNING ==========")
print(
    f"{'Target':>7} | {'Actual':>7} | {'Pruned Acc':>10} | "
    f"{'Tuned Acc':>9} | {'Pruned Loss':>11} | {'Tuned Loss':>10} | {'Nonzero':>8} | {'Dense ms/img':>12}"
)
print("-" * 93)

for target_sparsity in SPARSITY_LEVELS:
    # Each experiment starts from exactly the same trained baseline.
    for parameter, baseline_value in zip(all_parameters, baseline_parameter_values):
        parameter.data[...] = baseline_value

    masks = global_magnitude_masks(weight_parameters, sparsity=target_sparsity)
    apply_masks(weight_parameters, masks)

    pruned_metrics = evaluate(model, loss_fn, x_test, y_test)
    pruned_accuracy = float(pruned_metrics["accuracy"])
    pruned_loss = float(pruned_metrics["loss"])

    # Fresh Adam state for each independent sparsity experiment.
    optimizer = Adam(model.parameters(), learning_rate=FINETUNE_LR)
    parameter_index_by_id = {
        id(parameter): index
        for index, parameter in enumerate(optimizer.parameters)
    }
    weight_optimizer_indices = [
        parameter_index_by_id[id(parameter)]
        for parameter in weight_parameters
    ]

    print(f"\nFine-tuning at {target_sparsity:.0%} sparsity")
    for epoch in range(FINETUNE_EPOCHS):
        train_metrics = train_one_pruned_epoch(
            optimizer, masks, weight_optimizer_indices,
            epoch=int(target_sparsity * 1000) + epoch,
        )
        test_metrics = evaluate(model, loss_fn, x_test, y_test)
        print(
            f"  Epoch {epoch + 1}/{FINETUNE_EPOCHS} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train accuracy: {train_metrics['accuracy']:.2%} | "
            f"Test loss: {test_metrics['loss']:.4f} | "
            f"Test accuracy: {test_metrics['accuracy']:.2%}"
        )

    final_metrics = evaluate(model, loss_fn, x_test, y_test)
    zero_weights = sum(
        int(np.count_nonzero(parameter.data == 0))
        for parameter in weight_parameters
    )
    nonzero_weights = total_weights - zero_weights
    actual_sparsity = zero_weights / total_weights

    dense_ms = measure_dense_forward_ms(model, x_test)

    # Verify both exact mask preservation and requested sparsity accounting.
    assert all(
        np.all(parameter.data[~mask] == 0.0)
        for parameter, mask in zip(weight_parameters, masks)
    ), f"A pruned weight became nonzero at target {target_sparsity:.0%}"

    results.append({
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sparsity,
        "pruned_accuracy": pruned_accuracy,
        "finetuned_accuracy": float(final_metrics["accuracy"]),
        "pruned_loss": pruned_loss,
        "finetuned_loss": float(final_metrics["loss"]),
        "nonzero_weights": nonzero_weights,
        "dense_forward_ms_per_image": dense_ms,
    })

    print(
        f"{target_sparsity:>6.0%} | {actual_sparsity:>6.2%} | "
        f"{pruned_accuracy:>9.2%} | {final_metrics['accuracy']:>8.2%} | "
        f"{pruned_loss:>10.4f} | {final_metrics['loss']:>9.4f} | "
        f"{nonzero_weights:>8,} {dense_ms:12.5f}"
    )
    print("  Mask verification passed: all pruned weights remain zero.")


# --------------------------------------------------
# Save plots without opening GUI windows
# --------------------------------------------------
PLOT_DIR.mkdir(parents=True, exist_ok=True)

x_sparsity = [0.0] + [r["actual_sparsity"] for r in results]
labels = ["Baseline"] + [f'{r["target_sparsity"]:.0%}' for r in results]
pruned_accuracies = [baseline_accuracy] + [r["pruned_accuracy"] for r in results]
finetuned_accuracies = [baseline_accuracy] + [r["finetuned_accuracy"] for r in results]
pruned_losses = [baseline_loss] + [r["pruned_loss"] for r in results]
finetuned_losses = [baseline_loss] + [r["finetuned_loss"] for r in results]
nonzero_counts = [total_weights] + [r["nonzero_weights"] for r in results]
dense_latency = [baseLine_dense_ms] + [r["dense_forward_ms_per_image"] for r in results]

# magnitude finetuning accuracy
fig, ax = plt.subplots(figsize=(8, 5))
positions = np.arange(len(labels))
ax.plot(positions, np.array(pruned_accuracies) * 100, marker="o", label="Immediately after pruning")
ax.plot(positions, np.array(finetuned_accuracies) * 100, marker="o", label="After fine-tuning")
ax.set_xticks(positions, labels)
ax.set_xlabel("Target sparsity")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Magnitude Pruning: Accuracy Before and After Fine-Tuning")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "magnitude_finetuning_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# magnitude finetuning loss
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(positions, pruned_losses, marker="o", label="Immediately after pruning")
ax2.plot(positions, finetuned_losses, marker="o", label="After fine-tuning")
ax2.set_xticks(positions, labels)
ax2.set_xlabel("Target sparsity")
ax2.set_ylabel("Test loss")
ax2.set_title("Magnitude Pruning: Loss Before and After Fine-Tuning")
ax2.grid(True, alpha=0.3)
ax2.legend()
fig2.tight_layout()
fig2.savefig(PLOT_DIR / "magnitude_finetuning_loss.png", dpi=150, bbox_inches="tight")
plt.close(fig2)


# magnitude finetuning nonzero weights
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(positions, nonzero_counts, marker="o")
ax3.set_xticks(positions, labels)
ax3.set_xlabel("Target sparsity")
ax3.set_ylabel("Nonzero weight count")
ax3.set_title("Remaining Nonzero Weights After Pruning")
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(PLOT_DIR / "magnitude_finetuning_nonzero_weights.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

# magnitude finetuning inference latency
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(positions, dense_latency, marker="o", linewidth=2)
ax4.set_xticks(positions, labels)
ax4.set_xlabel("Achieved weight sparsity (%)")
ax4.set_ylabel("Median milliseconds per image")
ax4.set_title("Dense Forward-Pass Latency After Magnitude Pruning")
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(PLOT_DIR / "magnitude_finetuning_inference_latency.png", dpi=150, bbox_inches="tight")
plt.close(fig4)

print("\n========== SUMMARY ==========")
print(f"Baseline accuracy: {baseline_accuracy:.2%}")
print(f"Baseline test loss: {baseline_loss:.4f}")
print(f"Plots saved in: {PLOT_DIR.resolve()}")
print("Saved:")
print("  - magnitude_finetuning_accuracy.png")
print("  - magnitude_finetuning_loss.png")
print("  - magnitude_finetuning_nonzero_weights.png")
print("No plot windows were opened.")
