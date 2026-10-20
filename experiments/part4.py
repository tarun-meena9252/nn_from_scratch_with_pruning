
import os
from os.path import join
from time import perf_counter
import copy

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
from pruning.structured import structured_prune_model
from pruning.taylor import compute_taylor_scores
from training.data_utils import preprocess_mnist
from training.trainer import evaluate
from training.pruning_recovery import fine_tune_with_recovery


INPUT_PATH = "./archive"
PLOT_DIR = "./plots/part4/"
os.makedirs(PLOT_DIR, exist_ok=True)

BATCH_SIZE = 64
BASELINE_EPOCHS = 5
FINETUNE_EPOCHS = 3


def build_model(seed=1):
    return Sequential([
        Linear(784, 128, initializer="he_normal", seed=seed), # 128
        ReLU(),
        Linear(128, 64, initializer="he_normal", seed=seed + 1), # 64
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

    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = (
        loader.load_data()
    )

    x_train, y_train = preprocess_mnist(x_train_raw, y_train_raw)
    x_test, y_test = preprocess_mnist(x_test_raw, y_test_raw)

    return x_train, y_train, x_test, y_test


def count_weights(model):
    return sum(
        layer.W.data.size
        for layer in model.layers
        if isinstance(layer, Linear)
    )


def count_parameters(model):
    return sum(
        layer.W.data.size + layer.b.data.size
        for layer in model.layers
        if isinstance(layer, Linear)
    )


def measure_dense_forward_ms(model, x, repeats=10):
    batch = x[:min(1000, len(x))]
    model.forward(batch)

    timings = []
    for _ in range(repeats):
        start = perf_counter()
        model.forward(batch)
        timings.append(
            (perf_counter() - start) * 1000 / len(batch)
        )

    return float(np.median(timings))


def plot_sparsity_accuracy_pareto(summary, baseline_accuracies):
    """Test accuracy vs. weight reduction, with Pareto-optimal points."""

    baseline_accuracy = float(np.mean(baseline_accuracies))

    plt.figure(figsize=(10, 6))

    # Include the unpruned baseline.
    plt.scatter(
        [0],
        [baseline_accuracy],
        marker="*",
        s=220,
        label="Baseline",
        zorder=5,
    )

    for criterion in ("magnitude", "taylor"):
        rows = sorted(
            [r for r in summary if r["criterion"] == criterion],
            key=lambda r: r["reduction"],
        )

        reductions = np.array([r["reduction"] for r in rows])
        accuracies = np.array([r["accuracy_mean"] for r in rows])
        accuracy_std = np.array([r["accuracy_std"] for r in rows])

        plt.errorbar(
            reductions,
            accuracies,
            yerr=accuracy_std,
            marker="o",
            capsize=4,
            linewidth=2,
            label=criterion.capitalize(),
        )

        # A point is Pareto-optimal if no other point has both
        # at least as much weight reduction and at least as much accuracy,
        # with one of these objectives strictly better.
        points = list(zip(reductions, accuracies))
        pareto = []

        for i, (reduction_i, accuracy_i) in enumerate(points):
            dominated = any(
                reduction_j >= reduction_i
                and accuracy_j >= accuracy_i
                and (
                    reduction_j > reduction_i
                    or accuracy_j > accuracy_i
                )
                for j, (reduction_j, accuracy_j) in enumerate(points)
                if j != i
            )

            if not dominated:
                pareto.append((reduction_i, accuracy_i))

        if pareto:
            pareto.sort()
            px, py = zip(*pareto)
            plt.scatter(
                px,
                py,
                marker="D",
                s=75,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                label=f"{criterion.capitalize()} Pareto points",
                zorder=4,
            )

    plt.axhline(
        baseline_accuracy,
        linestyle="--",
        linewidth=1,
        label="Baseline accuracy",
    )

    plt.xlabel("Weight reduction (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Structured Pruning: Accuracy–Compression Pareto Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "part4_sparsity_accuracy_pareto.png"),
        dpi=200,
    )
    plt.close()


def plot_latency_vs_reduction(summary):
    """Inference latency vs. weight reduction."""

    plt.figure(figsize=(10, 6))

    for criterion in ("magnitude", "taylor"):
        rows = sorted(
            [r for r in summary if r["criterion"] == criterion],
            key=lambda r: r["reduction"],
        )

        plt.errorbar(
            [r["reduction"] for r in rows],
            [r["latency_mean"] for r in rows],
            yerr=[r["latency_std"] for r in rows],
            marker="o",
            capsize=4,
            linewidth=2,
            label=criterion.capitalize(),
        )

    plt.xlabel("Weight reduction (%)")
    plt.ylabel("Inference latency (ms/image)")
    plt.title("Inference Latency vs. Weight Reduction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "part4_latency_vs_reduction.png"),
        dpi=200,
    )
    plt.close()


def plot_accuracy_vs_latency(summary, baseline_accuracies):
    """Accuracy–latency trade-off."""

    baseline_accuracy = float(np.mean(baseline_accuracies))

    plt.figure(figsize=(10, 6))

    for criterion in ("magnitude", "taylor"):
        rows = [r for r in summary if r["criterion"] == criterion]

        plt.errorbar(
            [r["latency_mean"] for r in rows],
            [r["accuracy_mean"] for r in rows],
            xerr=[r["latency_std"] for r in rows],
            yerr=[r["accuracy_std"] for r in rows],
            marker="o",
            capsize=4,
            linestyle="-",
            label=criterion.capitalize(),
        )

        # Label each point with its weight-reduction percentage.
        for r in rows:
            plt.annotate(
                f'{r["reduction"]:.1f}%',
                (r["latency_mean"], r["accuracy_mean"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.axhline(
        baseline_accuracy,
        linestyle="--",
        linewidth=1,
        label="Baseline accuracy",
    )

    plt.xlabel("Inference latency (ms/image)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs. Inference Latency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "part4_accuracy_vs_latency.png"),
        dpi=200,
    )
    plt.close()


def main():
    x_train, y_train, x_test, y_test = load_mnist()

    # splitting training dataset further into validation dataset
    rng = np.random.default_rng(2026)
    indices = rng.permutation(len(x_train))

    val_indices = indices[:5000]
    train_indices = indices[5000:]

    x_val, y_val = x_train[val_indices], y_train[val_indices]
    x_train, y_train = x_train[train_indices], y_train[train_indices]

    # finished

    loss_fn = CrossEntropyLoss()

    seeds = [11, 22, 33]

    # Each tuple gives the fraction to prune from the CURRENT
    # width of hidden layer 1 and hidden layer 2.
    stages = [
        (0.25, 0.25),       # 128 -> 96; 64 -> 48
        (1 / 3, 1 / 3),     # 96 -> 64; 48 -> 32
        (0.50, 0.50),       # 64 -> 32; 32 -> 16
        (0.50, 0.50),       # 32 -> 16; 16 -> 8
        (9 / 16, 5 / 8),    # 16 -> 7; 8 -> 3
    ]

    all_results = []
    baseline_accuracies = []

    for seed in seeds:
        print(f"\n{'=' * 22} SEED {seed} {'=' * 22}")

        # Train one baseline for this seed.
        baseline = build_model(seed=seed)
        optimizer = Adam(baseline.parameters(), learning_rate=0.001)

        baseline, baseline_metrics = fine_tune_with_recovery(
            model=baseline,
            loss_fn=loss_fn,
            optimizer=optimizer,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            checkpoint_path=f"checkpoints/baseline_seed{seed}.npz",
            seed=seed,
            max_epochs=15,
            batch_size=BATCH_SIZE,
        )

        baseline_metrics = evaluate(baseline, loss_fn, x_test, y_test)
        baseline_accuracy = float(baseline_metrics["accuracy"])
        baseline_accuracies.append(baseline_accuracy * 100)
        seed_baseline_weights = count_weights(baseline)

        print(
            f"Baseline accuracy: {baseline_accuracy * 100:.2f}% | "
            
            f"Weights: {seed_baseline_weights:,}"
        )

        # Both methods start from the same baseline for this seed.
        for criterion in ("magnitude", "taylor"):
            model = copy.deepcopy(baseline)

            print(f"\n--- {criterion.upper()} | seed={seed} ---")

            for stage_index, (fraction_1, fraction_2) in enumerate(
                stages, start=1
            ):
                importance_scores = None

                # Recompute Taylor scores for the current model
                # before each pruning stage.
                if criterion == "taylor":
                    importance_scores = compute_taylor_scores(
                        model,
                        loss_fn,
                        x_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        max_batches=100,
                    )

                kwargs = {}
                if criterion == "taylor":
                    kwargs["importance_scores"] = importance_scores

                model = structured_prune_model(
                    model,
                    prune_fractions={
                        0: fraction_1,
                        1: fraction_2,
                    },
                    criterion=criterion,
                    **kwargs,
                )

                before_metrics = evaluate(
                    model, loss_fn, x_test, y_test
                )

                before_acc = before_metrics['accuracy']

                # Fine-tune the smaller model.

                checkpoint_path = (
                    f"checkpoints/pruning/"
                    f"{criterion}_seed{seed}_stage{stage_index}.npz"
                )

                optimizer = Adam(model.parameters(), learning_rate=0.001)

                model, _ = fine_tune_with_recovery(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    checkpoint_path=checkpoint_path,
                    seed=seed * 10000 + stage_index * 100,
                    max_epochs=15,
                    batch_size=BATCH_SIZE,
                )

                test_metrics = evaluate(model, loss_fn, x_test, y_test)

                test_acc = float(test_metrics["accuracy"])
                test_loss = float(test_metrics["loss"])

                weights = count_weights(model)
                parameters = count_parameters(model)
                reduction = 100 * (
                    1 - weights / seed_baseline_weights
                )
                latency_ms = measure_dense_forward_ms(model, x_test)

                result = {
                    "seed": seed,
                    "criterion": criterion,
                    "stage": stage_index,
                    "weights": weights,
                    "parameters": parameters,
                    "reduction": reduction,
                    "before_accuracy": before_acc * 100,
                    "accuracy": test_acc * 100,
                    "loss": test_loss,
                    "latency_ms": latency_ms,
                }
                all_results.append(result)

                print(
                    f"Stage {stage_index} | "
                    f"weights={weights:,} | "
                    f"reduction={reduction:.2f}% | "
                    f"before={before_acc * 100:.2f}% | "
                    f"after={test_acc * 100:.2f}% | "
                    f"latency={latency_ms:.6f} ms/image"
                )

    # Aggregate across the three paired runs.
    print("\n========== THREE-SEED SUMMARY ==========")
    print(
        f"Baseline accuracy: "
        f"{np.mean(baseline_accuracies):.2f}% "
        f"+/- {np.std(baseline_accuracies, ddof=1):.2f} pp"
    )

    print(
        f"\n{'Method':<12} {'Stage':>5} {'Weights':>9} "
        f"{'Reduction':>10} {'Accuracy mean±std':>21} "
        f"{'Latency mean':>13}"
    )
    print("-" * 80)

    summary = []

    for criterion in ("magnitude", "taylor"):
        for stage_index in range(1, len(stages) + 1):
            group = [
                r for r in all_results
                if r["criterion"] == criterion
                and r["stage"] == stage_index
            ]

            accuracies = np.array([r["accuracy"] for r in group])
            latencies = np.array([r["latency_ms"] for r in group])

            row = {
                "criterion": criterion,
                "stage": stage_index,
                "weights": group[0]["weights"],
                "reduction": group[0]["reduction"],
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=1)),
                "latency_mean": float(np.mean(latencies)),
                "latency_std": float(np.std(latencies, ddof=1)),
            }
            summary.append(row)

            print(
                f"{criterion:<12} {stage_index:>5} "
                f"{row['weights']:>9,} "
                f"{row['reduction']:>9.2f}% "
                f"{row['accuracy_mean']:>8.2f} +/- "
                f"{row['accuracy_std']:<6.2f} "
                f"{row['latency_mean']:>13.6f}"
            )

    # Generate all plots using the existing experiment summary.
    plot_sparsity_accuracy_pareto(summary, baseline_accuracies)
    plot_latency_vs_reduction(summary)
    plot_accuracy_vs_latency(summary, baseline_accuracies)

    print("\nAll Part 4 plots saved:")
    print(f"- {PLOT_DIR}/part4_sparsity_accuracy_pareto.png")
    print(f"- {PLOT_DIR}/part4_latency_vs_reduction.png")
    print(f"- {PLOT_DIR}/part4_accuracy_vs_latency.png")

    
if __name__ == "__main__":
    main()
