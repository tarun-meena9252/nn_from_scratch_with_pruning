# nn_from_scratch_with_pruning

This repository contains code for building a dense neural network from scratch, including a reverse-mode autodiff engine, an Adam optimizer, and training code for the MNIST dataset. It also contains pruning experiments (magnitude pruning, self-pruning during training, and a Pareto sweep) along with their results.

---

## Installation

Developed and tested with Python 3.14.3.

```bash
# 1. Clone the repository and enter it
git clone <your-repo-url>
cd nn_from_scratch_with_pruning

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
#    Linux / macOS:
source venv/bin/activate
#    Windows (PowerShell):
venv\Scripts\Activate.ps1
#    Windows (cmd):
venv\Scripts\activate.bat

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Run all commands below from the repository root with the virtual environment activated.

---

## Running the experiments

### Part 1 — Reverse-mode autodiff engine that survives a changing graph

Runs the full core test suite, including the gradient-check tests:

```bash
python -m pytest tests/test_core.py tests/test_adam.py tests/test_linear.py tests/test_initializers.py tests/test_activations.py tests/test_sequential.py tests/test_cross_entropy.py tests/test_gradient_check.py -q
```

To run only the gradient checks:

```bash
python -m pytest tests/test_gradient_check.py -q
```

### Part 2 — Train a network with a real optimizer

Runs the tests (including the training test), then reproduces Part 2 training:

```bash
python -m pytest tests/test_core.py tests/test_adam.py tests/test_linear.py tests/test_initializers.py tests/test_activations.py tests/test_sequential.py tests/test_cross_entropy.py tests/test_gradient_check.py tests/test_training.py -q && python -m experiments.part2
```

### (Optional) Unstructured magnitude pruning

One-shot pruning using a mask (zero-multiplication), followed by a fine-tuning sweep:

```bash
python -m experiments.finetune_magnitude_sweep
```

### Part 3 — Make it prune itself

Runs the full test suite, then reproduces the self-pruning run:

```bash
python -m pytest -q && python -m experiments.part3
```

### Part 4 — Prove it (Pareto sweep)

Runs the full test suite, then reproduces the Pareto sweep:

```bash
python -m pytest -q && python -m experiments.part4
```

---

## Quick reference

| Goal | Command |
|---|---|
| Gradient-check tests | `python -m pytest tests/test_gradient_check.py -q` |
| Part 1 (all core tests) | see [Part 1](#part-1--reverse-mode-autodiff-engine-that-survives-a-changing-graph) |
| Part 2 training | see [Part 2](#part-2--train-a-network-with-a-real-optimizer) |
| Magnitude pruning sweep (optional) | `python -m experiments.finetune_magnitude_sweep` |
| Part 3 self-pruning run | `python -m pytest -q && python -m experiments.part3` |
| Part 4 Pareto sweep | `python -m pytest -q && python -m experiments.part4` |
