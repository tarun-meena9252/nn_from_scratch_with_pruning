import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.probabilities = None
        self.targets = None

    def forward(self, logits, targets):
        """
        logits:  (batch_size, num_classes)
        targets: (batch_size,), integer class indices

        Returns mean cross-entropy loss.
        """
        if logits.ndim != 2:
            raise ValueError("Logits must be a 2D array")

        if targets.ndim != 1:
            raise ValueError("Targets must be a 1D array")

        batch_size, num_classes = logits.shape

        if batch_size == 0:
            raise ValueError("Batch must not be empty")

        if targets.shape[0] != batch_size:
            raise ValueError("Batch sizes of logits and targets differ")

        if not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("Targets must contain integer class indices")

        if np.any(targets < 0) or np.any(targets >= num_classes):
            raise ValueError("Target class index out of range")

        # Numerically stable log-softmax.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(
            np.sum(np.exp(shifted), axis=1, keepdims=True)
        )
        log_probabilities = shifted - log_sum_exp

        self.probabilities = np.exp(log_probabilities)
        self.targets = targets.copy()

        correct_log_probabilities = log_probabilities[
            np.arange(batch_size), targets
        ]

        return -np.mean(correct_log_probabilities)

    def backward(self):
        """
        Gradient of mean cross-entropy with respect to logits.
        """
        if self.probabilities is None or self.targets is None:
            raise RuntimeError("Call forward() before backward()")

        batch_size = self.targets.shape[0]

        grad = self.probabilities.copy()
        grad[np.arange(batch_size), self.targets] -= 1.0
        grad /= batch_size

        return grad
