
import numpy as np

from nn.core.module import Module
from nn.core.parameter import Parameter
from nn.initializers import weights as init


class Linear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        initializer="he_normal",
        seed=None,
    ):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive")

        initializers = {
            "zeros": init.zeros,
            "xavier_uniform": init.xavier_uniform,
            "xavier_normal": init.xavier_normal,
            "he_uniform": init.he_uniform,
            "he_normal": init.he_normal,
        }

        if initializer not in initializers:
            raise ValueError(
                f"Unknown initializer: {initializer}. "
                f"Choose from {list(initializers)}"
            )

        rng = np.random.default_rng(seed)

        if initializer == "zeros":
            weights = init.zeros((in_features, out_features))
        else:
            weights = initializers[initializer](
                in_features,
                out_features,
                rng=rng,
            )

        biases = np.zeros((1, out_features), dtype=np.float64)

        self.W = Parameter(weights)
        self.b = Parameter(biases)
        self.x = None

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                "Input must have shape (batch_size, features)"
            )

        if x.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Expected {self.W.shape[0]} features, "
                f"received {x.shape[1]}"
            )

        self.x = x
        return x @ self.W.data + self.b.data

    def backward(self, grad_output):
        if self.x is None:
            raise RuntimeError("Call forward() before backward()")

        if grad_output.shape != (
            self.x.shape[0],
            self.W.shape[1],
        ):
            raise ValueError("Invalid gradient shape")

        self.W.grad += self.x.T @ grad_output
        self.b.grad += np.sum(
            grad_output,
            axis=0,
            keepdims=True,
        )

        return grad_output @ self.W.data.T

    def parameters(self):
        return [self.W, self.b]
