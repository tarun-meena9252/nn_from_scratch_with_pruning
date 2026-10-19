import numpy as np
from nn.core.module import Module


class ReLU(Module):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        if self.x is None:
            raise RuntimeError("Call forward() before backward()")

        return grad_output * (self.x > 0)


class Sigmoid(Module):
    def __init__(self):
        self.output = None

    def forward(self, x):
        # Numerically stable sigmoid.
        output = np.empty_like(x, dtype=np.result_type(x, np.float64))

        positive = x >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

        exp_x = np.exp(x[~positive])
        output[~positive] = exp_x / (1.0 + exp_x)

        self.output = output
        return output

    def backward(self, grad_output):
        if self.output is None:
            raise RuntimeError("Call forward() before backward()")

        return grad_output * self.output * (1.0 - self.output)


class Tanh(Module):
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        if self.output is None:
            raise RuntimeError("Call forward() before backward()")

        return grad_output * (1.0 - self.output ** 2)


class Softmax(Module):
    def __init__(self):
        self.output = None

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("Softmax expects a 2D input")

        # Subtract the row maximum to avoid overflow.
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)

        self.output = exp_x / np.sum(
            exp_x, axis=1, keepdims=True
        )

        return self.output

    def backward(self, grad_output):
        if self.output is None:
            raise RuntimeError("Call forward() before backward()")

        # Jacobian-vector product, without constructing
        # a full Jacobian matrix.
        dot = np.sum(
            grad_output * self.output,
            axis=1,
            keepdims=True
        )

        return self.output * (grad_output - dot)
