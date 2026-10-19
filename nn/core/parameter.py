import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise TypeError("Parameter data must be a NumPy array")

        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError("Parameter data must have a floating-point dtype")

        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        """Reset accumulated gradients."""
        self.grad.fill(0.0)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return (
            f"Parameter(shape={self.shape}, "
            f"dtype={self.data.dtype}, "
            f"grad_norm={np.linalg.norm(self.grad):.6f})"
        )
