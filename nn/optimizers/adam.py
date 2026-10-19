import numpy as np


class Adam:
    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")

        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # First and second moment estimates for each parameter.
        self.m = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]
        self.v = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

        self.t = 0

    def step(self):
        """Update all parameters using the Adam algorithm."""
        self.t += 1

        for i, parameter in enumerate(self.parameters):
            grad = parameter.grad

            # Update biased first moment estimate.
            self.m[i] = (
                self.beta1 * self.m[i]
                + (1.0 - self.beta1) * grad
            )

            # Update biased second moment estimate.
            self.v[i] = (
                self.beta2 * self.v[i]
                + (1.0 - self.beta2) * (grad ** 2)
            )

            # Bias correction.
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Parameter update.
            parameter.data -= (
                self.learning_rate
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )

    def zero_grad(self):
        """Reset gradients before processing the next batch."""
        for parameter in self.parameters:
            parameter.zero_grad()
