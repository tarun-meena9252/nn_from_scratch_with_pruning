import numpy as np


class BacktrackingLineSearch:
    def __init__(
        self,
        parameters,
        initial_step_size=1.0,
        shrink_factor=0.5,
        armijo_constant=1e-4,
        max_iterations=25,
        min_step_size=1e-12,
    ):
        if initial_step_size <= 0:
            raise ValueError("Initial step size must be positive")

        if not 0 < shrink_factor < 1:
            raise ValueError("Shrink factor must be between 0 and 1")

        if not 0 < armijo_constant < 1:
            raise ValueError("Armijo constant must be between 0 and 1")

        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        if min_step_size <= 0:
            raise ValueError("Minimum step size must be positive")

        self.parameters = list(parameters)
        self.initial_step_size = initial_step_size
        self.shrink_factor = shrink_factor
        self.armijo_constant = armijo_constant
        self.max_iterations = max_iterations
        self.min_step_size = min_step_size

        self.last_step_size = None

    def step(self, loss_function):
        """
        Perform a gradient-descent step with Armijo backtracking.

        Assumes parameter gradients already contain the gradient
        of the scalar loss returned by loss_function.
        """
        params = self.parameters

        original_values = [
            p.data.copy() for p in params
        ]
        gradients = [
            p.grad.copy() for p in params
        ]

        directional_derivative = -sum(
            np.sum(grad * grad)
            for grad in gradients
        )

        if not np.isfinite(directional_derivative):
            raise ValueError("Gradients must be finite")

        if directional_derivative == 0:
            self.last_step_size = 0.0
            return False

        original_loss = float(loss_function())

        if not np.isfinite(original_loss):
            raise ValueError("Current loss must be finite")

        step_size = self.initial_step_size

        for _ in range(self.max_iterations):
            if step_size < self.min_step_size:
                break

            for parameter, original, grad in zip(
                params, original_values, gradients
            ):
                parameter.data[...] = original - step_size * grad

            candidate_loss = float(loss_function())

            if (
                np.isfinite(candidate_loss)
                and candidate_loss
                <= original_loss
                + self.armijo_constant
                * step_size
                * directional_derivative
            ):
                self.last_step_size = step_size
                return True

            step_size *= self.shrink_factor

        # No candidate was acceptable: restore all parameters.
        for parameter, original in zip(params, original_values):
            parameter.data[...] = original

        self.last_step_size = 0.0
        return False

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
