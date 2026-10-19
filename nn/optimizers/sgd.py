class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def step(self):
        """Update parameters using their accumulated gradients."""
        for parameter in self.parameters:
            parameter.data -= (
                self.learning_rate * parameter.grad
            )

    def zero_grad(self):
        """Reset gradients before the next training batch."""
        for parameter in self.parameters:
            parameter.zero_grad()
