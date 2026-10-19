class Module:
    def forward(self, x):
        raise NotImplementedError(
            "Each module must implement forward()"
        )

    def backward(self, grad_output):
        raise NotImplementedError(
            "Each module must implement backward()"
        )

    def parameters(self):
        """
        Return trainable parameters owned by this module.
        """
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()
