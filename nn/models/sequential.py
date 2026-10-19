from nn.core.module import Module


class Sequential(Module):
    def __init__(self, layers):
        if not layers:
            raise ValueError("Sequential requires at least one layer")

        self.layers = list(layers)

        for layer in self.layers:
            if not isinstance(layer, Module):
                raise TypeError(
                    "Every layer must inherit from Module"
                )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(axis=1)
