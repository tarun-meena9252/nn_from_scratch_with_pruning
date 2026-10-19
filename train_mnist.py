from os.path import join

from dataLoader import MnistDataloader
from training.data_utils import preprocess_mnist
from training.trainer import train_one_epoch, evaluate

from nn.models.sequential import Sequential
from nn.layers.linear import Linear
from nn.activations.functions import ReLU
from nn.losses.cross_entropy import CrossEntropyLoss
from nn.optimizers.adam import Adam


# Load MNIST.
input_path = "./archive"

loader = MnistDataloader(
    join(input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"),
    join(input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
    join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"),
    join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
)

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = loader.load_data()

x_train, y_train = preprocess_mnist(x_train_raw, y_train_raw)
x_test, y_test = preprocess_mnist(x_test_raw, y_test_raw)

# Construct model.
model = Sequential([
    Linear(784, 128, initializer="he_normal", seed=1),
    ReLU(),
    Linear(128, 64, initializer="he_normal", seed=2),
    ReLU(),
    Linear(64, 10, initializer="xavier_uniform", seed=3),
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), learning_rate=0.001)

# Train and evaluate.
epochs = 5

for epoch in range(epochs):
    train_metrics = train_one_epoch(
        model,
        loss_fn,
        optimizer,
        x_train,
        y_train,
        batch_size=64,
        seed=epoch,
    )

    test_metrics = evaluate(
        model,
        loss_fn,
        x_test,
        y_test,
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {train_metrics['loss']:.4f} | "
        f"Train accuracy: {train_metrics['accuracy']:.2%} | "
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"Test accuracy: {test_metrics['accuracy']:.2%}"
    )