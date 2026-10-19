
import numpy as np


def preprocess_mnist(images, labels):
    # Flatten 28x28 images into 784 features.
    x = np.asarray(images, dtype=np.float32).reshape(-1, 784)

    # Normalize pixel intensities from [0, 255] to [0, 1].
    x /= 255.0

    # Convert labels to a NumPy integer array.
    y = np.asarray(labels, dtype=np.int64)

    if len(x) != len(y):
        raise ValueError("Image and label counts do not match")

    return x, y
