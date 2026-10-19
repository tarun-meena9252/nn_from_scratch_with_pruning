import numpy as np


def numerical_gradient(loss_function, parameter, h=1e-5):
    """
    Calculate numerical gradients for every element of parameter.data.

    loss_function must return a scalar loss.
    """
    numerical_grad = np.zeros_like(parameter.data, dtype=np.float64)

    for index in np.ndindex(parameter.data.shape):
        original_value = parameter.data[index]

        try:
            parameter.data[index] = original_value + h
            loss_plus = loss_function()

            parameter.data[index] = original_value - h
            loss_minus = loss_function()

            numerical_grad[index] = (
                loss_plus - loss_minus
            ) / (2.0 * h)

        finally:
            parameter.data[index] = original_value

    return numerical_grad


def relative_error(analytic, numerical, epsilon=1e-12):
    """
    Elementwise relative error, made robust near zero.
    """
    numerator = np.abs(analytic - numerical)
    denominator = np.maximum(
        np.abs(analytic) + np.abs(numerical),
        epsilon,
    )

    return numerator / denominator
