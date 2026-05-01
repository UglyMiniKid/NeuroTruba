import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]


def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]


def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def categorical_cross_entropy_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / y_true.shape[0]


LOSS_FUNCTIONS = {
    'mse': (mean_squared_error, mse_derivative),
    'binary_crossentropy': (binary_cross_entropy, binary_cross_entropy_derivative),
    'categorical_crossentropy': (categorical_cross_entropy, categorical_cross_entropy_derivative)
}


def get_loss_function(name):
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]