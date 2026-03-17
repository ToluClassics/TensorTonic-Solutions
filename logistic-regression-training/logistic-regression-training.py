import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):

    X = np.array(X)
    y = np.array(y)

    N, F = X.shape

    W = np.zeros((F,1))
    b = 0.0

    y = y.reshape(N,1)

    for _ in range(steps):

        y_hat = _sigmoid(X @ W + b)

        error = y_hat - y

        dw = (X.T @ error) / N
        db = error.mean()

        W -= lr * dw
        b -= lr * db

    return (W.flatten(), b)