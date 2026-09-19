import numpy as np

X = np.array([
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5]
], dtype=float)

y = np.array([3, 6, 9, 12, 15], dtype=float)

# Add bias column
X = np.column_stack([
    np.ones(len(X)),
    X
])

w = np.zeros(X.shape[1])

learning_rate = 0.01
lambda_ = 0.1
epochs = 1000

n = len(X)

for epoch in range(epochs):
    # prediction
    y_pred = X @ w

    # error
    error = y_pred - y

    # gradient of MSE
    gradient = (2 / n) * (X.T @ error)

    # regularization gradient
    regularization = 2 * lambda_ * w

    # total gradient
    gradient += regularization

    # update
    w = w - learning_rate * gradient

print(w)