# Training data
x = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# Parameters we want to learn
w = 0.0
b = 0.0

# Learning rate
lr = 0.01

# Number of training iterations
epochs = 80000
n = len(x)

for epoch in range(epochs):

    # -------------------------
    # Forward pass
    # -------------------------
    predictions = []

    for i in range(n):
        y_hat = w * x[i] + b
        predictions.append(y_hat)

    # -------------------------
    # Calculate MSE loss
    # -------------------------
    loss = 0.0
    for i in range(n):
        error = predictions[i] - y[i]
        loss += error ** 2
    loss /= n

    # -------------------------
    # Calculate gradients
    # -------------------------
    dw = 0.0
    db = 0.0

    for i in range(n):
        error = predictions[i] - y[i]
        dw += error * x[i]
        db += error

    dw = (2 / n) * dw
    db = (2 / n) * db
    
    # -------------------------
    # Gradient descent
    # -------------------------
    w = w - lr * dw
    b = b - lr * db

    # Print progress
    if epoch % 100 == 0:
        print(
            "epoch:", epoch,
            "loss:", loss,
            "w:", w,
            "b:", b
        )

print("\nFinal parameters:")
print("w =", w)
print("b =", b)

# # Test
# x_test = 6
# prediction = w * x_test + b

# print("Prediction for x =", x_test)
# print("Prediction =", prediction)