import math

# Training data
x = [1, 2, 3, 4, 5, 6]
y = [0, 0, 0, 1, 1, 1]

w = 0.0
b = 0.0

lr = 0.1
epochs = 1000

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


for epoch in range(epochs):
    dw = 0.0
    db = 0.0
    loss = 0.0
    
    for i in range(len(x)):
        # Forward pass
        z = w * x[i] + b
        y_hat = sigmoid(z)

        # Loss
        loss += -(
            y[i] * math.log(y_hat)
            + (1 - y[i]) * math.log(1 - y_hat)
        )

        # Gradient
        error = y_hat - y[i]

        dw += error * x[i]
        db += error

    # Average
    n = len(x)

    loss /= n
    dw /= n
    db /= n

    # Gradient descent
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(
            "epoch:", epoch,
            "loss:", loss,
            "w:", w,
            "b:", b
        )

# Prediction
x_test = 3.5

probability = sigmoid(w * x_test + b)

print("Probability:", probability)

if probability >= 0.5:
    print("Class: 1")
else:
    print("Class: 0")