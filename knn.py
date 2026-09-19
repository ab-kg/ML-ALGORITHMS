# ============================================================
# K-NEAREST NEIGHBORS (KNN)
# ============================================================
import numpy as np

# ------------------------------------------------------------
# 1. Euclidean Distance
# ------------------------------------------------------------

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# ------------------------------------------------------------
# 2. KNN Classification - From Scratch
# ------------------------------------------------------------

def knn_predict(X_train, y_train, x_test, k):
    distances = []

    # Calculate distance from test point to every training point
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], x_test)
        # Store (distance, label)
        distances.append((distance, y_train[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Select K nearest neighbors
    nearest_neighbors = distances[:k]

    # Extract labels
    labels = [label for _, label in nearest_neighbors]

    # Majority voting
    prediction = max(set(labels), key=labels.count)
    return prediction

# ------------------------------------------------------------
# 3. Example Dataset
# ------------------------------------------------------------

X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 7],
    [7, 8],
    [8, 7]
], dtype=float)

y_train = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

# New point
x_test = np.array([5, 6], dtype=float)

# Number of neighbors
k = 3


# ------------------------------------------------------------
# 4. Make Prediction
# ------------------------------------------------------------

prediction = knn_predict(
    X_train,
    y_train,
    x_test,
    k
)
print("Prediction:", prediction)

# ============================================================
# KNN REGRESSION - FROM SCRATCH
# ============================================================

def knn_regression_predict(X_train, y_train, x_test, k):
    distances = []

    # Calculate distances
    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], x_test)
        distances.append((distance, y_train[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Select K nearest neighbors
    nearest_neighbors = distances[:k]

    # Extract target values
    values = [value for _, value in nearest_neighbors]

    # Average
    prediction = np.mean(values)
    return prediction


# Example regression data
y_train_regression = np.array([
    100,
    120,
    110,
    200,
    220,
    210
], dtype=float)

regression_prediction = knn_regression_predict(
    X_train,
    y_train_regression,
    x_test,
    k=3
)

print("Regression prediction:", regression_prediction)


# ============================================================
# KNN USING SCIKIT-LEARN
# ============================================================

from sklearn.neighbors import KNeighborsClassifier

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Training
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([x_test])

print("Sklearn prediction:", prediction)


# ============================================================
# FEATURE SCALING
# ============================================================


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit scaler and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the SAME scaler
x_test_scaled = scaler.transform([x_test])

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train_scaled, y_train)

# Predict
prediction = model.predict(x_test_scaled)
print("Scaled KNN prediction:", prediction)



# ============================================================
# IMPORTANT KNN CONCEPTS
# ============================================================

# Classification:
#     Majority vote among K nearest neighbors
#
# Regression:
#     Average target value of K nearest neighbors
#
# Small K:
#     Low bias
#     High variance
#     Sensitive to noise
#
# Large K:
#     Higher bias
#     Lower variance
#     Smoother decision boundary
#
# KNN is a lazy learner:
#     Almost no training
#     Expensive prediction
#
# Feature scaling is important because
# distance-based algorithms are sensitive to feature scales.