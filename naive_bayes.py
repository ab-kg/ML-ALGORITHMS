# ============================================================
# NAIVE BAYES - FROM SCRATCH
# ============================================================
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        # Unique classes
        self.classes = np.unique(y)

        # Store probabilities
        self.class_probs = {}
        self.feature_probs = {}

        n_samples = len(y)

        # Calculate probabilities for each class
        for c in self.classes:
            # Samples belonging to class c
            X_c = X[y == c]

            # Prior probability: P(class)
            self.class_probs[c] = len(X_c) / n_samples

            # P(feature = 1 | class)
            # Laplace smoothing (alpha = 1)
            self.feature_probs[c] = (
                (np.sum(X_c, axis=0) + 1)
                / (len(X_c) + 2)
            )

    def predict(self, X):
        predictions = []
        
        for sample in X:
            class_scores = {}

            for c in self.classes:
                # Start with log prior
                score = np.log(self.class_probs[c])

                # Feature probabilities
                probs = self.feature_probs[c]

                # Calculate log likelihood
                for j in range(len(sample)):
                    if sample[j] == 1:
                        score += np.log(probs[j])
                    else:
                        score += np.log(1 - probs[j])
                class_scores[c] = score

            # Choose class with highest probability
            prediction = max(
                class_scores,
                key=class_scores.get
            )

            predictions.append(prediction)
        return np.array(predictions)


# ============================================================
# DATASET
# ============================================================

# Features:
# [contains "free", contains "money"]

X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])

# Labels:
# 1 = Spam
# 0 = Not Spam

y = np.array([
    1,
    1,
    0,
    0
])

# ============================================================
# TRAIN
# ============================================================

model = NaiveBayes()
model.fit(X, y)

# ============================================================
# PREDICT
# ============================================================

X_test = np.array([
    [1, 1]
])
prediction = model.predict(X_test)
print("Prediction:", prediction)