import numpy as np
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
california_housing = fetch_california_housing()
X = california_housing.data  # Features
y = california_housing.target  # Target variable (median house values)

# Normalize features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Add bias term
X_normalized_with_bias = np.hstack((np.ones((X.shape[0], 1)), X_normalized))

# Initialize coefficients
num_features_with_bias = X_normalized_with_bias.shape[1]
coeffs = np.random.rand(num_features_with_bias)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Gradient Descent
for _ in range(epochs):
    y_pred = np.dot(X_normalized_with_bias, coeffs)  # Predicted values
    error = y_pred - y  # Error between predicted and actual values

    # Update coefficients using gradients
    gradient = np.dot(X_normalized_with_bias.T, error) / X.shape[0]
    coeffs -= learning_rate * gradient

# Make predictions
new_features = np.array(
    [[0.02, 8, 3.5, 1, 0.55, 4.9, 24.7, 5.1, 2, 192, 20.1, 392.8]])  # New feature values for prediction
new_features_normalized = (new_features - X_mean) / X_std
new_features_with_bias = np.hstack((np.ones((new_features.shape[0], 1)), new_features_normalized))
predicted_median_value = np.dot(new_features_with_bias, coeffs)
print("Predicted Median Value:", predicted_median_value[0])
