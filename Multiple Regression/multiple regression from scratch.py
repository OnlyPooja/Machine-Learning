import numpy as np

# Generate synthetic data
np.random.seed(0)
num_samples = 100
num_features = 3

# Simulate feature data (e.g., area, bedrooms, bathrooms)
X = np.random.rand(num_samples, num_features) * 10

# True coefficients for the features
true_coeffs = np.array([2.5, 1.8, 3.2])

# Simulate noise
noise = np.random.randn(num_samples) * 2

# Simulate target variable (house price)
y = np.dot(X, true_coeffs) + noise

# Normalize features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Add bias term
X_normalized = np.hstack((np.ones((num_samples, 1)), X_normalized))

# Initialize parameters (coefficients)
num_features_with_bias = num_features + 1
coeffs = np.random.rand(num_features_with_bias)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Gradient Descent
for _ in range(epochs):
    y_pred = np.dot(X_normalized, coeffs)  # Predicted values
    error = y_pred - y  # Error between predicted and actual values

    # Update coefficients using gradients
    gradient = np.dot(X_normalized.T, error) / num_samples
    coeffs -= learning_rate * gradient

# Make predictions
new_features = np.array([[5.0, 3.0, 2.5]])  # New feature values for prediction
new_features_normalized = (new_features - X_mean) / X_std
new_features_with_bias = np.hstack((np.ones((1, 1)), new_features_normalized))
predicted_price = np.dot(new_features_with_bias, coeffs)
print("Predicted Price:", predicted_price[0])
