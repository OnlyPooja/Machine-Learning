import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data[:, np.newaxis, 2]  # Use a single feature for simplicity
y = diabetes.target

# Add polynomial features
degree = 3
#X_poly = [[x1, x1^2, x1^3],
    #     [x2, x2^2, x2^3],
    #     [x3, x3^2, x3^3],
    #     ...]
X_poly = np.hstack([X**d for d in range(1, degree + 1)])

# Gradient descent parameters
learning_rate = 0.001
num_epochs = 10000
m = len(y)

# Initialize parameters (coefficients)
theta = np.zeros(degree)

# Gradient descent
for epoch in range(num_epochs):
    y_pred = np.dot(X_poly, theta)  # Predicted values
    error = y_pred - y              # Error between predicted and actual values

    # Update parameters using gradients
    gradient = np.dot(X_poly.T, error) / m
    theta -= learning_rate * gradient

# Predict using the trained model
y_pred = np.dot(X_poly, theta)

# Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the results
plt.scatter(X, y, label='Training data')
plt.plot(X, y_pred, color='r', label='Polynomial regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
