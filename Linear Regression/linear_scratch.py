import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
#The seed is like the starting point for those rules. If you use the same seed, you'll always get the same sequence of numbers from the machine.

np.random.seed(0)

#generates an array X containing 100 random numbers between 0 and 10.

X = np.random.rand(100, 1) * 10  # Random features (e.g., house size)
# X is independent variable example housing size,area anything

#creates an array y where each value is obtained by applying a linear equation to the values in X, and then adding some random noise to the linear relationship.

y = 2 * X + 3 + np.random.randn(100, 1) * 2  # True relationship with noise
#y is price of the house

# Initialize parameters
m = 0.1  # Initial slope
b = 0.1  # Initial intercept
learning_rate = 0.01 #how quick the model learns
epochs = 1000   #This is the number of times the gradient descent algorithm iterates over the entire dataset. Each iteration is called an epoch

# Gradient Descent
for _ in range(epochs):
    y_pred = m * X + b  # Predicted values of price calculated by program
    error = y_pred - y   # Error between predicted and actual values

    # Update parameters using gradients
    m -= learning_rate * np.mean(error * X)
    b -= learning_rate * np.mean(error)

# Make predictions
new_X = np.array([[4.5]])  # New house size for prediction
predicted_price = m * new_X + b
print("Predicted Price:", predicted_price[0][0])

# Plotting the results
plt.scatter(X, y, label="Actual data")
plt.plot(X, m * X + b, color='red', label="Linear regression")
plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("Linear Regression for House Price Prediction")
plt.legend()
plt.show()
