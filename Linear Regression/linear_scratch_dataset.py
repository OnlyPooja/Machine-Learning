import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
california_housing = fetch_california_housing()

# (features) data: This is an attribute of the dataset that contains the feature data. In the California housing dataset, the feature data includes various attributes or columns that describe
# the characteristics of housing districts, such as average rooms, average bedrooms, population,
X = california_housing.data

#.target: This is an attribute of the dataset that contains the target variable data. In the context of regression problems, the
# target variable is the one you're trying to predict.
y = california_housing.target

# Normalize the features
X_mean = np.mean(X, axis=0)  # the line calculates the mean value for each feature (column) in your dataset X and stores these mean values in the array X_mean.
X_std = np.std(X, axis=0)     #calculates the standard deviation of the features along the specified axis for the dataset X
X_normalized = (X - X_mean) / X_std

# Add a column of ones to X for the intercept term
X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize parameters
#the line num_features = X_train.shape[1] is used to calculate the number of features in your training data
num_features = X_train.shape[1]
#the line initializes the parameter vector theta with zeros.This is a vector of parameters that your machine learning model will learn during training
theta = np.zeros(num_features)
learning_rate = 0.01
epochs = 1000

# Gradient Descent
for _ in range(epochs):
    y_pred = np.dot(X_train, theta)  # Predicted values
    error = y_pred - y_train          # Error between predicted and actual values

    # Update parameters using gradients
    gradient = np.dot(X_train.T, error) / len(X_train)
    theta -= learning_rate * gradient

# Make predictions on the test set
y_pred_test = np.dot(X_test, theta)

# Calculate the mean squared error on the test set
mse = np.mean((y_pred_test - y_test)**2)
print("Mean squared error:", mse)

# Plotting the results
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices")
plt.show()
