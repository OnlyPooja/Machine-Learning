

# Importing Packages
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
#print(housing.data[:5])
print("attributes",housing.feature_names)


#The goal of this preprocessing step is to transform the features in a way that they have a mean of approximately zero and a range of approximately 1.
X = (housing.data - housing.data.mean(axis=0)) / (housing.data.max(axis=0) - housing.data.min(axis=0))

y = housing.target    #the target variable is the variable that you're trying to predict or model.
#These are the values that you'll use to train and validate your machine learning model, and eventually, you'll compare the model's predictions to these actual target values to evaluate its performance.


# validation is that data for which u will make predictions
# splitting for training and validation set(The validation set is a separate subset of the data that you set aside for evaluating the model's performance during training)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=5)

from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
model = linear_regression.fit(x_train,y_train)

predictions = linear_regression.predict(x_val)


# MAE_val_with_sklearn = (1/y_val.shape[0]) * np.sum(np.abs(predictions - y_val))

print(predictions)  #prices of house
#The predictions you see are the model's estimated median house values based on its learned coefficients and the features of the validation set districts.

# Plotting the linear regression line, actual prices, and predicted prices
plt.scatter(y_val, predictions, alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], linestyle='--', color='red', label="Linear Regression Line")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices")
plt.legend()
plt.show()



