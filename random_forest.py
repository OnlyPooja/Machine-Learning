import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'your_dataset.csv' with your dataset's file path)
# Example: df = pd.read_csv('your_dataset.csv')
# Ensure your dataset is in a format that can be used for classification

# For demonstration, let's use a sample dataset
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
})

# Split the data into features (X) and labels (y)
X = data[['Feature1', 'Feature2']]
y = data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier
random_forest_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = random_forest_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
