# Import necessary libraries
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Replace 'your_data.csv' with the actual CSV file path.
csv_file_path = 'population-projections.csv'

# Use pandas to read the CSV file into a DataFrame and specify data types
data = pd.read_csv(csv_file_path, dtype={
    'Country Name': str,
    'Year': int,
    'Country Code': str,
    'Low Fertility': float,
    'Medium Fertility': float,
    'High Fertility': float
})

# Assuming your CSV file has 'Year' and 'Low Fertility', 'Medium Fertility', 'High Fertility' columns, you can access them like this:
X = data[['Year', 'Low Fertility', 'High Fertility']]  # Features
y = data['Medium Fertility']  # Target is now 'Medium Fertility'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the AdaBoost model and set hyperparameters
base_model = DecisionTreeRegressor(max_depth=3)  # You can adjust the base model's hyperparameters
n_estimators = 100
learning_rate = 0.1

adaboost_model = AdaBoostRegressor(
    base_model,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=42
)

# Train the AdaBoost model
adaboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = adaboost_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Now, you can use the trained model to predict 'Medium Fertility' for a given set of features:
# For example, if 'X_next' contains the feature values for prediction:
X_next = pd.DataFrame({'Year': [2024], 'Low Fertility': 1.5, 'High Fertility': 2.5})  # Replace with actual values.
medium_fertility_next = adaboost_model.predict(X_next)
print(f"Predicted 'Medium Fertility' for the next year: {medium_fertility_next}")

