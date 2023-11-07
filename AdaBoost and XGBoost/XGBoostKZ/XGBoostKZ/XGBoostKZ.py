import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

# Define the XGBoost model and set hyperparameters
params = {
    'max_depth': 3,  # You can adjust the maximum depth of the trees.
    'n_estimators': 100,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',  # For regression tasks.
    'random_state': 42
}

xgboost_model = xgb.XGBRegressor(**params)

# Train the XGBoost model
xgboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgboost_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared (Coefficient of Determination): {r2}")

# Now, you can use the trained XGBoost model to predict 'Medium Fertility' for a given set of features:
# For example, if 'X_next' contains the feature values for prediction:
X_next = pd.DataFrame({'Year': [2024], 'Low Fertility': 1.5, 'High Fertility': 2.5})  # Replace with actual values.
medium_fertility_next = xgboost_model.predict(X_next)
print(f"Predicted 'Medium Fertility' for the next year: {medium_fertility_next}")
