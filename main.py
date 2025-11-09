import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
weatherdata = pd.read_csv('seattle-weather.csv')

# Display first few rows of the dataset
print(weatherdata.head())

# Define features and target variable
X = weatherdata[['temp_max', 'temp_min', 'precipitation', 'wind']]
y = weatherdata['weather']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Intialize knn model with 3 neighbors
knnModel = KNeighborsClassifier(n_neighbors=3)

# fit the model
knnModel.fit(X_train, y_train)

# Evaluate the model
trainaccuracy = knnModel.score(X_train, y_train)
testaccuracy = knnModel.score(X_test, y_test)
trainpredictions = knnModel.predict(X_train)

print("Train Accuracy:", trainaccuracy)
print("Predictions:", trainpredictions)
print("Test Accuracy:", testaccuracy)

# Example prediction for new data
#new_data = pd.DataFrame([[12.8, 5.0, 0.0, 4.7]], columns=['temp_max', 'temp_min', 'precipitation', 'wind'])  # Example: temp_min=50, precipitation=0.1, wind=5
#new_data_scaled = scaler.transform(new_data)
#new_prediction = knnModel.predict(new_data_scaled)
#print("New Data Prediction:", new_prediction)
