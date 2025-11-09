import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
weatherdata = pd.read_csv('seattle-weather.csv')

# Display first few rows of the dataset
print(weatherdata.head())

# Define features and target variable
X = weatherdata[['temp_min', 'precipitation', 'wind']]
y = weatherdata['weather']

# Intialize model with 5 neighbors
knnModel = KNeighborsClassifier(n_neighbors=3)

# fit the model
knnModel.fit(X, np.ravel(y))

# Calculate prediction for each instance in X
predictions = knnModel.predict(X)
print("Predictions:", predictions)

# Calculate accuracy
accuracy = knnModel.score(X, np.ravel(y))
print("Accuracy:", accuracy)
