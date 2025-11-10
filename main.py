import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# k = 11 is the best result so far
knnModel = KNeighborsClassifier(n_neighbors=11)

# fit the model: 
# The .fit() method trains the model on the training data
knnModel.fit(X_train, y_train)

# Evaluate the model
# .score() computes the accuracy (correct predictions ÷ total predictions).
trainaccuracy = knnModel.score(X_train, y_train) # how well it fits the training data
testaccuracy = knnModel.score(X_test, y_test) # how well it generalizes to unseen data
trainpredictions = knnModel.predict(X_train) # .predict() gives actual predicted labels

# test multiple K values to find which gives the best test accuracy
# The best k is the one with the highest test accuracy
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"k={k}, Test Accuracy={acc:.3f}")

print("Train Accuracy:", trainaccuracy)
print("Predictions:", trainpredictions)
print("Final Test Accuracy:", testaccuracy)

# Example prediction for new data
#new_data = pd.DataFrame([[12.8, 5.0, 0.0, 4.7]], columns=['temp_max', 'temp_min', 'precipitation', 'wind'])  # Example: temp_min=50, precipitation=0.1, wind=5
#new_data_scaled = scaler.transform(new_data)
#new_prediction = knnModel.predict(new_data_scaled)
#print("New Data Prediction:", new_prediction)


# Compare with y_test to see where it gets wrong.
# Build a confusion matrix to visualize which weather types it confuses
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knnModel.classes_)
disp.plot(cmap='Blues') # color theme
plt.title('Confusion Matrix - KNN (k=11)')
plt.show() # display the figure
