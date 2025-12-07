import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from helpers import get_season

# Load dataset
weatherdata = pd.read_csv('seattle-weather.csv')

# weatherdata['season'] = weatherdata['date'].apply(get_season)
# weatherdata = pd.get_dummies(weatherdata, columns=['season'])

# Merge drizzle => rain
weatherdata['weather'] = weatherdata['weather'].replace({
    'drizzle': 'rain',
    'snow': 'rain'
})

# Display first few rows of the dataset
print(weatherdata.head())


# Define features and target variable
feature_columns = [
    "temp_max",
    "temp_min",
    "precipitation",
    "wind",
]
X = weatherdata[feature_columns]
y = weatherdata['weather']

# Print all column names
print("All column names:")
print(weatherdata.columns.tolist())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# test multiple K values to find which gives the best test accuracy
# The best k is the one with the highest test accuracy
accuracy_list = []
maxAccScore = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    maxAccScore = max(maxAccScore, acc)
    accuracy_list.append(acc)
    print(f"k={k}, Test Accuracy={acc:.3f}")

# k = 11 is the best result so far
bestK = accuracy_list.index(maxAccScore) + 1
print(f"best k={bestK}")
knnModel = KNeighborsClassifier(n_neighbors=bestK, metric='manhattan')

# fit the model: 
# The .fit() method trains the model on the training data
knnModel.fit(X_train, y_train)

# Evaluate the model
# .score() computes the accuracy (correct predictions ÷ total predictions).
trainaccuracy = knnModel.score(X_train, y_train) # how well it fits the training data
testaccuracy = knnModel.score(X_test, y_test) # how well it generalizes to unseen data
trainpredictions = knnModel.predict(X_train) # .predict() gives actual predicted labels

print("Train Accuracy:", trainaccuracy)
print("Final Test Accuracy:", testaccuracy)
print("Predictions:", trainpredictions)

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