import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
weatherdata = pd.read_csv('seattle-weather.csv')

# Display first few rows of the dataset
print(weatherdata.head())

# Group less frequent weather types into one category
weatherdata['weather_grouped'] = weatherdata['weather'].replace({"drizzle": "rain", "snow": "rain"})
weatherdata = weatherdata[weatherdata['weather_grouped'] != "fog"]

# Define features and target variable
X = weatherdata[['temp_max', 'temp_min', 'precipitation', 'wind']]
y = weatherdata['weather_grouped']

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
# plt.show() # display the figure

# ------- LOGISTIC REGRESSION MODEL -------

log_reg_model = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    solver='lbfgs'
)

log_reg_model.fit(X_train, y_train)

y_pred_log = log_reg_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Test Accuracy:", acc_log)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log, zero_division=0))

# Confusion matrix for Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=log_reg_model.classes_)
disp_log.plot(cmap='Greens')
plt.title('Confusion Matrix - Logistic Regression')

# Show plots (uncomment if running in an environment that supports plotting)
# plt.show()

# ------ RANDOM FOREST CLASSIFIER --------
rf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Test Accuracy:", acc_rf)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf.classes_)
disp_rf.plot(cmap='Oranges')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# ------- SIMPLE COMPARISON TABLE -------

comparison = pd.DataFrame({
    'Model': ['KNN (k=11)', 'Logistic Regression', 'Random Forest'],
    'Test Accuracy': [testaccuracy, acc_log, acc_rf]
})

print("\nModel Comparison:\n", comparison)