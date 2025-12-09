import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ------------------ LOAD DATA ------------------

weatherdata = pd.read_csv('seattle-weather.csv')

# Combine minority classes into 'rain'
weatherdata['weather'] = weatherdata['weather'].replace({
    'drizzle': 'rain',
    'snow': 'rain'
})

# Display first few rows of the dataset
# print(weatherdata.head())

# Group less frequent weather types into one category
weatherdata['weather_grouped'] = weatherdata['weather'].replace({"drizzle": "rain", "snow": "rain"})
weatherdata = weatherdata[weatherdata['weather_grouped'] != "fog"]

# Define features and target variable
X = weatherdata[['temp_max', 'temp_min', 'precipitation', 'wind']]
y = weatherdata['weather_grouped']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ KNN MODEL (BASELINE) ------------------



# Test multiple K values to see performance
#print("\nKNN accuracy for k = 1 to 20:")
best_k = None
best_acc = 0.0

for k in range(1, 21):
   knn = KNeighborsClassifier(n_neighbors=k)
   knn.fit(X_train, y_train)
   y_pred_k = knn.predict(X_test)
   acc = accuracy_score(y_test, y_pred_k)
   print(f"k={k}, Test Accuracy={acc:.3f}")
   if acc > best_acc:
       best_acc = acc
       best_k = k

print(f"\nBest k based on this run: k={best_k}, Test Accuracy={best_acc:.3f}")

knnModel = KNeighborsClassifier(n_neighbors=best_k)

knnModel.fit(X_train, y_train)

trainaccuracy_knn = knnModel.score(X_train, y_train)

# Final KNN predictions with k=best_k (chosen model)
y_pred_knn = knnModel.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\nKNN Train Accuracy:", trainaccuracy_knn)
print(f"KNN Test Accuracy (k={best_k}):", acc_knn)
print("KNN Classification Report:\n",
      classification_report(y_test, y_pred_knn, zero_division=0))

# ------------------ LOGISTIC REGRESSION MODEL ------------------

# Logistic Regression model
log_reg_model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs'
)

log_reg_model.fit(X_train, y_train)

# Predictions and accuracy
y_pred_log = log_reg_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Test Accuracy:", acc_log)
print("Logistic Regression Classification Report:\n",
      classification_report(y_test, y_pred_log, zero_division=0))

# ------------------ CONFUSION MATRICES FOR BOTH MODELS ------------------

labels = knnModel.classes_

# KNN confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=labels)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=labels)

plt.figure(figsize=(6, 5))
disp_knn.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix - KNN (k={best_k})')
plt.tight_layout()
plt.savefig("confusion_matrix_knn.png", dpi=300)

# Logistic Regression confusion matrix
cm_log = confusion_matrix(y_test, y_pred_log, labels=labels)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=labels)

plt.figure(figsize=(6, 5))
disp_log.plot(cmap='Greens', values_format='d')
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
plt.savefig("confusion_matrix_logreg.png", dpi=300)

# Show plots (comment out if running in a non-GUI environment)
plt.show()

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
    'Model': [f'KNN (k={best_k})', 'Logistic Regression', 'Random Forest'],
    # 'Test Accuracy': [testaccuracy, acc_log, acc_rf]
})
# ------------------ SIMPLE COMPARISON TABLE ------------------

comparison = pd.DataFrame({
    'Model': [f'KNN (k={best_k})', 'Logistic Regression'],
    'Test Accuracy': [acc_knn, acc_log]
})

print("\nModel Comparison:\n", comparison)