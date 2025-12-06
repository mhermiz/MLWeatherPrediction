import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
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

# Normalize Class Imbalance (Oversampling)
oversampler  = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# print("Before oversampling:", y_train.value_counts())
# print("\nAfter oversampling:", y_train_resampled.value_counts())

# Standardize the feature variables
scaler_original = StandardScaler()
X_train_original_scaled = pd.DataFrame(
    scaler_original.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
X_test_original_scaled = pd.DataFrame(
    scaler_original.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

# Standardize the feature variables for oversampled data
scaler_resampled = StandardScaler()
X_train_resampled_scaled = pd.DataFrame(
    scaler_resampled.fit_transform(X_train_resampled),
    columns=X_train_resampled.columns,
    index=X_train_resampled.index,
)
X_test_resampled_scaled = pd.DataFrame(
    scaler_resampled.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

results = []  # list of tuples: (k, original_acc, oversampled_acc)

print("\nk   |  Original Acc  |  Oversampled Acc")
print("----------------------------------------")

for k in range(1, 21):
    # Model on original (imbalanced) data
    knn_original = KNeighborsClassifier(n_neighbors=k)
    knn_original.fit(X_train_original_scaled, y_train)
    y_pred_original = knn_original.predict(X_test_original_scaled)
    accuracy_original = accuracy_score(y_test, y_pred_original)

    # Model on oversampled data
    knn_oversampled = KNeighborsClassifier(n_neighbors=k)
    knn_oversampled.fit(X_train_resampled_scaled, y_train_resampled)
    y_pred_oversampled = knn_oversampled.predict(X_test_resampled_scaled)
    accuracy_oversampled = accuracy_score(y_test, y_pred_oversampled)

    results.append((k, accuracy_original, accuracy_oversampled))
    print(f"k={k:2d} |    {accuracy_original:.3f}     |      {accuracy_oversampled:.3f}")


# Best k for original data (based on original accuracy)
best_k_original, best_acc_original, _ = max(results, key=lambda x: x[1])

# Best k for oversampled data (based on oversampled accuracy)
best_k_oversampled, _, best_acc_oversampled = max(results, key=lambda x: x[2])

print(f"\nBest k (original)    = {best_k_original} with accuracy = {best_acc_original:.3f}")
print(f"Best k (oversampled) = {best_k_oversampled} with accuracy = {best_acc_oversampled:.3f}")


# Final model on original data
final_knn_original = KNeighborsClassifier(n_neighbors=best_k_original)
final_knn_original.fit(X_train_original_scaled, y_train)

# Final model on oversampled data
final_knn_oversampled = KNeighborsClassifier(n_neighbors=best_k_oversampled)
final_knn_oversampled.fit(X_train_resampled_scaled, y_train_resampled)


# Train / test accuracy for original model
train_accuracy_original = final_knn_original.score(X_train_original_scaled, y_train)
test_accuracy_original = final_knn_original.score(X_test_original_scaled, y_test)

# Train / test accuracy for oversampled model
train_accuracy_oversampled = final_knn_oversampled.score(
    X_train_resampled_scaled, y_train_resampled
)
test_accuracy_oversampled = final_knn_oversampled.score(
    X_test_resampled_scaled, y_test
)

# Example predictions (on test set, oversampled model)
test_predictions_oversampled = final_knn_oversampled.predict(X_test_resampled_scaled)

print("\n===== Final Model (Original Data) =====")
print(f"Best k: {best_k_original}")
print(f"Train Accuracy (original): {train_accuracy_original:.3f}")
print(f"Test Accuracy  (original): {test_accuracy_original:.3f}")

print("\n===== Final Model (Oversampled Data) =====")
print(f"Best k: {best_k_oversampled}")
print(f"Train Accuracy (oversampled): {train_accuracy_oversampled:.3f}")
print(f"Test Accuracy  (oversampled): {test_accuracy_oversampled:.3f}")
print("Sample Test Predictions (oversampled model):", test_predictions_oversampled[:20])


# Example prediction for new data
#new_data = pd.DataFrame([[12.8, 5.0, 0.0, 4.7]], columns=['temp_max', 'temp_min', 'precipitation', 'wind'])  # Example: temp_min=50, precipitation=0.1, wind=5
#new_data_scaled = scaler.transform(new_data)
#new_prediction = knnModel.predict(new_data_scaled)
#print("New Data Prediction:", new_prediction)


# Compare with y_test to see where it gets wrong.
# Build a confusion matrix to visualize which weather types it confuses
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knnModel.classes_)
# disp.plot(cmap='Blues') # color theme
# plt.title('Confusion Matrix - KNN (k=11)')
# plt.show() # display the figure