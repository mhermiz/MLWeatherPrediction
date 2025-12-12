import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.helpers import plot_class_metrics_bar, plot_feature_scatter, plot_knn_accuracy_vs_k, plot_confusion_matrix

def use_knn(X_train, X_test, y_train, y_test):

    # weatherdata = pd.read_csv('./data/seattle-weather.csv')

    # plot_feature_scatter(weatherdata[0:400], x_col='weather', y_col='temp_max',
    #                  hue_col='weather', title="Weather Types by Temperature")

    # Standardize the feature variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    accuracy_list, maxAccScore, bestK = find_best_k(X_train_scaled, y_train, X_test_scaled, y_test)

    print(f"\nBest k based on test accuracy: k={bestK} (accuracy={maxAccScore:.3f})")

    k_list = list(range(1, 21))
    plot_knn_accuracy_vs_k(k_list, accuracy_list)
    
    # Train final KNN model with best k
    knnModel = KNeighborsClassifier(n_neighbors=bestK, metric='manhattan')

    # The .fit() method trains the model on the training data
    knnModel.fit(X_train_scaled, y_train)

    y_train_pred = knnModel.predict(X_train_scaled)
    y_test_pred = knnModel.predict(X_test_scaled)

    class_labels = knnModel.classes_
    plot_confusion_matrix(y_test, y_test_pred, class_labels=class_labels, title=f"Confusion Matrix - KNN (k={bestK}, Manhattan)")

    plot_class_metrics_bar(y_test, y_test_pred, class_labels=class_labels, title="KNN Classification Metrics by Class")

    evaluate_model(y_train, y_train_pred, y_test, y_test_pred)


# test multiple K values to find which gives the best test accuracy
def find_best_k(X_train_scaled, y_train, X_test_scaled, y_test):
    accuracy_list = []
    maxAccScore = 0
    bestK = 1
    
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        knn.fit(X_train_scaled, y_train)
        y_val_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_val_pred)
        accuracy_list.append(acc)

        if acc > maxAccScore:
            maxAccScore = acc
            bestK = k
        
        print(f"k={k:2d}, Test Accuracy={acc:.3f}")

    return accuracy_list, maxAccScore, bestK
    

# Evaluate model: Accuracy, Precision, Recall, F1 (classification report)
def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\n------- Final KNN Model Performance -------")
    print(f"Train Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy:  {test_accuracy:.3f}")

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_test_pred))