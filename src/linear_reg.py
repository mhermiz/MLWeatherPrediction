from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from src.helpers import plot_class_metrics_bar, plot_confusion_matrix

def use_linear_reg(X_train, X_test, y_train, y_test):
    # Standardize the feature variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg_model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs'
    )

    log_reg_model.fit(X_train_scaled, y_train)

    train_acc_log = log_reg_model.score(X_train, y_train)
    print("Logistic Regression Train Accuracy:", train_acc_log)

    # Predictions and accuracy
    y_pred_log = log_reg_model.predict(X_test_scaled)
    acc_log = accuracy_score(y_test, y_pred_log)

    print("\nLogistic Regression Test Accuracy:", acc_log)
    print("Logistic Regression Classification Report:\n",
        classification_report(y_test, y_pred_log, zero_division=0))
    
    class_labels = log_reg_model.classes_
    plot_confusion_matrix(y_test, y_pred_log, class_labels=class_labels, title=f"Confusion Matrix - Linear Regression)")

    plot_class_metrics_bar(y_test, y_pred_log, class_labels=class_labels, title="LR Classification Metrics by Class")