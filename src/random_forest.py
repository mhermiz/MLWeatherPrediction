
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from src.helpers import plot_class_metrics_bar, plot_confusion_matrix

def use_random_forest(X_train, X_test, y_train, y_test):
    # Standardize the feature variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print("\nRandom Forest Test Accuracy:", acc_rf)
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=0))

    class_labels = rf.classes_
    plot_confusion_matrix(y_test, y_pred_rf, class_labels=class_labels, title=f"Confusion Matrix - RandomForest)")

    plot_class_metrics_bar(y_test, y_pred_rf, class_labels=class_labels, title="RF Classification Metrics by Class")