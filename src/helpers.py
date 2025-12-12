import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

# Plot KNN accuracy vs k
def plot_knn_accuracy_vs_k(k_values, accuracy_values, title="KNN Test Accuracy vs k"):
    """
    Plots test accuracy as a function of k (number of neighbors).

    Args:
        k_values (list or array): List of k values tried.
        accuracy_values (list or array): Corresponding test accuracies.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Test Accuracy")
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_labels=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix for the given true and predicted labels.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_labels (list): List of class names in the desired display order.
        title (str): Plot title.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    if class_labels is None:
        class_labels = np.unique(np.concatenate([y_true, y_pred]))

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# Plot classification metrics (precision/recall/F1) as bars
def plot_class_metrics_bar(y_true, y_pred, class_labels=None, title="Classification Metrics by Class"):
    """
    Plots precision, recall, and F1-score as grouped bars for each class.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_labels (list): Optional custom class order.
        title (str): Plot title.
    """
    if class_labels is None:
        class_labels = np.unique(np.concatenate([y_true, y_pred]))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0
    )

    x = np.arange(len(class_labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, prec, width, label='Precision')
    plt.bar(x,         rec, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')

    plt.xticks(x, class_labels)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


# Scatter plot of two features colored by class
def plot_feature_scatter(df, x_col, y_col, hue_col, title=None):
    """
    Creates a scatter plot of two numeric features colored by a categorical column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        hue_col (str): Column name used for color groups (categories).
        title (str): Plot title.
    """
    if title is None:
        title = f"{hue_col} by {x_col} and {y_col}"

    categories = df[hue_col].astype('category')
    codes = categories.cat.codes

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df[x_col], df[y_col], c=codes, cmap='viridis', alpha=0.7)

    handles = []
    cmap = scatter.cmap
    norm = scatter.norm

    for code, cat in enumerate(categories.cat.categories):
        color = cmap(norm(code))
        handles.append(
            plt.Line2D(
                [], [], 
                marker='o', 
                color=color, 
                linestyle='',
                label=cat
            )
        )

    plt.legend(handles=handles, title=hue_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()