# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Analyze final target performance by class and dominant confusion."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def calculate_per_class_accuracy(labels, predictions, class_names):
    """Calculate correct counts and accuracy separately for every PACS class."""
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    if labels_array.shape != predictions_array.shape or labels_array.ndim != 1:
        raise ValueError("Labels and predictions must be matching one-dimensional arrays.")

    rows = []
    for class_index, class_name in enumerate(class_names):
        class_mask = labels_array == class_index
        total = int(class_mask.sum())
        correct = int((predictions_array[class_mask] == class_index).sum())
        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": float(correct / total) if total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calculate_class_accuracy_changes(baseline_table, method_table, method_name):
    """Compare one adapted model's target per-class accuracy with Source-only."""
    baseline = baseline_table[["class_index", "class_name", "accuracy"]].rename(
        columns={"accuracy": "baseline_source_only_accuracy"}
    )
    method_accuracy_column = f"{method_name}_accuracy"
    method = method_table[["class_index", "accuracy"]].rename(
        columns={"accuracy": method_accuracy_column}
    )
    comparison = baseline.merge(method, on="class_index", validate="one_to_one")
    comparison["accuracy_change"] = (
        comparison[method_accuracy_column]
        - comparison["baseline_source_only_accuracy"]
    )
    return comparison.sort_values("accuracy_change", ascending=False).reset_index(drop=True)


def calculate_dominant_confusions(labels, predictions, class_names):
    """Find the most frequent incorrect predicted class for each true class."""
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    matrix = confusion_matrix(
        labels_array,
        predictions_array,
        labels=np.arange(len(class_names)),
    )
    rows = []
    for class_index, class_name in enumerate(class_names):
        error_counts = matrix[class_index].copy()
        error_counts[class_index] = 0
        if int(error_counts.sum()) == 0:
            confused_name = None
            confusion_count = 0
        else:
            confused_index = int(np.argmax(error_counts))
            confused_name = class_names[confused_index]
            confusion_count = int(error_counts[confused_index])
        rows.append(
            {
                "true_class": class_name,
                "dominant_confusion": confused_name,
                "confusion_count": confusion_count,
            }
        )
    return pd.DataFrame(rows), matrix


def create_failure_case_table(prediction_table, class_names, maximum_cases=50):
    """Create a compact table of misclassified target image identifiers."""
    failures = prediction_table[
        prediction_table["label"] != prediction_table["prediction"]
    ].copy()
    failures["true_class"] = failures["label"].map(
        lambda index: class_names[int(index)]
    )
    failures["predicted_class"] = failures["prediction"].map(
        lambda index: class_names[int(index)]
    )
    return failures.head(maximum_cases).reset_index(drop=True)


def plot_confusion_matrix(matrix, class_names, title, output_file):
    """Save a readable target confusion-matrix figure without extra dependencies."""
    matrix_array = np.asarray(matrix, dtype=np.int64)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix_array, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix_array.max() / 2.0 if matrix_array.size else 0.0
    for row_index in range(matrix_array.shape[0]):
        for column_index in range(matrix_array.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix_array[row_index, column_index]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix_array[row_index, column_index] > threshold
                    else "black"
                ),
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
