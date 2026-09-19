# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Fit and apply a shared-diagonal Mahalanobis unknownness score."""

from pathlib import Path

import numpy as np


def fit_mahalanobis_statistics(features, labels, number_of_classes=10, epsilon=1e-6):
    """Estimate class means and one regularized shared diagonal covariance."""
    feature_matrix = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if feature_matrix.ndim != 2 or label_array.ndim != 1:
        raise ValueError("Mahalanobis features and labels have invalid dimensions.")
    if len(feature_matrix) != len(label_array) or len(feature_matrix) == 0:
        raise ValueError("Mahalanobis features and labels must have matching lengths.")
    if float(epsilon) <= 0.0:
        raise ValueError("Mahalanobis covariance regularization must be positive.")

    class_means = []
    residual_batches = []
    for class_index in range(int(number_of_classes)):
        class_features = feature_matrix[label_array == class_index]
        if len(class_features) == 0:
            raise ValueError(f"No training features found for class {class_index}.")
        class_mean = class_features.mean(axis=0)
        class_means.append(class_mean)
        residual_batches.append(class_features - class_mean)
    class_means = np.stack(class_means, axis=0)
    residuals = np.concatenate(residual_batches, axis=0)
    diagonal_variance = np.mean(residuals ** 2, axis=0) + float(epsilon)
    return {
        "class_means": class_means,
        "diagonal_variance": diagonal_variance,
        "epsilon": float(epsilon),
        "number_of_training_examples": int(len(feature_matrix)),
    }


def calculate_mahalanobis_unknownness(features, statistics):
    """Return minimum diagonal Mahalanobis distance to any known class mean."""
    feature_matrix = np.asarray(features, dtype=np.float64)
    class_means = np.asarray(statistics["class_means"], dtype=np.float64)
    diagonal_variance = np.asarray(
        statistics["diagonal_variance"],
        dtype=np.float64,
    )
    if feature_matrix.ndim != 2 or class_means.ndim != 2:
        raise ValueError("Mahalanobis inputs must be two-dimensional.")
    if feature_matrix.shape[1] != class_means.shape[1]:
        raise ValueError("Mahalanobis feature dimensions do not match.")
    if diagonal_variance.shape != (feature_matrix.shape[1],):
        raise ValueError("Mahalanobis diagonal covariance has the wrong shape.")
    differences = feature_matrix[:, None, :] - class_means[None, :, :]
    squared_distances = np.sum(
        differences ** 2 / diagonal_variance[None, None, :],
        axis=2,
    )
    return squared_distances.min(axis=1)


def save_mahalanobis_statistics(statistics, output_file):
    """Save class means and covariance statistics in a compressed NumPy file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        class_means=statistics["class_means"],
        diagonal_variance=statistics["diagonal_variance"],
        epsilon=np.asarray(statistics["epsilon"], dtype=np.float64),
        number_of_training_examples=np.asarray(
            statistics["number_of_training_examples"],
            dtype=np.int64,
        ),
    )
    return output_path


def load_mahalanobis_statistics(statistics_file):
    """Load previously fitted Mahalanobis statistics without pickle data."""
    with np.load(Path(statistics_file), allow_pickle=False) as saved:
        return {
            "class_means": saved["class_means"],
            "diagonal_variance": saved["diagonal_variance"],
            "epsilon": float(saved["epsilon"]),
            "number_of_training_examples": int(
                saved["number_of_training_examples"]
            ),
        }
