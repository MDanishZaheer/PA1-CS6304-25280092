# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calibrate unknownness thresholds using CIFAR-10 validation data only."""

import numpy as np


def calculate_validation_threshold(validation_unknownness, percentile=95.0):
    """Return the requested percentile of known-validation unknownness."""
    scores = np.asarray(validation_unknownness, dtype=np.float64)
    percentile = float(percentile)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration requires finite validation scores.")
    if not 0.0 < percentile < 100.0:
        raise ValueError("The validation percentile must lie between 0 and 100.")
    return float(np.percentile(scores, percentile))


def calculate_acceptance_rate(unknownness, threshold):
    """Return the fraction accepted as known using score less than or equal to tau."""
    scores = np.asarray(unknownness, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Acceptance measurement requires one-dimensional scores.")
    return float(np.mean(scores <= float(threshold)))


def calculate_rejection_rate(unknownness, threshold):
    """Return the fraction rejected as unknown using score greater than tau."""
    return 1.0 - calculate_acceptance_rate(unknownness, threshold)
