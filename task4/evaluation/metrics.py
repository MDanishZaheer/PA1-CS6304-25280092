# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate closed-set accuracy and all required open-set metrics."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from task4.evaluation.thresholds import (
    calculate_acceptance_rate,
    calculate_rejection_rate,
)


def calculate_closed_set_metrics(labels, known_logits):
    """Calculate CSA and macro-F1 from the ten known-class logits only."""
    label_array = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(known_logits, dtype=np.float64)
    if logits.ndim != 2 or label_array.ndim != 1:
        raise ValueError("Closed-set labels and logits have invalid dimensions.")
    if len(logits) != len(label_array) or len(logits) == 0:
        raise ValueError("Closed-set labels and logits must have matching lengths.")
    predictions = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "macro_f1": float(
            f1_score(
                label_array,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "number_of_examples": int(len(label_array)),
    }


def calculate_unknown_auroc(known_scores, unknown_scores):
    """Calculate AUROC with unknown examples as the positive class."""
    known = np.asarray(known_scores, dtype=np.float64)
    unknown = np.asarray(unknown_scores, dtype=np.float64)
    if known.ndim != 1 or unknown.ndim != 1 or not len(known) or not len(unknown):
        raise ValueError("AUROC requires non-empty one-dimensional score arrays.")
    labels = np.concatenate(
        [
            np.zeros(len(known), dtype=np.int64),
            np.ones(len(unknown), dtype=np.int64),
        ]
    )
    scores = np.concatenate([known, unknown])
    return float(roc_auc_score(labels, scores))


def calculate_osr_metrics(
    known_test_scores,
    near_unknown_scores,
    far_unknown_scores,
    threshold,
):
    """Calculate near, far, and combined AUROC plus one calibrated operating point."""
    known = np.asarray(known_test_scores, dtype=np.float64)
    near = np.asarray(near_unknown_scores, dtype=np.float64)
    far = np.asarray(far_unknown_scores, dtype=np.float64)
    all_unknown = np.concatenate([near, far])
    near_rejection = calculate_rejection_rate(near, threshold)
    far_rejection = calculate_rejection_rate(far, threshold)
    all_rejection = calculate_rejection_rate(all_unknown, threshold)
    return {
        "known_vs_near_auroc": calculate_unknown_auroc(known, near),
        "known_vs_far_auroc": calculate_unknown_auroc(known, far),
        "known_vs_all_unknown_auroc": calculate_unknown_auroc(
            known,
            all_unknown,
        ),
        "threshold": float(threshold),
        "known_test_acceptance_rate": calculate_acceptance_rate(
            known,
            threshold,
        ),
        "near_unknown_rejection_rate": near_rejection,
        "far_unknown_rejection_rate": far_rejection,
        "all_unknown_rejection_rate": all_rejection,
        "near_fpr_at_95_tpr": 1.0 - near_rejection,
        "far_fpr_at_95_tpr": 1.0 - far_rejection,
        "all_unknown_fpr_at_95_tpr": 1.0 - all_rejection,
    }


def save_json(data, output_file):
    """Save Task 4 metrics or metadata as formatted JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return output_path
