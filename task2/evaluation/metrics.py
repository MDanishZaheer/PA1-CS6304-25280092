# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate and save Task 2 classification metrics and predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


def to_numpy(values):
    """Move a tensor to CPU or convert another array-like object."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def calculate_classification_metrics(labels, predictions):
    """Calculate top-1 accuracy and macro-F1 from integer predictions."""
    labels_array = to_numpy(labels).astype(np.int64)
    predictions_array = to_numpy(predictions).astype(np.int64)
    if labels_array.ndim != 1 or predictions_array.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional.")
    if len(labels_array) == 0 or labels_array.shape != predictions_array.shape:
        raise ValueError("Labels and predictions must have the same non-zero length.")
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "macro_f1": float(
            f1_score(
                labels_array,
                predictions_array,
                average="macro",
                zero_division=0,
            )
        ),
        "number_of_examples": int(len(labels_array)),
    }


@torch.inference_mode()
def collect_classification_predictions(model, data_loader, device):
    """Collect labels, predictions, logits, and identifiers from a labeled loader."""
    model.eval()
    all_labels = []
    all_predictions = []
    all_logits = []
    all_identifiers = []

    for batch in data_loader:
        if "label" not in batch:
            raise ValueError("Classification evaluation requires a labeled dataset.")
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)

        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_logits.append(logits.float().cpu())
        all_identifiers.extend(str(value) for value in batch["identifier"])

    if not all_labels:
        raise ValueError("Cannot evaluate an empty data loader.")
    return {
        "labels": torch.cat(all_labels).numpy(),
        "predictions": torch.cat(all_predictions).numpy(),
        "logits": torch.cat(all_logits).numpy(),
        "identifiers": all_identifiers,
    }


def evaluate_classification_loader(model, data_loader, device):
    """Evaluate one labeled loader and return metrics with prediction records."""
    outputs = collect_classification_predictions(model, data_loader, device)
    metrics = calculate_classification_metrics(
        outputs["labels"],
        outputs["predictions"],
    )
    prediction_table = pd.DataFrame(
        {
            "identifier": outputs["identifiers"],
            "label": outputs["labels"],
            "prediction": outputs["predictions"],
        }
    )
    return metrics, prediction_table, outputs["logits"]


def evaluate_source_validation(model, validation_loaders, device):
    """Evaluate each source domain and calculate its mean selection metrics."""
    domain_metrics = {}
    for domain, data_loader in validation_loaders.items():
        metrics, _, _ = evaluate_classification_loader(model, data_loader, device)
        domain_metrics[domain] = metrics

    mean_accuracy = float(
        np.mean([metrics["accuracy"] for metrics in domain_metrics.values()])
    )
    mean_macro_f1 = float(
        np.mean([metrics["macro_f1"] for metrics in domain_metrics.values()])
    )
    return {
        "domains": domain_metrics,
        "mean_accuracy": mean_accuracy,
        "mean_macro_f1": mean_macro_f1,
    }


def make_json_serializable(value):
    """Convert arrays, tensors, and paths into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data, output_file):
    """Save metric or metadata dictionaries as formatted JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(data), file, indent=2)
