# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate and select models without loading the Sketch domain."""

from pathlib import Path

import numpy as np
import torch

from task2.evaluation.metrics import evaluate_classification_loader


def evaluate_source_validation(model, validation_loaders, device):
    """Calculate per-source, mean-source, and worst-source validation metrics."""
    domain_metrics = {}
    for domain, data_loader in validation_loaders.items():
        metrics, _, _ = evaluate_classification_loader(model, data_loader, device)
        domain_metrics[domain] = metrics

    accuracies = [metrics["accuracy"] for metrics in domain_metrics.values()]
    macro_f1_values = [metrics["macro_f1"] for metrics in domain_metrics.values()]
    return {
        "domains": domain_metrics,
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "worst_accuracy": float(np.min(accuracies)),
        "worst_macro_f1": float(np.min(macro_f1_values)),
    }


def get_source_selection_value(validation_metrics):
    """Return the assignment's mean source-validation macro-F1 criterion."""
    return float(validation_metrics["mean_macro_f1"])


def load_erm_checkpoint(checkpoint_file, model, device):
    """Load and validate the unchanged Task 2 Source-only checkpoint."""
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Task 3 ERM requires the Task 2 Source-only checkpoint: "
            f"{checkpoint_path.resolve()}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint_method = checkpoint.get("configuration", {}).get("method", {}).get(
        "name"
    )
    if checkpoint_method != "source_only":
        raise ValueError("The Task 3 ERM checkpoint is not Task 2 Source-only.")
    if checkpoint.get("selection_metric") != "mean_source_validation_macro_f1":
        raise ValueError("The reused ERM checkpoint used the wrong selection metric.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
