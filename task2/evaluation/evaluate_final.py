# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Lock experiments before exposing Sketch labels for final Task 2 evaluation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from shared_task_2and3.pacs_protocol import build_task2_datasets
from task2.configs.config_loader import get_output_paths
from task2.evaluation.class_analysis import (
    calculate_class_accuracy_changes,
    calculate_dominant_confusions,
    calculate_per_class_accuracy,
    create_failure_case_table,
    plot_confusion_matrix,
)
from task2.evaluation.domain_separability import (
    calculate_domain_separability,
    extract_backbone_features,
)
from task2.evaluation.metrics import (
    evaluate_classification_loader,
    evaluate_source_validation,
    save_json,
)
from task2.models.classifier_head import build_classifier
from task2.training.train import load_training_checkpoint, select_training_device


def calculate_file_sha256(file_path):
    """Calculate a stable SHA-256 digest for one saved checkpoint."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_experiment_lock(configurations, lock_file):
    """Record fixed configurations and checkpoint hashes before target evaluation."""
    locked_runs = {}
    for result_name, configuration in configurations.items():
        paths = get_output_paths(configuration)
        run_name = configuration["method"]["run_name"]
        checkpoint_file = paths["checkpoints_directory"] / f"{run_name}_best.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Missing checkpoint for {result_name}: {checkpoint_file}")
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("configuration") != configuration:
            raise ValueError(
                f"Checkpoint and current configuration differ for {result_name}."
            )
        configuration_text = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        locked_runs[result_name] = {
            "method": configuration["method"]["name"],
            "run_name": run_name,
            "checkpoint_file": str(checkpoint_file.resolve()),
            "checkpoint_sha256": calculate_file_sha256(checkpoint_file),
            "configuration_sha256": hashlib.sha256(configuration_text).hexdigest(),
            "configuration": configuration,
        }

    lock_data = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 6304,
        "target_domain": "sketch",
        "target_labels_used_for_selection": False,
        "runs": locked_runs,
    }
    save_json(lock_data, lock_file)
    print(f"Locked experiment decisions: {Path(lock_file).resolve()}")
    return lock_data


def validate_experiment_lock(configurations, lock_file):
    """Verify checkpoint and configuration hashes before loading target labels."""
    lock_path = Path(lock_file)
    if not lock_path.exists():
        raise FileNotFoundError(
            "Create the experiment lock before running final target evaluation."
        )
    with lock_path.open("r", encoding="utf-8") as file:
        lock_data = json.load(file)

    for result_name, configuration in configurations.items():
        if result_name not in lock_data["runs"]:
            raise ValueError(f"The experiment lock is missing run: {result_name}")
        locked_run = lock_data["runs"][result_name]
        checkpoint_file = Path(locked_run["checkpoint_file"])
        if calculate_file_sha256(checkpoint_file) != locked_run["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after locking: {result_name}")
        configuration_text = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        current_config_hash = hashlib.sha256(configuration_text).hexdigest()
        if current_config_hash != locked_run["configuration_sha256"]:
            raise ValueError(f"Configuration changed after locking: {result_name}")
    return lock_data


def create_evaluation_loader(dataset, configuration, generator_offset=0):
    """Create a deterministic non-shuffled loader for final evaluation."""
    training = configuration["training"]
    generator = torch.Generator()
    generator.manual_seed(int(configuration["seed"]) + generator_offset)
    number_of_workers = int(training["number_of_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=number_of_workers,
        pin_memory=bool(training["pin_memory"] and torch.cuda.is_available()),
        generator=generator,
        persistent_workers=number_of_workers > 0,
    )


def _flatten_method_metrics(
    result_name,
    source_metrics,
    target_metrics,
    separability_metrics,
):
    """Create one row for the assignment's main comparison table."""
    row = {
        "result_name": result_name,
        "mean_source_accuracy": source_metrics["mean_accuracy"],
        "mean_source_macro_f1": source_metrics["mean_macro_f1"],
        "target_accuracy": target_metrics["accuracy"],
        "target_macro_f1": target_metrics["macro_f1"],
        "domain_separability": separability_metrics["accuracy"],
    }
    for domain, metrics in source_metrics["domains"].items():
        row[f"{domain}_validation_accuracy"] = metrics["accuracy"]
        row[f"{domain}_validation_macro_f1"] = metrics["macro_f1"]
    return row


def evaluate_locked_method(
    result_name,
    configuration,
    datasets,
    device,
):
    """Load one fixed checkpoint and calculate all common final measurements."""
    paths = get_output_paths(configuration)
    run_name = configuration["method"]["run_name"]
    checkpoint_file = paths["checkpoints_directory"] / f"{run_name}_best.pt"
    model = build_classifier(configuration, device)
    load_training_checkpoint(checkpoint_file, model, device=device)

    validation_loaders = {
        domain: create_evaluation_loader(dataset, configuration, index)
        for index, (domain, dataset) in enumerate(
            datasets["source_validation"].items()
        )
    }
    target_loader = create_evaluation_loader(
        datasets["target_evaluation"],
        configuration,
        generator_offset=100,
    )
    target_diagnostic_loader = create_evaluation_loader(
        datasets["target_diagnostic"],
        configuration,
        generator_offset=200,
    )

    source_metrics = evaluate_source_validation(model, validation_loaders, device)
    target_metrics, target_predictions, _ = evaluate_classification_loader(
        model,
        target_loader,
        device,
    )
    source_feature_batches = []
    for validation_loader in validation_loaders.values():
        features, _ = extract_backbone_features(model, validation_loader, device)
        source_feature_batches.append(features)
    source_features = np.concatenate(source_feature_batches, axis=0)
    target_features, _ = extract_backbone_features(
        model,
        target_diagnostic_loader,
        device,
    )
    evaluation = configuration["evaluation"]
    separability_metrics = calculate_domain_separability(
        source_features,
        target_features,
        seed=int(configuration["seed"]),
        training_ratio=float(evaluation["diagnostic_train_ratio"]),
        logistic_regression_c=float(evaluation["logistic_regression_c"]),
    )
    return {
        "result_name": result_name,
        "source_metrics": source_metrics,
        "target_metrics": target_metrics,
        "target_predictions": target_predictions,
        "domain_separability": separability_metrics,
        "summary_row": _flatten_method_metrics(
            result_name,
            source_metrics,
            target_metrics,
            separability_metrics,
        ),
    }


def save_final_class_analysis(method_results, class_names, output_paths):
    """Save target per-class changes, confusions, and failure-case tables."""
    baseline_predictions = method_results["source_only"]["target_predictions"]
    baseline_table = calculate_per_class_accuracy(
        baseline_predictions["label"],
        baseline_predictions["prediction"],
        class_names,
    )
    analysis = {}
    for result_name, result in method_results.items():
        predictions = result["target_predictions"]
        per_class_table = calculate_per_class_accuracy(
            predictions["label"],
            predictions["prediction"],
            class_names,
        )
        change_table = calculate_class_accuracy_changes(
            baseline_table,
            per_class_table,
            result_name,
        )
        confusion_table, confusion_matrix = calculate_dominant_confusions(
            predictions["label"],
            predictions["prediction"],
            class_names,
        )
        failure_table = create_failure_case_table(predictions, class_names)

        per_class_table.to_csv(
            output_paths["metrics_directory"] / f"{result_name}_target_per_class.csv",
            index=False,
        )
        change_table.to_csv(
            output_paths["metrics_directory"] / f"{result_name}_class_changes.csv",
            index=False,
        )
        confusion_table.to_csv(
            output_paths["metrics_directory"] / f"{result_name}_dominant_confusions.csv",
            index=False,
        )
        failure_table.to_csv(
            output_paths["predictions_directory"] / f"{result_name}_failure_cases.csv",
            index=False,
        )
        plot_confusion_matrix(
            confusion_matrix,
            class_names,
            title=f"{result_name} target confusion matrix",
            output_file=(
                output_paths["figures_directory"]
                / f"{result_name}_target_confusion_matrix.png"
            ),
        )
        analysis[result_name] = {
            "largest_improvement": change_table.iloc[0].to_dict(),
            "largest_degradation": change_table.iloc[-1].to_dict(),
        }
    return analysis


def run_final_evaluation(
    configurations,
    image_root,
    protocol,
    lock_file,
    device=None,
):
    """Evaluate fixed checkpoints after deliberately unlocking Sketch labels."""
    if "source_only" not in configurations:
        raise ValueError("Final comparison requires the Source-only baseline.")
    validate_experiment_lock(configurations, lock_file)
    selected_device = select_training_device(device)
    reference_configuration = configurations["source_only"]
    transforms = reference_configuration["transforms"]

    datasets = build_task2_datasets(
        image_root,
        protocol,
        resize_size=int(transforms["resize_size"]),
        crop_size=int(transforms["crop_size"]),
        include_target_labels=True,
    )
    method_results = {
        result_name: evaluate_locked_method(
            result_name,
            configuration,
            datasets,
            selected_device,
        )
        for result_name, configuration in configurations.items()
    }

    baseline_accuracy = method_results["source_only"]["target_metrics"]["accuracy"]
    summary_rows = []
    for result in method_results.values():
        row = result["summary_row"]
        row["target_accuracy_change_from_source_only"] = (
            row["target_accuracy"] - baseline_accuracy
        )
        summary_rows.append(row)

    output_paths = get_output_paths(reference_configuration)
    summary_table = pd.DataFrame(summary_rows)
    summary_file = output_paths["metrics_directory"] / "task2_final_summary.csv"
    summary_table.to_csv(summary_file, index=False)
    for result_name, result in method_results.items():
        result["target_predictions"].to_csv(
            output_paths["predictions_directory"]
            / f"{result_name}_target_predictions.csv",
            index=False,
        )

    class_analysis = save_final_class_analysis(
        method_results,
        reference_configuration["dataset"]["class_names"],
        output_paths,
    )
    metric_record = {
        result_name: {
            "source_validation": result["source_metrics"],
            "target": result["target_metrics"],
            "domain_separability": result["domain_separability"],
        }
        for result_name, result in method_results.items()
    }
    metric_record["class_analysis_summary"] = class_analysis
    save_json(
        metric_record,
        output_paths["metrics_directory"] / "task2_final_metrics.json",
    )
    print(f"Saved final comparison: {summary_file.resolve()}")
    return summary_table, method_results, class_analysis
