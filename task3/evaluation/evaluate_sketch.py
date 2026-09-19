# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate Sketch only after all Task 3 decisions are cryptographically locked."""

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from shared_task_2and3.pacs import (
    PACSDomainDataset,
    create_evaluation_transform,
    find_pacs_image_root,
    scan_domain_records,
)
from task2.evaluation.metrics import evaluate_classification_loader, save_json
from task3.configs.config_loader import (
    create_output_directories,
    get_method_checkpoint_path,
    get_output_paths,
)
from task3.evaluation.class_analysis import (
    calculate_class_accuracy_changes,
    calculate_dominant_confusions,
    calculate_per_class_accuracy,
    create_failure_case_table,
    plot_confusion_matrix,
)
from task3.evaluation.domain_metrics import (
    create_evaluation_loader,
    evaluate_source_model,
)
from task3.evaluation.sharpness import (
    calculate_local_sharpness,
    create_fixed_validation_batch,
)
from task3.evaluation.source_domain_separability import (
    evaluate_source_domain_separability,
)
from task3.evaluation.task2_comparison import (
    create_task2_task3_class_comparison,
    create_task2_task3_summary,
)
from task3.training.train import select_training_device


def calculate_file_sha256(file_path):
    """Calculate the SHA-256 digest of one fixed model checkpoint."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_for_lock(configuration, checkpoint):
    """Verify ERM provenance or exact Task 3 training configuration."""
    method_name = configuration["method"]["name"]
    checkpoint_configuration = checkpoint.get("configuration", {})
    if method_name == "erm":
        checkpoint_method = checkpoint_configuration.get("method", {}).get("name")
        if checkpoint_method != "source_only":
            raise ValueError("Task 3 ERM must use the Task 2 Source-only checkpoint.")
    else:
        if checkpoint_configuration != configuration:
            raise ValueError(
                f"Checkpoint configuration does not match {method_name}."
            )
        if checkpoint.get("sketch_used_during_training") is not False:
            raise ValueError("A Task 3 checkpoint must certify no Sketch usage.")
    if checkpoint.get("selection_metric") != "mean_source_validation_macro_f1":
        raise ValueError("Every checkpoint must use source macro-F1 selection.")


def _load_completed_hypotheses(hypotheses_file):
    """Load student predictions and reject blank or TODO entries before locking."""
    hypotheses_path = Path(hypotheses_file)
    if not hypotheses_path.exists():
        raise FileNotFoundError(
            "Save the pre-Sketch Task 3 hypotheses before creating the lock: "
            f"{hypotheses_path.resolve()}"
        )
    with hypotheses_path.open("r", encoding="utf-8") as file:
        hypotheses = json.load(file)
    if not isinstance(hypotheses, dict) or not hypotheses:
        raise ValueError("Task 3 hypotheses must be a non-empty JSON mapping.")
    incomplete_keys = [
        key
        for key, value in hypotheses.items()
        if not isinstance(value, str)
        or not value.strip()
        or "TODO" in value.upper()
    ]
    if incomplete_keys:
        raise ValueError(
            "Replace the pre-Sketch hypothesis placeholders before locking: "
            + ", ".join(incomplete_keys)
        )
    return hypotheses_path, hypotheses


def create_experiment_lock(configurations, lock_file):
    """Lock every checkpoint and configuration before exposing Sketch."""
    if not configurations or "erm" not in configurations:
        raise ValueError("The Task 3 lock requires at least the shared ERM run.")
    reference_paths = get_output_paths(configurations["erm"])
    hypotheses_file, hypotheses = _load_completed_hypotheses(
        reference_paths["metrics_directory"] / "task3_hypotheses.json"
    )
    locked_runs = {}
    for result_name, configuration in configurations.items():
        checkpoint_file = get_method_checkpoint_path(configuration)
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {result_name}: {checkpoint_file}"
            )
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        _validate_checkpoint_for_lock(configuration, checkpoint)
        configuration_text = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        locked_runs[result_name] = {
            "method": configuration["method"]["name"],
            "run_name": configuration["method"]["run_name"],
            "checkpoint_file": str(checkpoint_file.resolve()),
            "checkpoint_sha256": calculate_file_sha256(checkpoint_file),
            "configuration_sha256": hashlib.sha256(configuration_text).hexdigest(),
            "configuration": configuration,
        }
        del checkpoint

    lock_data = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 6304,
        "target_domain": "sketch",
        "sketch_used_for_training_or_selection": False,
        "hypotheses_file": str(hypotheses_file.resolve()),
        "hypotheses_sha256": calculate_file_sha256(hypotheses_file),
        "hypotheses": hypotheses,
        "runs": locked_runs,
    }
    save_json(lock_data, lock_file)
    print(f"Locked Task 3 decisions: {Path(lock_file).resolve()}")
    return lock_data


def validate_experiment_lock(configurations, lock_file):
    """Reject any checkpoint or configuration changed after the lock."""
    lock_path = Path(lock_file)
    if not lock_path.exists():
        raise FileNotFoundError(
            "Create the Task 3 experiment lock before loading Sketch."
        )
    with lock_path.open("r", encoding="utf-8") as file:
        lock_data = json.load(file)
    if lock_data.get("sketch_used_for_training_or_selection") is not False:
        raise ValueError("The Task 3 lock does not certify target-free selection.")
    hypotheses_file = Path(lock_data["hypotheses_file"])
    if calculate_file_sha256(hypotheses_file) != lock_data["hypotheses_sha256"]:
        raise ValueError("Pre-Sketch hypotheses changed after locking.")

    for result_name, configuration in configurations.items():
        if result_name not in lock_data["runs"]:
            raise ValueError(f"The Task 3 lock is missing run: {result_name}")
        locked_run = lock_data["runs"][result_name]
        checkpoint_file = Path(locked_run["checkpoint_file"])
        if calculate_file_sha256(checkpoint_file) != locked_run["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed after locking: {result_name}")
        configuration_text = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        current_hash = hashlib.sha256(configuration_text).hexdigest()
        if current_hash != locked_run["configuration_sha256"]:
            raise ValueError(f"Configuration changed after locking: {result_name}")
    return lock_data


def build_sketch_evaluation_dataset(image_root, configuration):
    """Construct the labeled Sketch dataset only inside final evaluation."""
    verified_root = find_pacs_image_root(image_root)
    records = scan_domain_records(
        verified_root,
        configuration["dataset"]["target_domain"],
        include_labels=True,
    )
    transform_settings = configuration["transforms"]
    evaluation_transform = create_evaluation_transform(
        resize_size=int(transform_settings["resize_size"]),
        crop_size=int(transform_settings["crop_size"]),
    )
    return PACSDomainDataset(
        verified_root,
        records,
        transform=evaluation_transform,
        include_labels=True,
    )


def _flatten_result_row(
    result_name,
    source_metrics,
    sketch_metrics,
    separability,
    sharpness,
):
    """Flatten one method into the complete Task 3 comparison-table row."""
    row = {
        "result_name": result_name,
        "mean_source_accuracy": source_metrics["mean_accuracy"],
        "mean_source_macro_f1": source_metrics["mean_macro_f1"],
        "worst_source_accuracy": source_metrics["worst_accuracy"],
        "worst_source_macro_f1": source_metrics["worst_macro_f1"],
        "sketch_accuracy": sketch_metrics["accuracy"],
        "sketch_macro_f1": sketch_metrics["macro_f1"],
        "source_domain_separability": separability["accuracy"],
        "sharpness_increase": sharpness["sharpness_increase"],
    }
    for domain, metrics in source_metrics["domains"].items():
        row[f"{domain}_validation_accuracy"] = metrics["accuracy"]
        row[f"{domain}_validation_macro_f1"] = metrics["macro_f1"]
    return row


def evaluate_locked_method(
    result_name,
    configuration,
    source_datasets,
    sketch_loader,
    fixed_sharpness_batch,
    device,
):
    """Evaluate source metrics, diagnostics, and final Sketch recognition."""
    source_result = evaluate_source_model(
        configuration,
        source_datasets,
        device,
    )
    model = source_result["model"]
    validation_loaders = source_result["validation_loaders"]
    diagnostics = configuration["diagnostics"]
    separability = evaluate_source_domain_separability(
        model,
        validation_loaders,
        device,
        seed=int(configuration["seed"]),
        training_ratio=float(diagnostics["diagnostic_train_ratio"]),
        logistic_regression_c=float(diagnostics["logistic_regression_c"]),
    )
    sharpness = calculate_local_sharpness(
        model,
        fixed_sharpness_batch,
        device,
        radius=float(diagnostics["sharpness_radius"]),
    )
    sketch_metrics, sketch_predictions, _ = evaluate_classification_loader(
        model,
        sketch_loader,
        device,
    )
    row = _flatten_result_row(
        result_name,
        source_result["metrics"],
        sketch_metrics,
        separability,
        sharpness,
    )
    source_metrics = source_result["metrics"]

    model.to("cpu")
    del source_result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "source_metrics": source_metrics,
        "sketch_metrics": sketch_metrics,
        "sketch_predictions": sketch_predictions,
        "source_domain_separability": separability,
        "sharpness": sharpness,
        "summary_row": row,
    }


def save_sketch_class_analysis(method_results, class_names, output_paths):
    """Save per-class changes, confusions, failure cases, and figures."""
    baseline_predictions = method_results["erm"]["sketch_predictions"]
    baseline_table = calculate_per_class_accuracy(
        baseline_predictions["label"],
        baseline_predictions["prediction"],
        class_names,
    )
    analysis_summary = {}

    for result_name, result in method_results.items():
        predictions = result["sketch_predictions"]
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
        confusion_table, confusion_values = calculate_dominant_confusions(
            predictions["label"],
            predictions["prediction"],
            class_names,
        )
        failure_table = create_failure_case_table(predictions, class_names)

        per_class_table.to_csv(
            output_paths["metrics_directory"]
            / f"{result_name}_sketch_per_class.csv",
            index=False,
        )
        change_table.to_csv(
            output_paths["metrics_directory"]
            / f"{result_name}_class_changes.csv",
            index=False,
        )
        confusion_table.to_csv(
            output_paths["metrics_directory"]
            / f"{result_name}_dominant_confusions.csv",
            index=False,
        )
        failure_table.to_csv(
            output_paths["predictions_directory"]
            / f"{result_name}_failure_cases.csv",
            index=False,
        )
        plot_confusion_matrix(
            confusion_values,
            class_names,
            title=f"{result_name} Sketch confusion matrix",
            output_file=(
                output_paths["figures_directory"]
                / f"{result_name}_sketch_confusion_matrix.png"
            ),
        )
        analysis_summary[result_name] = {
            "largest_improvement": change_table.iloc[0].to_dict(),
            "largest_degradation": change_table.iloc[-1].to_dict(),
        }
    return analysis_summary


def save_cross_task_comparisons(configuration, task3_summary, output_paths):
    """Save Task 2 DAN versus Task 3 DAN-DG aggregate and class tables."""
    dependency_paths = get_output_paths(configuration)
    summary_comparison = create_task2_task3_summary(
        dependency_paths["task2_final_summary"],
        task3_summary,
    )
    if summary_comparison is not None:
        summary_comparison.to_csv(
            output_paths["metrics_directory"] / "task2_task3_comparison.csv",
            index=False,
        )

    class_comparison = create_task2_task3_class_comparison(
        dependency_paths["task2_metrics_directory"],
        output_paths["metrics_directory"],
    )
    if class_comparison is not None:
        class_comparison.to_csv(
            output_paths["metrics_directory"]
            / "task2_dan_vs_task3_dan_dg_classes.csv",
            index=False,
        )
    return summary_comparison, class_comparison


def run_final_sketch_evaluation(
    configurations,
    source_datasets,
    image_root,
    lock_file,
    device=None,
):
    """Run the one locked Task 3 evaluation that is allowed to load Sketch."""
    if "erm" not in configurations:
        raise ValueError("Task 3 final evaluation requires the shared ERM baseline.")
    reference_configuration = configurations["erm"]
    expected_domains = set(reference_configuration["dataset"]["source_domains"])
    if set(source_datasets) != {"source_train", "source_validation"}:
        raise ValueError("Final Task 3 evaluation requires source-only datasets.")
    for split_name in ("source_train", "source_validation"):
        if set(source_datasets[split_name]) != expected_domains:
            raise ValueError(
                f"Final Task 3 {split_name} contains an unexpected domain."
            )
    validate_experiment_lock(configurations, lock_file)
    selected_device = select_training_device(device)
    output_paths = create_output_directories(reference_configuration)

    fixed_batch, fixed_indices = create_fixed_validation_batch(
        source_datasets["source_validation"],
        examples_per_domain=int(
            reference_configuration["diagnostics"][
                "sharpness_examples_per_domain"
            ]
        ),
        seed=int(reference_configuration["seed"]),
    )
    save_json(
        fixed_indices,
        output_paths["metrics_directory"] / "sharpness_batch_indices.json",
    )

    sketch_dataset = build_sketch_evaluation_dataset(
        image_root,
        reference_configuration,
    )
    sketch_loader = create_evaluation_loader(
        sketch_dataset,
        reference_configuration,
        generator_offset=500,
    )
    method_results = {
        result_name: evaluate_locked_method(
            result_name,
            configuration,
            source_datasets,
            sketch_loader,
            fixed_batch,
            selected_device,
        )
        for result_name, configuration in configurations.items()
    }

    erm_accuracy = method_results["erm"]["sketch_metrics"]["accuracy"]
    summary_rows = []
    for result in method_results.values():
        row = result["summary_row"]
        row["sketch_accuracy_change_from_erm"] = (
            row["sketch_accuracy"] - erm_accuracy
        )
        summary_rows.append(row)
    summary_table = pd.DataFrame(summary_rows)
    summary_file = output_paths["metrics_directory"] / "task3_final_summary.csv"
    summary_table.to_csv(summary_file, index=False)

    for result_name, result in method_results.items():
        result["sketch_predictions"].to_csv(
            output_paths["predictions_directory"]
            / f"{result_name}_sketch_predictions.csv",
            index=False,
        )
    class_analysis = save_sketch_class_analysis(
        method_results,
        reference_configuration["dataset"]["class_names"],
        output_paths,
    )
    diagnostic_record = {
        result_name: {
            "source_domain_separability": result[
                "source_domain_separability"
            ],
            "sharpness": result["sharpness"],
        }
        for result_name, result in method_results.items()
    }
    save_json(
        diagnostic_record,
        output_paths["metrics_directory"] / "task3_diagnostics.json",
    )
    save_json(
        class_analysis,
        output_paths["metrics_directory"] / "task3_class_analysis_summary.json",
    )
    task2_comparison, task2_class_comparison = save_cross_task_comparisons(
        reference_configuration,
        summary_table,
        output_paths,
    )
    print(f"Saved final Task 3 comparison: {summary_file.resolve()}")
    return {
        "summary": summary_table,
        "method_results": method_results,
        "class_analysis": class_analysis,
        "task2_comparison": task2_comparison,
        "task2_class_comparison": task2_class_comparison,
    }
