# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Lock all Task 4 decisions before evaluating fixed CIFAR-100 unknowns."""

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from task4.configs.config_loader import (
    create_output_directories,
    get_checkpoint_path,
    get_known_feature_path,
    get_known_output_path,
    get_output_paths,
)
from task4.evaluation.extract_outputs import (
    collect_model_outputs,
    create_evaluation_loader,
    create_prediction_table,
    load_output_cache,
    save_feature_cache,
    save_output_cache,
)
from task4.evaluation.failure_analysis import (
    create_accepted_unknown_failure_table,
    plot_accepted_unknown_failures,
)
from task4.evaluation.metrics import (
    calculate_closed_set_metrics,
    calculate_osr_metrics,
    save_json,
)
from task4.evaluation.thresholds import calculate_validation_threshold
from task4.models.resnet_cifar import build_model
from task4.scores.energy import calculate_energy_unknownness
from task4.scores.mahalanobis import (
    calculate_mahalanobis_unknownness,
    fit_mahalanobis_statistics,
    load_mahalanobis_statistics,
    save_mahalanobis_statistics,
)
from task4.scores.mls import calculate_mls_unknownness
from task4.scores.msp import calculate_msp_unknownness
from task4.scores.proser_placeholder import (
    calculate_dummy_calibration_bias,
    calculate_proser_placeholder_unknownness,
)
from task4.training.train import (
    load_training_checkpoint,
    select_training_device,
)


SCORE_PLAN = {
    "vanilla": ("msp", "mls", "energy", "mahalanobis"),
    "gcsc": ("mls",),
    "proser": ("mls", "placeholder"),
}
KNOWN_CACHE_SPLITS = {
    "vanilla": ("train", "validation", "test"),
    "gcsc": ("validation", "test"),
    "proser": ("validation", "test"),
}


def calculate_file_sha256(file_path):
    """Calculate the SHA-256 digest of one fixed experiment artifact."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_for_evaluation(configuration, device):
    """Build and load one source-selected Task 4 model checkpoint."""
    model = build_model(configuration, device)
    checkpoint_file = get_checkpoint_path(configuration)
    checkpoint = load_training_checkpoint(
        checkpoint_file,
        model,
        configuration,
        device,
    )
    model.eval()
    return model, checkpoint, checkpoint_file


def calculate_unknownness(
    score_name,
    outputs,
    configuration,
    mahalanobis_statistics=None,
    proser_calibration_bias=None,
):
    """Calculate one registered score, always with larger values as more novel."""
    if score_name == "msp":
        return calculate_msp_unknownness(outputs["known_logits"])
    if score_name == "mls":
        return calculate_mls_unknownness(outputs["known_logits"])
    if score_name == "energy":
        return calculate_energy_unknownness(
            outputs["known_logits"],
            temperature=configuration["evaluation"]["energy_temperature"],
        )
    if score_name == "mahalanobis":
        if mahalanobis_statistics is None:
            raise ValueError("Mahalanobis scoring requires fitted training statistics.")
        return calculate_mahalanobis_unknownness(
            outputs["features"],
            mahalanobis_statistics,
        )
    if score_name == "placeholder":
        if proser_calibration_bias is None:
            raise ValueError("PROSER placeholder scoring requires a validation bias.")
        return calculate_proser_placeholder_unknownness(
            outputs["known_logits"],
            outputs["dummy_logits"],
            calibration_bias=proser_calibration_bias,
            temperature=configuration["evaluation"]["proser_temperature"],
        )
    raise ValueError(f"Unknown Task 4 score: {score_name}")


def _extract_known_split(model, dataset, configuration, split_name, device):
    """Extract and save one deterministic CIFAR-10 output and feature cache."""
    loader = create_evaluation_loader(
        dataset,
        configuration,
        generator_offset={"train": 0, "validation": 100, "test": 200}[split_name],
    )
    outputs = collect_model_outputs(model, loader, device)
    output_file = get_known_output_path(configuration, split_name)
    feature_file = get_known_feature_path(configuration, split_name)
    save_output_cache(outputs, output_file)
    save_feature_cache(outputs, feature_file)
    return outputs, output_file, feature_file


def prepare_known_evaluation(
    configurations,
    datasets_by_method,
    device=None,
):
    """Freeze CIFAR-10 outputs, statistics, scores, and thresholds before unknowns."""
    if set(configurations) != set(SCORE_PLAN):
        raise ValueError("Known evaluation requires Vanilla, GCSC, and PROSER configs.")
    if set(datasets_by_method) != set(configurations):
        raise ValueError("Each Task 4 method requires its known-only dataset mapping.")
    selected_device = select_training_device(device)
    reference_configuration = configurations["vanilla"]
    paths = create_output_directories(reference_configuration)
    known_outputs = {}
    cache_files = {}

    for method_name, configuration in configurations.items():
        model, checkpoint, checkpoint_file = load_model_for_evaluation(
            configuration,
            selected_device,
        )
        method_datasets = datasets_by_method[method_name]
        required_dataset_keys = {
            "train",
            "train_evaluation",
            "validation",
            "test",
        }
        if set(method_datasets) != required_dataset_keys:
            raise ValueError("Known output extraction received an unknown dataset.")
        known_outputs[method_name] = {}
        cache_files[method_name] = {}
        for split_name in KNOWN_CACHE_SPLITS[method_name]:
            dataset_key = "train_evaluation" if split_name == "train" else split_name
            outputs, output_file, feature_file = _extract_known_split(
                model,
                method_datasets[dataset_key],
                configuration,
                split_name,
                selected_device,
            )
            known_outputs[method_name][split_name] = outputs
            cache_files[method_name][split_name] = {
                "outputs": output_file,
                "features": feature_file,
            }
        model.to("cpu")
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Cached known outputs for {method_name}: {checkpoint_file.resolve()}")

    reference_validation_ids = known_outputs["vanilla"]["validation"]["identifiers"]
    reference_test_ids = known_outputs["vanilla"]["test"]["identifiers"]
    for method_name in ("gcsc", "proser"):
        if not np.array_equal(
            reference_validation_ids,
            known_outputs[method_name]["validation"]["identifiers"],
        ):
            raise ValueError("Methods received different CIFAR-10 validation examples.")
        if not np.array_equal(
            reference_test_ids,
            known_outputs[method_name]["test"]["identifiers"],
        ):
            raise ValueError("Methods received different CIFAR-10 test examples.")

    mahalanobis_statistics = fit_mahalanobis_statistics(
        known_outputs["vanilla"]["train"]["features"],
        known_outputs["vanilla"]["train"]["labels"],
        number_of_classes=reference_configuration["model"][
            "number_of_known_classes"
        ],
        epsilon=reference_configuration["evaluation"]["mahalanobis_epsilon"],
    )
    mahalanobis_file = (
        paths["mahalanobis_cache_directory"] / "vanilla_statistics.npz"
    )
    save_mahalanobis_statistics(mahalanobis_statistics, mahalanobis_file)

    proser_validation = known_outputs["proser"]["validation"]
    proser_bias = calculate_dummy_calibration_bias(
        proser_validation["known_logits"],
        proser_validation["dummy_logits"],
        known_acceptance_percent=reference_configuration["evaluation"][
            "threshold_percentile"
        ],
    )
    threshold_registry = {}
    percentile = reference_configuration["evaluation"]["threshold_percentile"]
    for method_name, score_names in SCORE_PLAN.items():
        configuration = configurations[method_name]
        validation_outputs = known_outputs[method_name]["validation"]
        for score_name in score_names:
            validation_scores = calculate_unknownness(
                score_name,
                validation_outputs,
                configuration,
                mahalanobis_statistics=mahalanobis_statistics,
                proser_calibration_bias=proser_bias,
            )
            result_name = f"{method_name}_{score_name}"
            threshold_registry[result_name] = {
                "model": method_name,
                "score": score_name,
                "threshold": calculate_validation_threshold(
                    validation_scores,
                    percentile=percentile,
                ),
                "validation_percentile": float(percentile),
                "number_of_validation_examples": int(len(validation_scores)),
            }
            if score_name == "placeholder":
                threshold_registry[result_name]["dummy_calibration_bias"] = float(
                    proser_bias
                )

    thresholds_file = paths["metrics_directory"] / "task4_thresholds.json"
    save_json(threshold_registry, thresholds_file)
    closed_set_rows = []
    for method_name, outputs_by_split in known_outputs.items():
        test_metrics = calculate_closed_set_metrics(
            outputs_by_split["test"]["labels"],
            outputs_by_split["test"]["known_logits"],
        )
        closed_set_rows.append(
            {
                "model": method_name,
                "cifar10_test_accuracy": test_metrics["accuracy"],
                "cifar10_test_macro_f1": test_metrics["macro_f1"],
            }
        )
        create_prediction_table(outputs_by_split["test"]).to_csv(
            paths["predictions_directory"]
            / f"{method_name}_cifar10_test_predictions.csv",
            index=False,
        )
    closed_set_table = pd.DataFrame(closed_set_rows)
    closed_set_table.to_csv(
        paths["metrics_directory"] / "known_closed_set_results.csv",
        index=False,
    )
    return {
        "known_outputs": known_outputs,
        "cache_files": cache_files,
        "mahalanobis_statistics": mahalanobis_statistics,
        "mahalanobis_file": mahalanobis_file,
        "thresholds": threshold_registry,
        "thresholds_file": thresholds_file,
        "closed_set_table": closed_set_table,
    }


def _load_completed_hypotheses(hypotheses_file):
    """Reject missing, blank, or TODO hypotheses before unknown evaluation."""
    hypotheses_path = Path(hypotheses_file)
    if not hypotheses_path.exists():
        raise FileNotFoundError(
            "Save Task 4 hypotheses before locking: "
            f"{hypotheses_path.resolve()}"
        )
    with hypotheses_path.open("r", encoding="utf-8") as file:
        hypotheses = json.load(file)
    if not isinstance(hypotheses, dict) or not hypotheses:
        raise ValueError("Task 4 hypotheses must be a non-empty JSON mapping.")
    incomplete = [
        key
        for key, value in hypotheses.items()
        if not isinstance(value, str)
        or not value.strip()
        or "TODO" in value.upper()
    ]
    if incomplete:
        raise ValueError(
            "Replace Task 4 hypothesis placeholders before locking: "
            + ", ".join(incomplete)
        )
    return hypotheses_path, hypotheses


def _validate_checkpoint_for_lock(configuration, checkpoint):
    """Verify exact configuration, selection, and absence of CIFAR-100 use."""
    if checkpoint.get("configuration") != configuration:
        raise ValueError("A Task 4 checkpoint does not match its configuration.")
    if checkpoint.get("selection_metric") != "validation_accuracy":
        raise ValueError("Task 4 checkpoint selection must use validation accuracy.")
    if checkpoint.get("cifar100_used_during_training") is not False:
        raise ValueError("A Task 4 checkpoint does not certify known-only training.")


def _register_locked_file(file_path):
    """Return an absolute path and hash for one immutable artifact."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Required lock artifact is missing: {path.resolve()}")
    return {
        "path": str(path.resolve()),
        "sha256": calculate_file_sha256(path),
    }


def create_experiment_lock(configurations, lock_file):
    """Hash models, known outputs, thresholds, scores, protocol, and hypotheses."""
    if set(configurations) != set(SCORE_PLAN):
        raise ValueError("The Task 4 lock requires Vanilla, GCSC, and PROSER.")
    reference_configuration = configurations["vanilla"]
    paths = get_output_paths(reference_configuration)
    hypotheses_file, hypotheses = _load_completed_hypotheses(
        paths["metrics_directory"] / "task4_hypotheses.json"
    )
    thresholds_file = paths["metrics_directory"] / "task4_thresholds.json"
    locked_files = {
        "hypotheses": _register_locked_file(hypotheses_file),
        "thresholds": _register_locked_file(thresholds_file),
        "unknown_protocol": _register_locked_file(paths["unknown_protocol_file"]),
        "mahalanobis_statistics": _register_locked_file(
            paths["mahalanobis_cache_directory"] / "vanilla_statistics.npz"
        ),
    }
    score_directory = Path(__file__).resolve().parents[1] / "scores"
    for score_file in sorted(score_directory.glob("*.py")):
        locked_files[f"score_code/{score_file.name}"] = _register_locked_file(
            score_file
        )
    task_directory = Path(__file__).resolve().parents[1]
    decision_code_files = (
        task_directory / "evaluation" / "evaluate_osr.py",
        task_directory / "evaluation" / "extract_outputs.py",
        task_directory / "evaluation" / "metrics.py",
        task_directory / "evaluation" / "thresholds.py",
        task_directory / "models" / "resnet_cifar.py",
        task_directory / "data" / "cifar10.py",
        task_directory / "data" / "cifar100_unknowns.py",
    )
    for code_file in decision_code_files:
        locked_files[f"decision_code/{code_file.name}"] = _register_locked_file(
            code_file
        )

    locked_runs = {}
    for method_name, configuration in configurations.items():
        checkpoint_file = get_checkpoint_path(configuration)
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
        locked_runs[method_name] = {
            "checkpoint": _register_locked_file(checkpoint_file),
            "configuration_sha256": hashlib.sha256(configuration_text).hexdigest(),
            "configuration": configuration,
        }
        del checkpoint
        for split_name in KNOWN_CACHE_SPLITS[method_name]:
            locked_files[f"known_outputs/{method_name}/{split_name}"] = (
                _register_locked_file(
                    get_known_output_path(configuration, split_name)
                )
            )

    lock_data = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 6304,
        "cifar100_used_for_training_selection_scores_or_thresholds": False,
        "hypotheses": hypotheses,
        "runs": locked_runs,
        "files": locked_files,
    }
    save_json(lock_data, lock_file)
    print(f"Locked Task 4 decisions: {Path(lock_file).resolve()}")
    return lock_data


def validate_experiment_lock(configurations, lock_file):
    """Reject any decision artifact changed after the Task 4 lock."""
    lock_path = Path(lock_file)
    if not lock_path.exists():
        raise FileNotFoundError("Create the Task 4 lock before loading CIFAR-100.")
    with lock_path.open("r", encoding="utf-8") as file:
        lock_data = json.load(file)
    if (
        lock_data.get(
            "cifar100_used_for_training_selection_scores_or_thresholds"
        )
        is not False
    ):
        raise ValueError("The Task 4 lock does not certify known-only decisions.")
    if set(lock_data["runs"]) != set(configurations):
        raise ValueError("The current Task 4 runs differ from the locked runs.")
    for method_name, configuration in configurations.items():
        locked_run = lock_data["runs"][method_name]
        configuration_text = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(configuration_text).hexdigest() != locked_run[
            "configuration_sha256"
        ]:
            raise ValueError(f"Configuration changed after locking: {method_name}")
    for artifact_name, artifact in lock_data["files"].items():
        if calculate_file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Artifact changed after locking: {artifact_name}")
    for method_name, run in lock_data["runs"].items():
        checkpoint = run["checkpoint"]
        if calculate_file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint changed after locking: {method_name}")
    return lock_data


def concatenate_outputs(first_outputs, second_outputs):
    """Concatenate matching near and far output dictionaries by example axis."""
    if set(first_outputs) != set(second_outputs):
        raise ValueError("Output dictionaries must contain the same arrays.")
    return {
        key: np.concatenate([first_outputs[key], second_outputs[key]], axis=0)
        for key in first_outputs
    }


def plot_vanilla_score_distributions(score_data, output_file):
    """Plot known and all-unknown distributions for MSP, MLS, and Mahalanobis."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, score_name in zip(axes, ("msp", "mls", "mahalanobis")):
        axis.hist(
            score_data[score_name]["known"],
            bins=50,
            density=True,
            alpha=0.65,
            label="CIFAR-10 known",
        )
        axis.hist(
            score_data[score_name]["unknown"],
            bins=50,
            density=True,
            alpha=0.65,
            label="CIFAR-100 unknown",
        )
        axis.axvline(
            score_data[score_name]["threshold"],
            color="black",
            linestyle="--",
            label="Validation threshold",
        )
        axis.set(
            title=score_name.upper(),
            xlabel="Unknownness (larger means more novel)",
            ylabel="Density",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_final_open_set_evaluation(configurations, lock_file, device=None):
    """Validate the lock, then load CIFAR-100 once for final analysis."""
    validate_experiment_lock(configurations, lock_file)
    selected_device = select_training_device(device)
    reference_configuration = configurations["vanilla"]
    paths = create_output_directories(reference_configuration)
    with (paths["metrics_directory"] / "task4_thresholds.json").open(
        "r",
        encoding="utf-8",
    ) as file:
        threshold_registry = json.load(file)
    mahalanobis_statistics = load_mahalanobis_statistics(
        paths["mahalanobis_cache_directory"] / "vanilla_statistics.npz"
    )

    from task4.data.cifar100_unknowns import (
        prepare_cifar100_unknown_datasets,
    )

    unknown_datasets, unknown_protocol = prepare_cifar100_unknown_datasets(
        reference_configuration,
        paths["cifar100_directory"],
        paths["unknown_protocol_file"],
    )
    method_results = {}
    result_rows = []
    for method_name, configuration in configurations.items():
        model, checkpoint, checkpoint_file = load_model_for_evaluation(
            configuration,
            selected_device,
        )
        unknown_outputs = {}
        for group_offset, group_name in enumerate(("near", "far")):
            loader = create_evaluation_loader(
                unknown_datasets[group_name],
                configuration,
                generator_offset=300 + group_offset,
            )
            outputs = collect_model_outputs(model, loader, selected_device)
            unknown_outputs[group_name] = outputs
            save_output_cache(
                outputs,
                paths["logits_cache_directory"]
                / f"{method_name}_{group_name}_final_outputs.npz",
            )
            save_feature_cache(
                outputs,
                paths["feature_cache_directory"]
                / f"{method_name}_{group_name}_final_features.npz",
            )

        known_test_outputs = load_output_cache(
            get_known_output_path(configuration, "test")
        )
        closed_set_metrics = calculate_closed_set_metrics(
            known_test_outputs["labels"],
            known_test_outputs["known_logits"],
        )
        method_results[method_name] = {
            "unknown_outputs": unknown_outputs,
            "closed_set_metrics": closed_set_metrics,
            "scores": {},
        }
        for score_name in SCORE_PLAN[method_name]:
            result_name = f"{method_name}_{score_name}"
            threshold_record = threshold_registry[result_name]
            calibration_bias = threshold_record.get("dummy_calibration_bias")
            known_scores = calculate_unknownness(
                score_name,
                known_test_outputs,
                configuration,
                mahalanobis_statistics=mahalanobis_statistics,
                proser_calibration_bias=calibration_bias,
            )
            near_scores = calculate_unknownness(
                score_name,
                unknown_outputs["near"],
                configuration,
                mahalanobis_statistics=mahalanobis_statistics,
                proser_calibration_bias=calibration_bias,
            )
            far_scores = calculate_unknownness(
                score_name,
                unknown_outputs["far"],
                configuration,
                mahalanobis_statistics=mahalanobis_statistics,
                proser_calibration_bias=calibration_bias,
            )
            osr_metrics = calculate_osr_metrics(
                known_scores,
                near_scores,
                far_scores,
                threshold_record["threshold"],
            )
            row = {
                "result_name": result_name,
                "model": method_name,
                "score": score_name,
                "cifar10_test_accuracy": closed_set_metrics["accuracy"],
                "cifar10_test_macro_f1": closed_set_metrics["macro_f1"],
                **osr_metrics,
            }
            result_rows.append(row)
            method_results[method_name]["scores"][score_name] = {
                "known": known_scores,
                "near": near_scores,
                "far": far_scores,
                "metrics": osr_metrics,
            }

        for group_name, outputs in unknown_outputs.items():
            prediction_table = create_prediction_table(outputs)
            for score_name, score_result in method_results[method_name]["scores"].items():
                prediction_table[f"{score_name}_unknownness"] = score_result[
                    group_name
                ]
            prediction_table.to_csv(
                paths["predictions_directory"]
                / f"{method_name}_{group_name}_unknown_predictions.csv",
                index=False,
            )
        model.to("cpu")
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Evaluated locked model: {checkpoint_file.resolve()}")

    complete_table = pd.DataFrame(result_rows)
    vanilla_score_table = complete_table[
        complete_table["model"] == "vanilla"
    ].reset_index(drop=True)
    model_comparison_names = {
        "vanilla_mls",
        "gcsc_mls",
        "proser_mls",
        "proser_placeholder",
    }
    model_comparison_table = complete_table[
        complete_table["result_name"].isin(model_comparison_names)
    ].reset_index(drop=True)
    complete_table.to_csv(
        paths["metrics_directory"] / "task4_complete_results.csv",
        index=False,
    )
    vanilla_score_table.to_csv(
        paths["metrics_directory"] / "vanilla_score_comparison.csv",
        index=False,
    )
    model_comparison_table.to_csv(
        paths["metrics_directory"] / "trained_model_comparison.csv",
        index=False,
    )

    vanilla_unknown_outputs = concatenate_outputs(
        method_results["vanilla"]["unknown_outputs"]["near"],
        method_results["vanilla"]["unknown_outputs"]["far"],
    )
    vanilla_mls = method_results["vanilla"]["scores"]["mls"]
    all_unknown_mls = np.concatenate([vanilla_mls["near"], vanilla_mls["far"]])
    mls_threshold = threshold_registry["vanilla_mls"]["threshold"]
    failure_table = create_accepted_unknown_failure_table(
        vanilla_unknown_outputs,
        all_unknown_mls,
        mls_threshold,
        reference_configuration["dataset"]["known_class_names"],
        minimum_cases=reference_configuration["evaluation"][
            "minimum_failure_cases_per_group"
        ],
    )
    failure_table.to_csv(
        paths["predictions_directory"] / "vanilla_mls_accepted_failures.csv",
        index=False,
    )
    plot_accepted_unknown_failures(
        unknown_datasets["near"].base_dataset,
        failure_table,
        paths["figures_directory"] / "vanilla_mls_accepted_failures.png",
    )

    vanilla_score_plot_data = {}
    for score_name in ("msp", "mls", "mahalanobis"):
        score_result = method_results["vanilla"]["scores"][score_name]
        vanilla_score_plot_data[score_name] = {
            "known": score_result["known"],
            "unknown": np.concatenate(
                [score_result["near"], score_result["far"]]
            ),
            "threshold": threshold_registry[f"vanilla_{score_name}"]["threshold"],
        }
    plot_vanilla_score_distributions(
        vanilla_score_plot_data,
        paths["figures_directory"] / "vanilla_score_distributions.png",
    )
    save_json(
        {
            "unknown_protocol": unknown_protocol,
            "number_of_near_unknowns": len(unknown_datasets["near"]),
            "number_of_far_unknowns": len(unknown_datasets["far"]),
            "lock_file": str(Path(lock_file).resolve()),
        },
        paths["metrics_directory"] / "task4_final_evaluation_metadata.json",
    )
    return {
        "complete_table": complete_table,
        "vanilla_score_table": vanilla_score_table,
        "model_comparison_table": model_comparison_table,
        "failure_table": failure_table,
        "method_results": method_results,
    }
