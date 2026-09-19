# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load, merge, validate, and save Task 3 YAML configurations."""

import copy
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = Path(__file__).resolve().parent
AVAILABLE_METHODS = ("erm", "dan_dg", "sam")


def load_yaml_file(config_file):
    """Load one YAML mapping from the Task 3 configuration directory."""
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = CONFIG_DIRECTORY / config_path
    with config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return configuration


def deep_merge(base_configuration, override_configuration):
    """Recursively merge method settings into a copy of common settings."""
    merged = copy.deepcopy(base_configuration)
    for key, value in override_configuration.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_project_path(path_value):
    """Resolve one configured path relative to the project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_output_paths(configuration):
    """Return Task 3 data, dependency, and result paths as Path objects."""
    configured_paths = configuration["paths"]
    results_directory = resolve_project_path(
        configured_paths["results_directory"]
    )
    return {
        "dataset_directory": resolve_project_path(
            configured_paths["dataset_directory"]
        ),
        "feature_cache_directory": resolve_project_path(
            configured_paths["feature_cache_directory"]
        ),
        "model_cache_directory": resolve_project_path(
            configured_paths["model_cache_directory"]
        ),
        "split_file": resolve_project_path(configured_paths["split_file"]),
        "task2_erm_checkpoint": resolve_project_path(
            configured_paths["task2_erm_checkpoint"]
        ),
        "task2_erm_history": resolve_project_path(
            configured_paths["task2_erm_history"]
        ),
        "task2_final_summary": resolve_project_path(
            configured_paths["task2_final_summary"]
        ),
        "task2_metrics_directory": resolve_project_path(
            configured_paths["task2_metrics_directory"]
        ),
        "results_directory": results_directory,
        "checkpoints_directory": results_directory / "checkpoints",
        "histories_directory": results_directory / "histories",
        "metrics_directory": results_directory / "metrics",
        "predictions_directory": results_directory / "predictions",
        "figures_directory": results_directory / "figures",
    }


def get_method_checkpoint_path(configuration):
    """Return Task 2 ERM or the configured Task 3 checkpoint path."""
    paths = get_output_paths(configuration)
    if configuration["method"]["name"] == "erm":
        return paths["task2_erm_checkpoint"]
    run_name = configuration["method"]["run_name"]
    return paths["checkpoints_directory"] / f"{run_name}_best.pt"


def validate_task3_config(configuration):
    """Check settings fixed by the Task 3 experimental protocol."""
    dataset = configuration["dataset"]
    transforms = configuration["transforms"]
    model = configuration["model"]
    training = configuration["training"]
    diagnostics = configuration["diagnostics"]
    controlled_study = configuration["controlled_study"]
    method = configuration["method"]

    if configuration["seed"] != 6304:
        raise ValueError("Task 3 comparisons must use seed 6304.")
    if dataset["name"] != "PACS":
        raise ValueError("Task 3 requires the PACS dataset.")
    if dataset["source_domains"] != ["photo", "art_painting", "cartoon"]:
        raise ValueError("Task 3 source domains must be Photo, Art Painting, Cartoon.")
    if dataset["class_names"] != [
        "dog",
        "elephant",
        "giraffe",
        "guitar",
        "horse",
        "house",
        "person",
    ]:
        raise ValueError("Task 3 must preserve the official PACS class order.")
    if dataset["target_domain"] != "sketch":
        raise ValueError("Task 3 final target must be Sketch.")
    if (
        dataset["source_train_ratio"] != 0.8
        or dataset["source_validation_ratio"] != 0.2
    ):
        raise ValueError("Task 3 must reuse the Task 2 80/20 source splits.")
    if model["backbone"] != "resnet18":
        raise ValueError("Task 3 requires ResNet-18.")
    if model["pretrained_weights"] != "IMAGENET1K_V1":
        raise ValueError("Task 3 requires ResNet18 IMAGENET1K_V1 weights.")
    if model["feature_dimension"] != 512 or model["number_of_classes"] != 7:
        raise ValueError("Task 3 requires a 512-dimensional feature and seven classes.")
    if model["freeze_batch_norm_statistics"] is not True:
        raise ValueError("ImageNet BatchNorm running statistics must remain frozen.")
    if (
        transforms["resize_size"] != 256
        or transforms["crop_size"] != 224
        or transforms["random_horizontal_flip"] is not True
        or transforms["normalization"] != "imagenet1k_v1"
    ):
        raise ValueError(
            "Task 3 requires resize 256, crop 224, training flips, and V1 normalization."
        )
    if training["source_batch_size_per_domain"] != 8:
        raise ValueError("Each source domain must contribute eight images per update.")
    if training["maximum_epochs"] > 30:
        raise ValueError("Task 3 permits at most 30 source epochs.")
    if training["early_stopping_patience"] != 5:
        raise ValueError("Task 3 early stopping patience must equal five.")
    if (
        training["learning_rate"] != 0.0001
        or training["weight_decay"] != 0.0001
    ):
        raise ValueError("Task 3 requires AdamW learning rate and weight decay 1e-4.")
    if training["checkpoint_metric"] != "mean_source_validation_macro_f1":
        raise ValueError("Task 3 checkpoints require mean source macro-F1.")
    if method["name"] not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown Task 3 method: {method['name']}")
    if diagnostics["sharpness_examples_per_domain"] != 32:
        raise ValueError("The sharpness batch requires 32 examples per source domain.")
    if diagnostics["sharpness_radius"] != 0.05:
        raise ValueError("The common sharpness diagnostic radius must equal 0.05.")
    if (
        diagnostics["logistic_regression_c"] != 1.0
        or diagnostics["diagnostic_train_ratio"] != 0.7
    ):
        raise ValueError("Source-domain separability requires C=1 and a 70/30 split.")
    if controlled_study["sam_rho_values"] != [0.01, 0.05, 0.1]:
        raise ValueError("The controlled SAM study must use rho 0.01, 0.05, and 0.1.")

    if method["name"] == "erm":
        if method["trainable"] or not method["reuse_task2_source_only"]:
            raise ValueError("Task 3 ERM must reuse the unchanged Task 2 checkpoint.")
    elif method["name"] == "dan_dg":
        if method["trainable"] is not True:
            raise ValueError("DAN-DG must train a new source-only model.")
        if method["mmd_weight"] != 1.0:
            raise ValueError("The main DAN-DG experiment requires MMD weight one.")
        if method["kernel_bandwidth_multipliers"] != [0.5, 1.0, 2.0]:
            raise ValueError("DAN-DG requires kernel multipliers 0.5, 1, and 2.")
    elif method["name"] == "sam":
        if method["trainable"] is not True:
            raise ValueError("SAM must train a new source-only model.")
        if method["adaptive"] is not False:
            raise ValueError("Task 3 requires standard non-adaptive SAM.")
        if method["rho"] not in controlled_study["sam_rho_values"]:
            raise ValueError("SAM rho must belong to the registered controlled study.")
        if method["rho"] != 0.05 and method["run_name"] == "sam":
            raise ValueError("A controlled SAM radius requires a unique run name.")


def load_config(method_name, overrides=None):
    """Load common settings and one Task 3 method-specific override."""
    normalized_method = str(method_name).lower()
    if normalized_method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Method must be one of {AVAILABLE_METHODS}, received {method_name}."
        )
    method_configuration = load_yaml_file(f"{normalized_method}.yaml")
    base_file = method_configuration.pop("base_config", "base.yaml")
    configuration = deep_merge(load_yaml_file(base_file), method_configuration)
    if overrides is not None:
        configuration = deep_merge(configuration, overrides)
    validate_task3_config(configuration)
    return configuration


def create_output_directories(configuration):
    """Create every configured Task 3 cache and result directory."""
    output_paths = get_output_paths(configuration)
    directory_keys = {
        "feature_cache_directory",
        "results_directory",
        "checkpoints_directory",
        "histories_directory",
        "metrics_directory",
        "predictions_directory",
        "figures_directory",
    }
    for key in directory_keys:
        output_paths[key].mkdir(parents=True, exist_ok=True)
    return output_paths


def save_config_snapshot(configuration, output_file):
    """Save one merged Task 3 configuration as formatted JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)
