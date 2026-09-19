# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load, merge, validate, and save Task 2 YAML configurations."""

import copy
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = Path(__file__).resolve().parent
AVAILABLE_METHODS = ("source_only", "dan", "dann", "cdan")


def load_yaml_file(config_file):
    """Load one YAML mapping from the Task 2 configuration directory."""
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = CONFIG_DIRECTORY / config_path
    with config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return configuration


def deep_merge(base_configuration, override_configuration):
    """Recursively merge method settings into a copy of the base settings."""
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
    """Resolve one configuration path relative to the project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_output_paths(configuration):
    """Return the configured Task 2 result and cache paths as Path objects."""
    configured_paths = configuration["paths"]
    results_directory = resolve_project_path(
        configured_paths["results_directory"]
    )
    return {
        "dataset_directory": resolve_project_path(
            configured_paths["dataset_directory"]
        ),
        "download_directory": resolve_project_path(
            configured_paths["download_directory"]
        ),
        "feature_cache_directory": resolve_project_path(
            configured_paths["feature_cache_directory"]
        ),
        "model_cache_directory": resolve_project_path(
            configured_paths["model_cache_directory"]
        ),
        "split_file": resolve_project_path(configured_paths["split_file"]),
        "results_directory": results_directory,
        "checkpoints_directory": results_directory / "checkpoints",
        "histories_directory": results_directory / "histories",
        "metrics_directory": results_directory / "metrics",
        "predictions_directory": results_directory / "predictions",
        "figures_directory": results_directory / "figures",
    }


def validate_task2_config(configuration):
    """Check settings that are fixed by the Task 2 assignment protocol."""
    dataset = configuration["dataset"]
    model = configuration["model"]
    training = configuration["training"]
    transforms = configuration["transforms"]
    method = configuration["method"]

    if configuration["seed"] != 6304:
        raise ValueError("Task 2 comparisons must use seed 6304.")
    if dataset["source_domains"] != ["photo", "art_painting", "cartoon"]:
        raise ValueError("Task 2 source domains must be Photo, Art Painting, Cartoon.")
    if dataset["target_domain"] != "sketch":
        raise ValueError("Task 2 target domain must be Sketch.")
    if (
        dataset["source_train_ratio"] != 0.8
        or dataset["source_validation_ratio"] != 0.2
    ):
        raise ValueError("Each PACS source domain requires an 80/20 split.")
    if len(dataset["class_names"]) != 7 or model["number_of_classes"] != 7:
        raise ValueError("PACS requires exactly seven output classes.")
    if model["backbone"] != "resnet18":
        raise ValueError("Task 2 requires a ResNet-18 backbone.")
    if model["pretrained_weights"] != "IMAGENET1K_V1":
        raise ValueError("Task 2 requires ResNet18 IMAGENET1K_V1 weights.")
    if model["feature_dimension"] != 512:
        raise ValueError("The ResNet-18 pre-classifier feature dimension must be 512.")
    if model["freeze_batch_norm_statistics"] is not True:
        raise ValueError("ImageNet BatchNorm running statistics must remain frozen.")
    if (
        transforms["resize_size"] != 256
        or transforms["crop_size"] != 224
        or transforms["random_horizontal_flip"] is not True
    ):
        raise ValueError("Task 2 requires resize 256, crop 224, and training flips.")
    if training["source_batch_size_per_domain"] != 8:
        raise ValueError("Each source domain must contribute eight training images.")
    if training["target_batch_size"] != 24:
        raise ValueError("Each adaptation update must contain 24 target images.")
    if training["maximum_epochs"] > 30:
        raise ValueError("Task 2 permits at most 30 source epochs.")
    if training["early_stopping_patience"] != 5:
        raise ValueError("Task 2 early stopping patience must equal five epochs.")
    if (
        training["learning_rate"] != 0.0001
        or training["weight_decay"] != 0.0001
    ):
        raise ValueError("Task 2 requires AdamW learning rate and weight decay 1e-4.")
    if training["mixed_precision"] is not False:
        raise ValueError("Task 2 uses full-precision training for numerical stability.")
    if training["gradient_clip_norm"] != 5.0:
        raise ValueError("Task 2 requires a shared gradient-clipping norm of five.")
    if training["checkpoint_metric"] != "mean_source_validation_macro_f1":
        raise ValueError("Checkpoints must use mean source-validation macro-F1.")
    if method["name"] not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown Task 2 method: {method['name']}")

    if method["name"] == "dan":
        if method["kernel_bandwidth_multipliers"] != [0.5, 1.0, 2.0]:
            raise ValueError("DAN requires kernel multipliers 0.5, 1, and 2.")
    if method["name"] in {"dann", "cdan"}:
        if (
            method["domain_loss_weight"] != 1.0
            or method["discriminator_hidden_dimension"] != 256
            or method["discriminator_dropout"] != 0.5
            or method["grl_maximum_strength"] != 1.0
        ):
            raise ValueError("DANN and CDAN discriminator settings are fixed.")


def load_config(method_name, overrides=None):
    """Load the common settings and one method-specific YAML override."""
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
    validate_task2_config(configuration)
    return configuration


def create_output_directories(configuration):
    """Create every configured Task 2 data-cache and result directory."""
    output_paths = get_output_paths(configuration)
    for key, path in output_paths.items():
        if key == "split_file":
            path.parent.mkdir(parents=True, exist_ok=True)
        elif key.endswith("directory"):
            path.mkdir(parents=True, exist_ok=True)
    return output_paths


def save_config_snapshot(configuration, output_file):
    """Save the merged configuration beside an experiment's outputs."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)
