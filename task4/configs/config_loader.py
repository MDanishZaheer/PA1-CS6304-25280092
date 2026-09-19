# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load, merge, validate, and save Task 4 YAML configurations."""

import copy
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = Path(__file__).resolve().parent
AVAILABLE_METHODS = ("vanilla", "gcsc", "proser")
KNOWN_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
NEAR_UNKNOWN_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel",
]
FAR_UNKNOWN_CLASSES = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe",
]


def load_yaml_file(config_file):
    """Load one YAML mapping from the Task 4 configuration directory."""
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = CONFIG_DIRECTORY / config_path
    with config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return configuration


def deep_merge(base_configuration, override_configuration):
    """Recursively merge method settings into common Task 4 settings."""
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
    """Resolve one configured path relative to the PA_1 project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_output_paths(configuration):
    """Return every Task 4 data, cache, and result path as a Path."""
    configured_paths = configuration["paths"]
    results_directory = resolve_project_path(
        configured_paths["results_directory"]
    )
    return {
        "cifar10_directory": resolve_project_path(
            configured_paths["cifar10_directory"]
        ),
        "cifar100_directory": resolve_project_path(
            configured_paths["cifar100_directory"]
        ),
        "feature_cache_directory": resolve_project_path(
            configured_paths["feature_cache_directory"]
        ),
        "logits_cache_directory": resolve_project_path(
            configured_paths["logits_cache_directory"]
        ),
        "mahalanobis_cache_directory": resolve_project_path(
            configured_paths["mahalanobis_cache_directory"]
        ),
        "cifar10_split_file": resolve_project_path(
            configured_paths["cifar10_split_file"]
        ),
        "unknown_protocol_file": resolve_project_path(
            configured_paths["unknown_protocol_file"]
        ),
        "results_directory": results_directory,
        "checkpoints_directory": results_directory / "checkpoints",
        "histories_directory": results_directory / "histories",
        "metrics_directory": results_directory / "metrics",
        "predictions_directory": results_directory / "predictions",
        "figures_directory": results_directory / "figures",
    }


def get_checkpoint_path(configuration):
    """Return the selected checkpoint path for one Task 4 method."""
    run_name = configuration["method"]["run_name"]
    return get_output_paths(configuration)["checkpoints_directory"] / (
        f"{run_name}_best.pt"
    )


def get_known_output_path(configuration, split_name):
    """Return a cache path for known logits and features from one split."""
    run_name = configuration["method"]["run_name"]
    cache_directory = get_output_paths(configuration)["logits_cache_directory"]
    return cache_directory / f"{run_name}_{split_name}_outputs.npz"


def get_known_feature_path(configuration, split_name):
    """Return a separate feature-only cache path for convenient analysis."""
    run_name = configuration["method"]["run_name"]
    cache_directory = get_output_paths(configuration)["feature_cache_directory"]
    return cache_directory / f"{run_name}_{split_name}_features.npz"


def validate_task4_config(configuration):
    """Enforce every fixed setting in the required Task 4 protocol."""
    dataset = configuration["dataset"]
    transforms = configuration["transforms"]
    model = configuration["model"]
    training = configuration["training"]
    evaluation = configuration["evaluation"]
    method = configuration["method"]

    if configuration["seed"] != 6304:
        raise ValueError("Task 4 comparisons must use seed 6304.")
    if dataset["known_name"] != "CIFAR10" or dataset["unknown_name"] != "CIFAR100":
        raise ValueError("Task 4 requires CIFAR-10 knowns and CIFAR-100 unknowns.")
    if dataset["known_class_names"] != KNOWN_CLASSES:
        raise ValueError("The configured CIFAR-10 class order is incorrect.")
    if dataset["near_unknown_classes"] != NEAR_UNKNOWN_CLASSES:
        raise ValueError("The near-unknown CIFAR-100 group is fixed by the assignment.")
    if dataset["far_unknown_classes"] != FAR_UNKNOWN_CLASSES:
        raise ValueError("The far-unknown CIFAR-100 group is fixed by the assignment.")
    if dataset["train_ratio"] != 0.9 or dataset["validation_ratio"] != 0.1:
        raise ValueError("CIFAR-10 requires the fixed stratified 90/10 split.")
    if (
        transforms["image_size"] != 32
        or transforms["crop_padding"] != 4
        or transforms["random_horizontal_flip"] is not True
        or transforms["normalization"] != "cifar10"
    ):
        raise ValueError("Task 4 requires the specified CIFAR augmentation recipe.")
    if (
        model["backbone"] != "resnet18"
        or model["pretrained"] is not False
        or model["first_convolution_kernel_size"] != 3
        or model["first_convolution_stride"] != 1
        or model["use_initial_max_pool"] is not False
        or model["feature_dimension"] != 512
        or model["number_of_known_classes"] != 10
    ):
        raise ValueError("Task 4 requires the CIFAR-appropriate ResNet-18.")
    if (
        training["optimizer"] != "sgd"
        or training["momentum"] != 0.9
        or training["weight_decay"] != 0.0005
        or training["batch_size"] != 128
        or training["scheduler"] != "cosine"
        or training["checkpoint_metric"] != "validation_accuracy"
    ):
        raise ValueError("Task 4 requires the fixed SGD training protocol.")
    if (
        evaluation["threshold_percentile"] != 95.0
        or evaluation["mahalanobis_epsilon"] != 0.000001
        or evaluation["energy_temperature"] != 1.0
        or evaluation["proser_temperature"] != 1024.0
    ):
        raise ValueError("Task 4 evaluation and calibration settings are fixed.")
    if method["name"] not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown Task 4 method: {method['name']}")

    if method["name"] in {"vanilla", "gcsc"}:
        if training["learning_rate"] != 0.1 or training["maximum_epochs"] != 100:
            raise ValueError("Vanilla and GCSC require learning rate 0.1 for 100 epochs.")
        if method["initialize_from_vanilla"] is not False:
            raise ValueError("Vanilla and GCSC must start from random initialization.")
        if method["number_of_dummy_classes"] != 0:
            raise ValueError("Vanilla and GCSC must have ten output classes.")
        expected_randaugment = method["name"] == "gcsc"
        if method["use_randaugment"] is not expected_randaugment:
            raise ValueError("Only GCSC may use RandAugment.")
        if method["name"] == "gcsc" and (
            method["randaugment_number_of_operations"] != 2
            or method["randaugment_magnitude"] != 9
        ):
            raise ValueError("GCSC requires RandAugment(num_ops=2, magnitude=9).")
    else:
        if training["learning_rate"] != 0.001 or training["maximum_epochs"] != 50:
            raise ValueError("PROSER requires learning rate 1e-3 for 50 epochs.")
        if method["initialize_from_vanilla"] is not True:
            raise ValueError("PROSER must initialize from the selected Vanilla model.")
        if (
            method["number_of_dummy_classes"] != 5
            or method["classifier_placeholder_weight"] != 1.0
            or method["data_placeholder_weight"] != 0.1
            or method["beta_distribution_alpha"] != 2.0
            or method["manifold_mixup_location"] != "after_layer2"
        ):
            raise ValueError("The required PROSER placeholder settings are fixed.")


def load_config(method_name, overrides=None):
    """Load common settings plus one method-specific Task 4 YAML file."""
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
    validate_task4_config(configuration)
    return configuration


def create_output_directories(configuration):
    """Create Task 4 cache and result folders without placeholder files."""
    paths = get_output_paths(configuration)
    directory_keys = {
        "feature_cache_directory",
        "logits_cache_directory",
        "mahalanobis_cache_directory",
        "results_directory",
        "checkpoints_directory",
        "histories_directory",
        "metrics_directory",
        "predictions_directory",
        "figures_directory",
    }
    for key in directory_keys:
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def save_config_snapshot(configuration, output_file):
    """Save one fully merged configuration as formatted JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)
