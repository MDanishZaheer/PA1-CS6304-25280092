# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create and validate the fixed stratified CIFAR-10 training split."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SEED = 6304
TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.10


def calculate_class_counts(labels, indices, class_names):
    """Count examples from every class for a selected list of indices."""
    counts = Counter(int(labels[index]) for index in indices)
    return {
        class_name: int(counts[class_index])
        for class_index, class_name in enumerate(class_names)
    }


def build_cifar10_split(labels, class_names, seed=SEED):
    """Build the assignment's reproducible stratified 90/10 split."""
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or len(label_array) == 0:
        raise ValueError("CIFAR-10 labels must be a non-empty one-dimensional array.")
    if set(np.unique(label_array)) != set(range(len(class_names))):
        raise ValueError("CIFAR-10 labels do not match the expected class range.")

    all_indices = np.arange(len(label_array))
    training_indices, validation_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_RATIO,
        random_state=int(seed),
        stratify=label_array,
    )
    training_indices = sorted(int(index) for index in training_indices)
    validation_indices = sorted(int(index) for index in validation_indices)
    return {
        "dataset": "CIFAR10",
        "partition": "train",
        "seed": int(seed),
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "class_names": list(class_names),
        "all_count": int(len(label_array)),
        "train_indices": training_indices,
        "validation_indices": validation_indices,
        "train_class_counts": calculate_class_counts(
            label_array,
            training_indices,
            class_names,
        ),
        "validation_class_counts": calculate_class_counts(
            label_array,
            validation_indices,
            class_names,
        ),
    }


def save_cifar10_split(protocol, split_file):
    """Save the CIFAR-10 split as machine-readable JSON."""
    split_path = Path(split_file)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as file:
        json.dump(protocol, file, indent=2)


def load_cifar10_split(split_file):
    """Load a previously generated CIFAR-10 split description."""
    with Path(split_file).open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_cifar10_split(protocol, labels, class_names):
    """Verify coverage, stratification, counts, and fixed split settings."""
    label_array = np.asarray(labels, dtype=np.int64)
    if protocol.get("dataset") != "CIFAR10" or protocol.get("partition") != "train":
        raise ValueError("The saved split does not describe CIFAR-10 training data.")
    if protocol.get("seed") != SEED:
        raise ValueError("The saved CIFAR-10 split uses a different seed.")
    if protocol.get("class_names") != list(class_names):
        raise ValueError("The saved CIFAR-10 class order is incorrect.")
    if (
        not np.isclose(protocol.get("train_ratio", -1), TRAIN_RATIO)
        or not np.isclose(
            protocol.get("validation_ratio", -1),
            VALIDATION_RATIO,
        )
    ):
        raise ValueError("The saved CIFAR-10 split is not 90/10.")
    if protocol.get("all_count") != len(label_array):
        raise ValueError("The saved CIFAR-10 image count is incorrect.")

    training_list = protocol["train_indices"]
    validation_list = protocol["validation_indices"]
    training_indices = set(int(index) for index in training_list)
    validation_indices = set(int(index) for index in validation_list)
    expected_indices = set(range(len(label_array)))
    if len(training_indices) != len(training_list):
        raise ValueError("The CIFAR-10 training indices contain duplicates.")
    if len(validation_indices) != len(validation_list):
        raise ValueError("The CIFAR-10 validation indices contain duplicates.")
    if training_indices & validation_indices:
        raise ValueError("The CIFAR-10 training and validation splits overlap.")
    if training_indices | validation_indices != expected_indices:
        raise ValueError("The CIFAR-10 split does not cover the training partition.")

    training_counts = calculate_class_counts(
        label_array,
        training_list,
        class_names,
    )
    validation_counts = calculate_class_counts(
        label_array,
        validation_list,
        class_names,
    )
    if training_counts != protocol["train_class_counts"]:
        raise ValueError("The saved CIFAR-10 training class counts are incorrect.")
    if validation_counts != protocol["validation_class_counts"]:
        raise ValueError("The saved CIFAR-10 validation class counts are incorrect.")
    for class_index, class_name in enumerate(class_names):
        class_total = int((label_array == class_index).sum())
        expected_validation = class_total * VALIDATION_RATIO
        if abs(validation_counts[class_name] - expected_validation) > 1:
            raise ValueError("The saved CIFAR-10 split is not stratified.")


def prepare_cifar10_split(labels, class_names, split_file):
    """Create the fixed split once or validate and reuse its saved indices."""
    split_path = Path(split_file)
    if split_path.exists():
        protocol = load_cifar10_split(split_path)
        print("Using the existing CIFAR-10 split.")
    else:
        protocol = build_cifar10_split(labels, class_names, seed=SEED)
        save_cifar10_split(protocol, split_path)
        print(f"Created CIFAR-10 split: {split_path.resolve()}")
    validate_cifar10_split(protocol, labels, class_names)
    return protocol
