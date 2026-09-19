# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create and save deterministic STL-10 splits and evaluation identifiers."""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10

from task1.configs.task1_config import (
    DATASET_NAME,
    SEED,
    SPLIT_FILE,
    STL10_CLASSES,
    STL10_DIR,
    TEST_IMAGES_PER_CLASS,
    TEST_SUBSET_SIZE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


def load_stl10_datasets(download=True):
    """Load the official labeled train and test partitions without transforms."""
    train_dataset = STL10(root=str(STL10_DIR), split="train", download=download)
    test_dataset = STL10(root=str(STL10_DIR), split="test", download=download)
    return train_dataset, test_dataset


def get_labels(dataset):
    """Return dataset labels as a one-dimensional integer array."""
    labels = np.asarray(dataset.labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != len(dataset):
        raise ValueError("STL-10 labels do not match the dataset size.")
    return labels


def count_classes(labels):
    """Count examples for every STL-10 class."""
    return {
        class_name: int(np.sum(labels == class_index))
        for class_index, class_name in enumerate(STL10_CLASSES)
    }


def create_train_validation_indices(train_labels):
    """Create the required stratified 80/20 split of the training partition."""
    if not np.isclose(TRAIN_RATIO + VALIDATION_RATIO, 1.0):
        raise ValueError("Training and validation ratios must add to one.")

    all_indices = np.arange(len(train_labels))
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_RATIO,
        random_state=SEED,
        stratify=train_labels,
    )
    train_indices = sorted(int(index) for index in train_indices)
    validation_indices = sorted(int(index) for index in validation_indices)
    return train_indices, validation_indices


def create_balanced_test_indices(test_labels):
    """Select up to 50 examples per class from the official test partition."""
    random_generator = np.random.default_rng(SEED)
    selected_indices = []

    for class_index in range(len(STL10_CLASSES)):
        class_indices = np.flatnonzero(test_labels == class_index)
        selected_count = min(TEST_IMAGES_PER_CLASS, len(class_indices))
        class_selection = random_generator.choice(
            class_indices,
            size=selected_count,
            replace=False,
        )
        selected_indices.extend(int(index) for index in class_selection)

    return sorted(selected_indices)


def create_test_identifiers(test_indices, test_labels):
    """Create stable identifiers tied to official STL-10 test indices."""
    identifiers = []
    for index in test_indices:
        class_index = int(test_labels[index])
        identifiers.append(
            {
                "identifier": f"test_{index:05d}",
                "official_index": index,
                "label_index": class_index,
                "class_name": STL10_CLASSES[class_index],
            }
        )
    return identifiers


def build_split_data(train_dataset, test_dataset):
    """Build the complete machine-readable split description."""
    train_labels = get_labels(train_dataset)
    test_labels = get_labels(test_dataset)
    train_indices, validation_indices = create_train_validation_indices(train_labels)
    test_indices = create_balanced_test_indices(test_labels)

    train_class_counts = count_classes(train_labels[np.asarray(train_indices)])
    validation_class_counts = count_classes(train_labels[np.asarray(validation_indices)])
    test_class_counts = count_classes(test_labels[np.asarray(test_indices)])
    insufficient_classes = [
        class_name
        for class_name, count in test_class_counts.items()
        if count < TEST_IMAGES_PER_CLASS
    ]

    return {
        "dataset_name": DATASET_NAME,
        "seed": SEED,
        "class_names": list(STL10_CLASSES),
        "split_protocol": {
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "requested_test_size": TEST_SUBSET_SIZE,
            "requested_test_images_per_class": TEST_IMAGES_PER_CLASS,
        },
        "official_partition_sizes": {
            "train": len(train_dataset),
            "test": len(test_dataset),
        },
        "indices": {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        },
        "class_counts": {
            "official_train": count_classes(train_labels),
            "official_test": count_classes(test_labels),
            "train": train_class_counts,
            "validation": validation_class_counts,
            "test": test_class_counts,
        },
        "test_subset": {
            "selected_size": len(test_indices),
            "is_balanced": len(set(test_class_counts.values())) == 1,
            "insufficient_classes": insufficient_classes,
        },
        "test_image_identifiers": create_test_identifiers(test_indices, test_labels),
    }


def validate_index_list(indices, partition_size, split_name):
    """Check that saved indices are unique and inside the official partition."""
    if len(indices) != len(set(indices)):
        raise ValueError(f"The {split_name} indices contain duplicates.")
    if any(index < 0 or index >= partition_size for index in indices):
        raise ValueError(f"The {split_name} indices contain an invalid index.")


def validate_split_data(split_data, train_dataset, test_dataset):
    """Verify that saved splits still satisfy the Task 1 protocol."""
    if split_data.get("dataset_name") != DATASET_NAME:
        raise ValueError("The saved split belongs to a different dataset.")
    if split_data.get("seed") != SEED:
        raise ValueError("The saved split uses a different random seed.")
    if split_data.get("class_names") != list(STL10_CLASSES):
        raise ValueError("The saved STL-10 class order does not match the configuration.")

    train_labels = get_labels(train_dataset)
    test_labels = get_labels(test_dataset)
    train_indices = split_data["indices"]["train"]
    validation_indices = split_data["indices"]["validation"]
    test_indices = split_data["indices"]["test"]
    validate_index_list(train_indices, len(train_dataset), "train")
    validate_index_list(validation_indices, len(train_dataset), "validation")
    validate_index_list(test_indices, len(test_dataset), "test")

    train_index_set = set(train_indices)
    validation_index_set = set(validation_indices)
    if train_index_set & validation_index_set:
        raise ValueError("Training and validation indices overlap.")
    if train_index_set | validation_index_set != set(range(len(train_dataset))):
        raise ValueError("Training and validation indices do not cover the train partition.")

    for class_index, class_name in enumerate(STL10_CLASSES):
        official_train_count = int(np.sum(train_labels == class_index))
        validation_count = int(np.sum(train_labels[validation_indices] == class_index))
        expected_validation_count = official_train_count * VALIDATION_RATIO
        if abs(validation_count - expected_validation_count) > 1:
            raise ValueError(f"The {class_name} train-validation split is not stratified.")

        official_test_count = int(np.sum(test_labels == class_index))
        selected_test_count = int(np.sum(test_labels[test_indices] == class_index))
        expected_test_count = min(TEST_IMAGES_PER_CLASS, official_test_count)
        if selected_test_count != expected_test_count:
            raise ValueError(f"The {class_name} test subset count is incorrect.")

    identifiers = split_data.get("test_image_identifiers", [])
    if len(identifiers) != len(test_indices):
        raise ValueError("The number of test identifiers is incorrect.")
    for index, identifier in zip(test_indices, identifiers):
        class_index = int(test_labels[index])
        if identifier.get("official_index") != index:
            raise ValueError("A test identifier has the wrong official index.")
        if identifier.get("label_index") != class_index:
            raise ValueError("A test identifier has the wrong label.")


def save_split_data(split_data, split_file=SPLIT_FILE):
    """Save split indices and metadata as formatted JSON."""
    split_path = Path(split_file)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as file:
        json.dump(split_data, file, indent=2)


def load_split_data(split_file=SPLIT_FILE):
    """Load previously saved split indices and metadata."""
    split_path = Path(split_file)
    with split_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def print_split_summary(split_data, split_file=SPLIT_FILE):
    """Print the sizes and class balance needed to verify the split."""
    indices = split_data["indices"]
    print(f"Dataset: {split_data['dataset_name']}")
    print(f"Seed: {split_data['seed']}")
    print(f"Training examples: {len(indices['train'])}")
    print(f"Validation examples: {len(indices['validation'])}")
    print(f"Selected test examples: {len(indices['test'])}")
    print(f"Test class counts: {split_data['class_counts']['test']}")
    print(f"Saved split file: {Path(split_file).resolve()}")

    insufficient_classes = split_data["test_subset"]["insufficient_classes"]
    if insufficient_classes:
        print(f"Classes with fewer than 50 test examples: {insufficient_classes}")


def prepare_stl10_splits(download=True, overwrite=False, split_file=SPLIT_FILE):
    """Load STL-10 and create or reuse its validated Task 1 splits."""
    train_dataset, test_dataset = load_stl10_datasets(download=download)
    split_path = Path(split_file)

    if split_path.exists() and not overwrite:
        split_data = load_split_data(split_path)
        print("Using the existing saved STL-10 split.")
    else:
        split_data = build_split_data(train_dataset, test_dataset)
        save_split_data(split_data, split_path)
        print("Created and saved a new STL-10 split.")

    validate_split_data(split_data, train_dataset, test_dataset)
    print_split_summary(split_data, split_path)
    return train_dataset, test_dataset, split_data


def main():
    """Prepare the assignment splits when this module is executed directly."""
    prepare_stl10_splits(download=True, overwrite=False)


if __name__ == "__main__":
    main()
