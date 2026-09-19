# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Construct the fixed CIFAR-100 near/far groups for final evaluation only."""

import json
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from task4.data.cifar10 import create_evaluation_transform


class CIFAR100UnknownDataset(Dataset):
    """Expose one fixed unknown group with names and original test indices."""

    def __init__(self, base_dataset, indices, group_name, transform):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.group_name = str(group_name)
        self.transform = transform
        self.class_names = list(base_dataset.classes)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dataset_index = self.indices[index]
        image, label = self.base_dataset[dataset_index]
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(label),
            "class_name": self.class_names[int(label)],
            "unknown_group": self.group_name,
            "identifier": f"cifar100_test_{dataset_index:05d}",
            "dataset_index": int(dataset_index),
        }


def load_unknown_protocol(protocol_file):
    """Load the assignment-provided CIFAR-100 near/far class mapping."""
    with Path(protocol_file).open("r", encoding="utf-8") as file:
        protocol = json.load(file)
    if protocol.get("dataset") != "CIFAR100" or protocol.get("partition") != "test":
        raise ValueError("The unknown protocol must describe CIFAR-100 test data.")
    return protocol


def validate_unknown_protocol(protocol, configuration, dataset):
    """Verify exact class names, indices, and expected group sizes."""
    configured_groups = {
        "near": configuration["dataset"]["near_unknown_classes"],
        "far": configuration["dataset"]["far_unknown_classes"],
    }
    for group_name, configured_names in configured_groups.items():
        class_mapping = protocol[group_name]["classes"]
        expected_count = int(protocol[group_name]["expected_count"])
        if list(class_mapping) != configured_names:
            raise ValueError(f"The fixed {group_name} unknown classes changed.")
        if expected_count != 800:
            raise ValueError(f"The fixed {group_name} group must specify 800 images.")
        for class_name, class_index in class_mapping.items():
            if dataset.classes[int(class_index)] != class_name:
                raise ValueError(
                    f"CIFAR-100 index {class_index} is not class {class_name}."
                )
        selected_labels = set(class_mapping.values())
        actual_count = sum(
            int(label) in selected_labels
            for label in dataset.targets
        )
        if actual_count != expected_count:
            raise ValueError(f"The {group_name} unknown group must contain 800 images.")


def prepare_cifar100_unknown_datasets(configuration, data_directory, protocol_file):
    """Load CIFAR-100 only when called by the locked final evaluator."""
    base_dataset = CIFAR100(
        root=data_directory,
        train=False,
        transform=None,
        download=bool(configuration["dataset"]["download"]),
    )
    protocol = load_unknown_protocol(protocol_file)
    validate_unknown_protocol(protocol, configuration, base_dataset)
    transform = create_evaluation_transform()
    datasets = {}
    for group_name in ("near", "far"):
        selected_labels = set(protocol[group_name]["classes"].values())
        selected_indices = [
            index
            for index, label in enumerate(base_dataset.targets)
            if int(label) in selected_labels
        ]
        datasets[group_name] = CIFAR100UnknownDataset(
            base_dataset,
            selected_indices,
            group_name=group_name,
            transform=transform,
        )
    return datasets, protocol
