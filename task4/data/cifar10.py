# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Download and construct known-only CIFAR-10 datasets for Task 4."""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from task4.configs.config_loader import get_output_paths
from task4.data.make_splits import prepare_cifar10_split


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STANDARD_DEVIATION = (0.2470, 0.2435, 0.2616)


def create_training_transform(configuration):
    """Create Vanilla/PROSER augmentation or the GCSC RandAugment recipe."""
    transform_settings = configuration["transforms"]
    method = configuration["method"]
    operations = [
        v2.RandomCrop(
            size=int(transform_settings["image_size"]),
            padding=int(transform_settings["crop_padding"]),
        ),
        v2.RandomHorizontalFlip(),
    ]
    if method["use_randaugment"]:
        operations.append(
            v2.RandAugment(
                num_ops=int(method["randaugment_number_of_operations"]),
                magnitude=int(method["randaugment_magnitude"]),
            )
        )
    operations.extend(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]
    )
    operations.append(
        v2.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STANDARD_DEVIATION,
        )
    )
    return v2.Compose(operations)


def create_evaluation_transform():
    """Create deterministic conversion and CIFAR-10 normalization."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STANDARD_DEVIATION,
            ),
        ]
    )


class IndexedCIFAR10Dataset(Dataset):
    """Expose deterministic identifiers while applying a split-specific transform."""

    def __init__(self, base_dataset, indices, partition, transform):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.partition = str(partition)
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
            "identifier": f"cifar10_{self.partition}_{dataset_index:05d}",
            "dataset_index": int(dataset_index),
        }


def prepare_cifar10_datasets(configuration):
    """Download CIFAR-10 and return optimization plus unaugmented views."""
    paths = get_output_paths(configuration)
    download = bool(configuration["dataset"]["download"])
    training_base = CIFAR10(
        root=paths["cifar10_directory"],
        train=True,
        transform=None,
        download=download,
    )
    test_base = CIFAR10(
        root=paths["cifar10_directory"],
        train=False,
        transform=None,
        download=download,
    )
    configured_classes = configuration["dataset"]["known_class_names"]
    if list(training_base.classes) != configured_classes:
        raise ValueError("Downloaded CIFAR-10 class order does not match the config.")
    protocol = prepare_cifar10_split(
        training_base.targets,
        training_base.classes,
        paths["cifar10_split_file"],
    )
    training_transform = create_training_transform(configuration)
    evaluation_transform = create_evaluation_transform()
    all_test_indices = range(len(test_base))
    datasets = {
        "train": IndexedCIFAR10Dataset(
            training_base,
            protocol["train_indices"],
            partition="train",
            transform=training_transform,
        ),
        "train_evaluation": IndexedCIFAR10Dataset(
            training_base,
            protocol["train_indices"],
            partition="train",
            transform=evaluation_transform,
        ),
        "validation": IndexedCIFAR10Dataset(
            training_base,
            protocol["validation_indices"],
            partition="train",
            transform=evaluation_transform,
        ),
        "test": IndexedCIFAR10Dataset(
            test_base,
            all_test_indices,
            partition="test",
            transform=evaluation_transform,
        ),
    }
    return datasets, protocol
