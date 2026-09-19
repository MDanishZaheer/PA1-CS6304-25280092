# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load fixed models and calculate per-domain source metrics."""

import torch
from torch.utils.data import DataLoader

from task3.configs.config_loader import get_method_checkpoint_path
from task3.models.classifier_head import build_classifier
from task3.selection.source_validation import (
    evaluate_source_validation,
    load_erm_checkpoint,
)
from task3.training.train import load_task3_checkpoint


def create_evaluation_loader(dataset, configuration, generator_offset=0):
    """Create a deterministic non-shuffled evaluation data loader."""
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


def create_source_validation_loaders(datasets, configuration):
    """Create one deterministic loader for each labeled source validation set."""
    if set(datasets) != {"source_train", "source_validation"}:
        raise ValueError("Task 3 source evaluation received a non-source dataset.")
    expected_domains = set(configuration["dataset"]["source_domains"])
    for split_name in ("source_train", "source_validation"):
        if set(datasets[split_name]) != expected_domains:
            raise ValueError(
                f"Task 3 {split_name} must contain exactly the three sources."
            )
    return {
        domain: create_evaluation_loader(dataset, configuration, domain_index)
        for domain_index, (domain, dataset) in enumerate(
            datasets["source_validation"].items()
        )
    }


def load_model_for_evaluation(configuration, device):
    """Build and load the correct fixed ERM or Task 3 model checkpoint."""
    model = build_classifier(configuration, device)
    checkpoint_file = get_method_checkpoint_path(configuration)
    if configuration["method"]["name"] == "erm":
        checkpoint = load_erm_checkpoint(checkpoint_file, model, device)
    else:
        checkpoint = load_task3_checkpoint(
            checkpoint_file,
            model,
            configuration,
            device,
        )
    model.eval()
    return model, checkpoint, checkpoint_file


def evaluate_source_model(configuration, datasets, device):
    """Load one fixed model and calculate all source-validation metrics."""
    model, checkpoint, checkpoint_file = load_model_for_evaluation(
        configuration,
        device,
    )
    validation_loaders = create_source_validation_loaders(
        datasets,
        configuration,
    )
    metrics = evaluate_source_validation(model, validation_loaders, device)
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "validation_loaders": validation_loaders,
        "metrics": metrics,
    }
