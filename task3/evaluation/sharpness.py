# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate the common local sharpness proxy on a fixed source batch."""

import numpy as np
import torch
from torch.nn import functional as torch_functional
from torch.utils.data import ConcatDataset, DataLoader, Subset


def select_fixed_validation_indices(
    validation_datasets,
    examples_per_domain=32,
    seed=6304,
):
    """Select the same fixed validation examples from each source domain."""
    random_generator = np.random.default_rng(seed)
    selected_indices = {}
    for domain, dataset in validation_datasets.items():
        if len(dataset) < examples_per_domain:
            raise ValueError(
                f"Source domain {domain} has fewer than {examples_per_domain} examples."
            )
        indices = random_generator.choice(
            len(dataset),
            size=examples_per_domain,
            replace=False,
        )
        selected_indices[domain] = sorted(int(index) for index in indices)
    return selected_indices


def create_fixed_validation_batch(
    validation_datasets,
    examples_per_domain=32,
    seed=6304,
):
    """Materialize one deterministic batch containing 32 examples per source."""
    selected_indices = select_fixed_validation_indices(
        validation_datasets,
        examples_per_domain=examples_per_domain,
        seed=seed,
    )
    subsets = [
        Subset(validation_datasets[domain], selected_indices[domain])
        for domain in validation_datasets
    ]
    combined_dataset = ConcatDataset(subsets)
    data_loader = DataLoader(
        combined_dataset,
        batch_size=len(combined_dataset),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(data_loader))
    return batch, selected_indices


def calculate_local_sharpness(model, validation_batch, device, radius=0.05):
    """Measure loss increase after one normalized gradient-ascent perturbation."""
    if radius <= 0.0:
        raise ValueError("Sharpness radius must be positive.")
    images = validation_batch["image"].to(device, non_blocking=True)
    labels = validation_batch["label"].to(device, non_blocking=True)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Sharpness evaluation requires trainable parameters.")

    model.eval()
    model.zero_grad(set_to_none=True)
    original_logits = model(images)
    original_loss = torch_functional.cross_entropy(original_logits, labels)
    gradients = torch.autograd.grad(
        original_loss,
        parameters,
        create_graph=False,
        retain_graph=False,
    )
    gradient_norm = torch.norm(
        torch.stack([gradient.detach().norm(p=2) for gradient in gradients])
    )
    if not torch.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise FloatingPointError("Sharpness requires a positive finite gradient norm.")

    perturbations = [
        float(radius) * gradient.detach() / (gradient_norm + 1e-12)
        for gradient in gradients
    ]
    original_parameters = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.add_(perturbation)
    try:
        with torch.no_grad():
            perturbed_logits = model(images)
            perturbed_loss = torch_functional.cross_entropy(
                perturbed_logits,
                labels,
            )
    finally:
        with torch.no_grad():
            for parameter, original_parameter in zip(
                parameters,
                original_parameters,
            ):
                parameter.copy_(original_parameter)
    model.zero_grad(set_to_none=True)
    return {
        "original_loss": float(original_loss.detach().item()),
        "perturbed_loss": float(perturbed_loss.detach().item()),
        "sharpness_increase": float(
            (perturbed_loss - original_loss.detach()).item()
        ),
        "gradient_norm": float(gradient_norm.detach().item()),
        "radius": float(radius),
        "number_of_examples": int(len(labels)),
    }
