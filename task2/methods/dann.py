# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement DANN gradient reversal and binary domain classification."""

import math

import torch
from torch.nn import functional as torch_functional


class GradientReversalFunction(torch.autograd.Function):
    """Pass inputs forward and reverse their gradients during backpropagation."""

    @staticmethod
    def forward(ctx, inputs, strength):
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients):
        return -ctx.strength * gradients, None


def reverse_gradient(features, strength):
    """Apply gradient reversal with the requested non-negative strength."""
    if strength < 0.0:
        raise ValueError("Gradient-reversal strength cannot be negative.")
    return GradientReversalFunction.apply(features, strength)


def calculate_grl_strength(progress, maximum_strength=1.0):
    """Calculate the standard increasing DANN schedule at progress in [0, 1]."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("Training progress must lie in [0, 1].")
    schedule_value = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return float(maximum_strength) * schedule_value


def create_binary_domain_labels(source_count, target_count, device):
    """Create zero source labels and one target labels for the discriminator."""
    source_domain_labels = torch.zeros(source_count, dtype=torch.long, device=device)
    target_domain_labels = torch.ones(target_count, dtype=torch.long, device=device)
    return torch.cat([source_domain_labels, target_domain_labels], dim=0)


def calculate_dann_loss(
    source_logits,
    source_labels,
    source_features,
    target_features,
    discriminator,
    grl_strength,
    domain_loss_weight=1.0,
):
    """Combine source classification with adversarial marginal alignment."""
    classification_loss = torch_functional.cross_entropy(
        source_logits,
        source_labels,
    )
    combined_features = torch.cat([source_features, target_features], dim=0)
    reversed_features = reverse_gradient(combined_features, grl_strength)
    domain_logits = discriminator(reversed_features)
    domain_labels = create_binary_domain_labels(
        len(source_features),
        len(target_features),
        device=combined_features.device,
    )
    domain_loss = torch_functional.cross_entropy(domain_logits, domain_labels)
    total_loss = classification_loss + float(domain_loss_weight) * domain_loss
    domain_accuracy = (domain_logits.argmax(dim=1) == domain_labels).float().mean()
    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "alignment_loss": domain_loss,
        "domain_loss": domain_loss,
        "domain_accuracy": domain_accuracy,
    }
