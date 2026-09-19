# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement class-conditional adversarial domain alignment for CDAN."""

import torch
from torch.nn import functional as torch_functional

from task2.methods.dann import create_binary_domain_labels, reverse_gradient


def create_conditional_representation(features, class_logits):
    """Vectorize the outer product of each feature and soft class prediction."""
    if features.ndim != 2 or class_logits.ndim != 2:
        raise ValueError("CDAN features and logits must be two-dimensional.")
    if len(features) != len(class_logits):
        raise ValueError("CDAN features and logits must contain equal batch sizes.")

    probabilities = torch_functional.softmax(class_logits, dim=1)
    outer_products = torch.bmm(
        probabilities.unsqueeze(2),
        features.unsqueeze(1),
    )
    return outer_products.flatten(start_dim=1)


def calculate_cdan_loss(
    source_logits,
    source_labels,
    source_features,
    target_logits,
    target_features,
    discriminator,
    grl_strength,
    domain_loss_weight=1.0,
):
    """Combine source classification with conditional adversarial alignment."""
    classification_loss = torch_functional.cross_entropy(
        source_logits,
        source_labels,
    )
    combined_features = torch.cat([source_features, target_features], dim=0)
    combined_logits = torch.cat([source_logits, target_logits], dim=0)
    conditional_features = create_conditional_representation(
        combined_features,
        combined_logits,
    )
    reversed_features = reverse_gradient(conditional_features, grl_strength)
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
