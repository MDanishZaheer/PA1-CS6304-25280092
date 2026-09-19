# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate PROSER data placeholders by mixing different classes after layer2."""

import torch


def select_different_class_pairs(labels):
    """Shuffle a batch and retain only pairs whose class labels differ."""
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("Manifold mixup requires at least two labels.")
    if torch.unique(labels).numel() < 2:
        raise ValueError("Manifold mixup requires at least two different classes.")

    best_permutation = None
    best_mask = None
    for _ in range(20):
        permutation = torch.randperm(len(labels), device=labels.device)
        valid_mask = labels != labels[permutation]
        if best_mask is None or int(valid_mask.sum()) > int(best_mask.sum()):
            best_permutation = permutation
            best_mask = valid_mask
        if bool(valid_mask.all()):
            break
    if best_mask is None or not bool(best_mask.any()):
        raise RuntimeError("Could not form a different-class mixup pair.")
    return best_permutation, best_mask


def create_layer2_data_placeholders(
    layer2_features,
    labels,
    beta_distribution_alpha=2.0,
):
    """Mix valid different-class layer2 representations using Beta(alpha, alpha)."""
    if layer2_features.ndim != 4 or len(layer2_features) != len(labels):
        raise ValueError("Layer2 features and labels must have matching batches.")
    alpha = float(beta_distribution_alpha)
    if alpha <= 0.0:
        raise ValueError("The Beta distribution parameter must be positive.")
    permutation, valid_mask = select_different_class_pairs(labels)
    mixing_weight = torch.distributions.Beta(alpha, alpha).sample().to(
        device=layer2_features.device,
        dtype=layer2_features.dtype,
    )
    first_features = layer2_features[valid_mask]
    second_features = layer2_features[permutation][valid_mask]
    mixed_features = (
        mixing_weight * first_features
        + (1.0 - mixing_weight) * second_features
    )
    return mixed_features, mixing_weight, int(valid_mask.sum())
