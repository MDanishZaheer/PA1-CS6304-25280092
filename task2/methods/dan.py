# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement DAN with a median-bandwidth multi-kernel MMD objective."""

import torch
from torch.nn import functional as torch_functional


def pairwise_squared_distances(first_features, second_features):
    """Calculate non-negative squared Euclidean distances between feature rows."""
    first_squared = torch.sum(first_features ** 2, dim=1, keepdim=True)
    second_squared = torch.sum(second_features ** 2, dim=1).unsqueeze(0)
    distances = first_squared + second_squared - 2.0 * first_features @ second_features.T
    return torch.clamp(distances, min=0.0)


def median_pairwise_squared_distance(features, epsilon=1e-8):
    """Estimate bandwidth from positive off-diagonal distances in one batch."""
    with torch.no_grad():
        distances = pairwise_squared_distances(features, features)
        off_diagonal_mask = ~torch.eye(
            len(features),
            dtype=torch.bool,
            device=features.device,
        )
        candidates = distances[off_diagonal_mask]
        positive_candidates = candidates[candidates > epsilon]
        if positive_candidates.numel() == 0:
            return features.new_tensor(1.0)
        return torch.median(positive_candidates).clamp_min(epsilon)


def multi_kernel_rbf(
    first_features,
    second_features,
    base_bandwidth,
    bandwidth_multipliers=(0.5, 1.0, 2.0),
):
    """Sum three RBF kernels using multiples of the current median distance."""
    distances = pairwise_squared_distances(first_features, second_features)
    kernel = torch.zeros_like(distances)
    for multiplier in bandwidth_multipliers:
        bandwidth = base_bandwidth * float(multiplier)
        kernel = kernel + torch.exp(-distances / (2.0 * bandwidth))
    return kernel


def calculate_mmd_loss(
    source_features,
    target_features,
    bandwidth_multipliers=(0.5, 1.0, 2.0),
):
    """Calculate biased squared MMD between source and target feature batches."""
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("MMD features must be two-dimensional tensors.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target feature dimensions must match.")

    with torch.autocast(device_type=source_features.device.type, enabled=False):
        source_float = source_features.float()
        target_float = target_features.float()
        combined_features = torch.cat([source_float, target_float], dim=0)
        median_distance = median_pairwise_squared_distance(combined_features)
        source_kernel = multi_kernel_rbf(
            source_float,
            source_float,
            median_distance,
            bandwidth_multipliers,
        )
        target_kernel = multi_kernel_rbf(
            target_float,
            target_float,
            median_distance,
            bandwidth_multipliers,
        )
        cross_kernel = multi_kernel_rbf(
            source_float,
            target_float,
            median_distance,
            bandwidth_multipliers,
        )
        mmd_loss = (
            source_kernel.mean()
            + target_kernel.mean()
            - 2.0 * cross_kernel.mean()
        )
    return torch.clamp(mmd_loss, min=0.0), median_distance


def calculate_dan_loss(
    source_logits,
    source_labels,
    source_features,
    target_features,
    mmd_weight=1.0,
    bandwidth_multipliers=(0.5, 1.0, 2.0),
):
    """Combine source classification loss with weighted MMD alignment."""
    classification_loss = torch_functional.cross_entropy(
        source_logits,
        source_labels,
    )
    alignment_loss, median_distance = calculate_mmd_loss(
        source_features,
        target_features,
        bandwidth_multipliers=bandwidth_multipliers,
    )
    total_loss = classification_loss + float(mmd_weight) * alignment_loss
    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "alignment_loss": alignment_loss,
        "domain_loss": classification_loss.new_zeros(()),
        "median_squared_distance": median_distance,
    }
