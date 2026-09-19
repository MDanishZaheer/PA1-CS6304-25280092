# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement target-free DAN-DG with pairwise source-domain MMD."""

from itertools import combinations

import torch

from task2.methods.dan import calculate_mmd_loss
from task3.methods.erm import calculate_domain_balanced_erm_loss


def calculate_pairwise_source_mmd(
    features_by_domain,
    bandwidth_multipliers=(0.5, 1.0, 2.0),
):
    """Average MMD over the three unordered pairs of source domains."""
    if len(features_by_domain) != 3:
        raise ValueError("DAN-DG requires exactly three source domains.")

    pair_losses = {}
    for first_domain, second_domain in combinations(features_by_domain, 2):
        pair_name = f"{first_domain}__{second_domain}"
        pair_loss, median_distance = calculate_mmd_loss(
            features_by_domain[first_domain],
            features_by_domain[second_domain],
            bandwidth_multipliers=bandwidth_multipliers,
        )
        pair_losses[pair_name] = {
            "mmd_loss": pair_loss,
            "median_squared_distance": median_distance,
        }

    average_mmd = torch.stack(
        [details["mmd_loss"] for details in pair_losses.values()]
    ).mean()
    return average_mmd, pair_losses


def calculate_dan_dg_loss(
    logits_by_domain,
    labels_by_domain,
    features_by_domain,
    mmd_weight=1.0,
    bandwidth_multipliers=(0.5, 1.0, 2.0),
):
    """Combine domain-balanced ERM with average pairwise source MMD."""
    classification_loss, domain_losses = calculate_domain_balanced_erm_loss(
        logits_by_domain,
        labels_by_domain,
    )
    mmd_loss, pair_losses = calculate_pairwise_source_mmd(
        features_by_domain,
        bandwidth_multipliers=bandwidth_multipliers,
    )
    total_loss = classification_loss + float(mmd_weight) * mmd_loss
    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "mmd_loss": mmd_loss,
        "domain_classification_losses": domain_losses,
        "pair_mmd_details": pair_losses,
    }
