# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate equal-weight empirical risk over the three source domains."""

import torch
from torch.nn import functional as torch_functional


def calculate_domain_balanced_erm_loss(logits_by_domain, labels_by_domain):
    """Average source cross-entropy so every domain has equal influence."""
    if set(logits_by_domain) != set(labels_by_domain):
        raise ValueError("Source logits and labels must contain the same domains.")
    if not logits_by_domain:
        raise ValueError("ERM requires at least one labeled source domain.")

    domain_losses = {}
    for domain in logits_by_domain:
        logits = logits_by_domain[domain]
        labels = labels_by_domain[domain]
        if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
            raise ValueError(f"Invalid logits or labels for source domain {domain}.")
        domain_losses[domain] = torch_functional.cross_entropy(logits, labels)

    classification_loss = torch.stack(list(domain_losses.values())).mean()
    return classification_loss, domain_losses
