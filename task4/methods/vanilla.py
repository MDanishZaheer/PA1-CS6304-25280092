# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate ordinary ten-class cross-entropy for the Vanilla model."""

from torch.nn import functional as torch_functional


def calculate_vanilla_loss(known_logits, labels):
    """Return closed-set cross-entropy over the ten CIFAR-10 classes."""
    if known_logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Vanilla logits and labels have invalid dimensions.")
    if len(known_logits) != len(labels):
        raise ValueError("Vanilla logits and labels must have matching lengths.")
    return torch_functional.cross_entropy(known_logits, labels)
