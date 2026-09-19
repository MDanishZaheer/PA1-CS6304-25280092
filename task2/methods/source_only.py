# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate the source-only empirical-risk-minimization objective."""

from torch.nn import functional as torch_functional


def calculate_source_only_loss(source_logits, source_labels):
    """Return cross-entropy over the domain-balanced labeled source batch."""
    if source_logits.ndim != 2:
        raise ValueError("Source logits must be a two-dimensional tensor.")
    if source_labels.ndim != 1 or len(source_labels) != len(source_logits):
        raise ValueError("Source labels must match the source logits.")
    classification_loss = torch_functional.cross_entropy(
        source_logits,
        source_labels,
    )
    return {
        "total_loss": classification_loss,
        "classification_loss": classification_loss,
        "alignment_loss": classification_loss.new_zeros(()),
        "domain_loss": classification_loss.new_zeros(()),
    }
