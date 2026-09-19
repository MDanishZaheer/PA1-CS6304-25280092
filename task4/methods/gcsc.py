# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate GCSC classification loss after its stronger data augmentation."""

from task4.methods.vanilla import calculate_vanilla_loss


def calculate_gcsc_loss(known_logits, labels):
    """Use the same closed-set loss so RandAugment is the only method change."""
    return calculate_vanilla_loss(known_logits, labels)
