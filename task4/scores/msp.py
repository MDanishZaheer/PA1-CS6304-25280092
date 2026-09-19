# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate Maximum Softmax Probability as an unknownness score."""

import numpy as np


def calculate_msp_unknownness(known_logits):
    """Return one minus maximum known-class probability for every example."""
    logits = np.asarray(known_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("MSP requires a non-empty two-dimensional logit matrix.")
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return (1.0 - probabilities.max(axis=1)).astype(np.float64)
