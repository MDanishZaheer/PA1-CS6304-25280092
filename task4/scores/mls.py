# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate the negative Maximum Logit Score unknownness measure."""

import numpy as np


def calculate_mls_unknownness(known_logits):
    """Negate each example's largest known-class logit."""
    logits = np.asarray(known_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("MLS requires a non-empty two-dimensional logit matrix.")
    return -logits.max(axis=1)
