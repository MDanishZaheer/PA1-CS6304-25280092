# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calculate energy unknownness from all ten known-class logits."""

import numpy as np


def calculate_energy_unknownness(known_logits, temperature=1.0):
    """Return negative temperature-scaled log-sum-exp for each example."""
    logits = np.asarray(known_logits, dtype=np.float64)
    temperature = float(temperature)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("Energy requires a non-empty two-dimensional logit matrix.")
    if temperature <= 0.0:
        raise ValueError("Energy temperature must be positive.")
    scaled_logits = logits / temperature
    maximum = scaled_logits.max(axis=1, keepdims=True)
    log_sum_exp = maximum[:, 0] + np.log(
        np.exp(scaled_logits - maximum).sum(axis=1)
    )
    return -temperature * log_sum_exp
