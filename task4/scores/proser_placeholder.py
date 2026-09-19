# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Calibrate and calculate the PROSER placeholder-based unknownness score."""

import numpy as np


def calculate_dummy_calibration_bias(
    validation_known_logits,
    validation_dummy_logits,
    known_acceptance_percent=95.0,
):
    """Choose a dummy-logit bias that recognizes 95 percent of validation knowns."""
    known_logits = np.asarray(validation_known_logits, dtype=np.float64)
    dummy_logits = np.asarray(validation_dummy_logits, dtype=np.float64)
    if known_logits.ndim != 2 or dummy_logits.ndim != 2:
        raise ValueError("PROSER calibration requires two-dimensional logits.")
    if len(known_logits) != len(dummy_logits) or len(known_logits) == 0:
        raise ValueError("PROSER calibration logits must have matching lengths.")
    acceptance = float(known_acceptance_percent)
    if not 0.0 < acceptance < 100.0:
        raise ValueError("Known validation acceptance must lie between 0 and 100.")
    logit_gap = known_logits.max(axis=1) - dummy_logits.max(axis=1)
    return float(np.percentile(logit_gap, 100.0 - acceptance))


def calculate_proser_placeholder_unknownness(
    known_logits,
    dummy_logits,
    calibration_bias,
    temperature=1024.0,
):
    """Return calibrated dummy probability minus maximum known probability."""
    known_values = np.asarray(known_logits, dtype=np.float64)
    dummy_values = np.asarray(dummy_logits, dtype=np.float64)
    temperature = float(temperature)
    if known_values.ndim != 2 or dummy_values.ndim != 2:
        raise ValueError("PROSER scoring requires two-dimensional logits.")
    if len(known_values) != len(dummy_values):
        raise ValueError("PROSER known and dummy logits must have matching lengths.")
    if temperature <= 0.0:
        raise ValueError("PROSER score temperature must be positive.")

    strongest_dummy = dummy_values.max(axis=1, keepdims=True) + float(
        calibration_bias
    )
    collapsed_logits = np.concatenate([known_values, strongest_dummy], axis=1)
    scaled_logits = collapsed_logits / temperature
    shifted_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    dummy_probability = probabilities[:, -1]
    strongest_known_probability = probabilities[:, :-1].max(axis=1)
    return dummy_probability - strongest_known_probability
