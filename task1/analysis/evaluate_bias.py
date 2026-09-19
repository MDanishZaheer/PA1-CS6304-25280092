# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate predictions, confidence, consistency, and cue preferences."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as torch_functional

from task1.configs.task1_config import (
    CLIP_PROMPT_TEMPLATE,
    STL10_CLASSES,
    TRANSLATION_DIRECTIONS,
    TRANSLATION_PIXELS,
)


def to_numpy(values):
    """Move a tensor to the CPU or convert another array-like value."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def validate_logits_and_labels(logits, labels):
    """Validate and return classification logits and integer labels."""
    logits_array = to_numpy(logits).astype(np.float64)
    labels_array = to_numpy(labels).astype(np.int64)

    if logits_array.ndim != 2 or logits_array.shape[0] == 0:
        raise ValueError("Logits must be a non-empty two-dimensional array.")
    if labels_array.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")
    if logits_array.shape[0] != labels_array.shape[0]:
        raise ValueError("Logits and labels must contain the same number of examples.")
    if not np.all(np.isfinite(logits_array)):
        raise ValueError("Logits must contain only finite values.")
    if np.any(labels_array < 0) or np.any(labels_array >= logits_array.shape[1]):
        raise ValueError("A label is outside the classifier's output range.")
    return logits_array, labels_array


def softmax_from_logits(logits):
    """Calculate a numerically stable softmax for every example."""
    logits_array = to_numpy(logits).astype(np.float64)
    if logits_array.ndim != 2 or logits_array.shape[0] == 0:
        raise ValueError("Logits must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(logits_array)):
        raise ValueError("Logits must contain only finite values.")

    shifted_logits = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exponentials = np.exp(shifted_logits)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def get_predictions_and_confidences(logits):
    """Return predicted classes and their maximum softmax confidence."""
    probabilities = softmax_from_logits(logits)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    confidences = np.max(probabilities, axis=1)
    return predictions, confidences


def calculate_classification_metrics(logits, labels):
    """Calculate top-1 accuracy, macro-F1, and mean maximum confidence."""
    logits_array, labels_array = validate_logits_and_labels(logits, labels)
    predictions, confidences = get_predictions_and_confidences(logits_array)
    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "macro_f1": float(
            f1_score(labels_array, predictions, average="macro", zero_division=0)
        ),
        "mean_max_confidence": float(np.mean(confidences)),
    }


def calculate_per_class_accuracy(logits, labels, class_names=STL10_CLASSES):
    """Calculate correct counts and accuracy separately for every class."""
    logits_array, labels_array = validate_logits_and_labels(logits, labels)
    if logits_array.shape[1] != len(class_names):
        raise ValueError("The number of class names does not match the logits.")

    predictions, _ = get_predictions_and_confidences(logits_array)
    per_class_results = {}
    for class_index, class_name in enumerate(class_names):
        class_mask = labels_array == class_index
        total = int(np.sum(class_mask))
        correct = int(np.sum(predictions[class_mask] == class_index))
        accuracy = float(correct / total) if total > 0 else None
        per_class_results[class_name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
    return per_class_results


def calculate_prediction_consistency(clean_predictions, transformed_predictions):
    """Calculate the fraction of transformed predictions unchanged from clean."""
    clean_array = to_numpy(clean_predictions).astype(np.int64)
    transformed_array = to_numpy(transformed_predictions).astype(np.int64)
    if clean_array.ndim != 1 or transformed_array.ndim != 1:
        raise ValueError("Predictions must be one-dimensional arrays.")
    if clean_array.shape != transformed_array.shape or clean_array.size == 0:
        raise ValueError("Prediction arrays must have the same non-zero length.")
    return float(np.mean(clean_array == transformed_array))


def compare_transformation_with_clean(clean_logits, transformed_logits, labels):
    """Compare one intervention with its model-specific clean baseline."""
    clean_array, labels_array = validate_logits_and_labels(clean_logits, labels)
    transformed_array, _ = validate_logits_and_labels(transformed_logits, labels)
    if clean_array.shape != transformed_array.shape:
        raise ValueError("Clean and transformed logits must have the same shape.")

    clean_metrics = calculate_classification_metrics(clean_array, labels_array)
    transformed_metrics = calculate_classification_metrics(
        transformed_array,
        labels_array,
    )
    clean_predictions, _ = get_predictions_and_confidences(clean_array)
    transformed_predictions, _ = get_predictions_and_confidences(transformed_array)
    transformed_metrics["accuracy_change"] = (
        transformed_metrics["accuracy"] - clean_metrics["accuracy"]
    )
    transformed_metrics["macro_f1_change"] = (
        transformed_metrics["macro_f1"] - clean_metrics["macro_f1"]
    )
    transformed_metrics["mean_max_confidence_change"] = (
        transformed_metrics["mean_max_confidence"]
        - clean_metrics["mean_max_confidence"]
    )
    transformed_metrics["prediction_consistency"] = calculate_prediction_consistency(
        clean_predictions,
        transformed_predictions,
    )
    return transformed_metrics


def categorize_cue_conflict_predictions(predictions, content_labels, style_labels):
    """Label every cue-conflict decision as shape, texture, or other."""
    predictions_array = to_numpy(predictions).astype(np.int64)
    content_array = to_numpy(content_labels).astype(np.int64)
    style_array = to_numpy(style_labels).astype(np.int64)

    if predictions_array.ndim != 1:
        raise ValueError("Cue-conflict predictions must be one-dimensional.")
    if predictions_array.size == 0:
        raise ValueError("Cue-conflict evaluation requires at least one example.")
    if predictions_array.shape != content_array.shape:
        raise ValueError("Predictions and content labels must have the same shape.")
    if predictions_array.shape != style_array.shape:
        raise ValueError("Predictions and style labels must have the same shape.")
    if np.any(content_array == style_array):
        raise ValueError("Content and style labels must differ for every conflict.")

    categories = np.full(predictions_array.shape, "other", dtype=object)
    categories[predictions_array == content_array] = "shape"
    categories[predictions_array == style_array] = "texture"
    return categories


def calculate_cue_conflict_metrics(predictions, content_labels, style_labels):
    """Calculate decision counts, shape bias, and shape-texture coverage."""
    categories = categorize_cue_conflict_predictions(
        predictions,
        content_labels,
        style_labels,
    )
    shape_count = int(np.sum(categories == "shape"))
    texture_count = int(np.sum(categories == "texture"))
    other_count = int(np.sum(categories == "other"))
    total_count = int(categories.size)
    covered_count = shape_count + texture_count

    shape_bias = None
    if covered_count > 0:
        shape_bias = float(100.0 * shape_count / covered_count)
    coverage = float(100.0 * covered_count / total_count)
    return {
        "shape_count": shape_count,
        "texture_count": texture_count,
        "other_count": other_count,
        "total_count": total_count,
        "shape_bias_percent": shape_bias,
        "coverage_percent": coverage,
    }


def get_translation_logits(translated_logits, displacement):
    """Read one displacement from a dictionary with integer or string keys."""
    if displacement in translated_logits:
        return translated_logits[displacement]
    if str(displacement) in translated_logits:
        return translated_logits[str(displacement)]
    raise ValueError(f"Translation logits are missing displacement {displacement}.")


def summarize_translation_results(clean_logits, translated_logits, labels):
    """Average translation metrics across the four cardinal directions."""
    clean_array, labels_array = validate_logits_and_labels(clean_logits, labels)
    results = []

    for displacement in TRANSLATION_PIXELS:
        direction_results = {}
        if displacement == 0:
            direction_logits = {
                direction: clean_array for direction in TRANSLATION_DIRECTIONS
            }
        else:
            direction_logits = get_translation_logits(translated_logits, displacement)

        for direction in TRANSLATION_DIRECTIONS:
            if direction not in direction_logits:
                raise ValueError(
                    f"Translation logits for {displacement} pixels are missing {direction}."
                )
            direction_results[direction] = compare_transformation_with_clean(
                clean_array,
                direction_logits[direction],
                labels_array,
            )

        average_fields = (
            "accuracy",
            "macro_f1",
            "mean_max_confidence",
            "accuracy_change",
            "macro_f1_change",
            "mean_max_confidence_change",
            "prediction_consistency",
        )
        displacement_result = {"displacement": displacement}
        for field in average_fields:
            displacement_result[field] = float(
                np.mean(
                    [direction_results[direction][field] for direction in TRANSLATION_DIRECTIONS]
                )
            )
        displacement_result["direction_results"] = direction_results
        results.append(displacement_result)
    return results


def create_zero_shot_prompts(
    class_names=STL10_CLASSES,
    prompt_template=CLIP_PROMPT_TEMPLATE,
):
    """Create the assignment's fixed CLIP prompt for every class."""
    return [prompt_template.format(class_name) for class_name in class_names]


def calculate_clip_zero_shot_logits(image_features, text_features, similarity_scale):
    """Calculate scaled CLIP image-text similarities for zero-shot evaluation."""
    image_tensor = torch.as_tensor(image_features)
    if not image_tensor.is_floating_point():
        image_tensor = image_tensor.float()
    text_tensor = torch.as_tensor(
        text_features,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    )
    scale_tensor = torch.as_tensor(
        similarity_scale,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    )

    if image_tensor.ndim != 2 or text_tensor.ndim != 2:
        raise ValueError("CLIP image and text features must be two-dimensional.")
    if image_tensor.shape[1] != text_tensor.shape[1]:
        raise ValueError("CLIP image and text feature dimensions must match.")
    if scale_tensor.numel() != 1:
        raise ValueError("CLIP similarity scale must contain one value.")

    image_tensor = torch_functional.normalize(image_tensor, dim=-1)
    text_tensor = torch_functional.normalize(text_tensor, dim=-1)
    return scale_tensor * image_tensor @ text_tensor.T


def make_json_serializable(value):
    """Convert tensors, arrays, and paths into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_metrics(metrics, output_file):
    """Save calculated metrics as formatted JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(metrics), file, indent=2)
