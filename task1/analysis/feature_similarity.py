# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Measure cosine stability between clean and transformed features."""

import numpy as np

from task1.analysis.evaluate_bias import to_numpy
from task1.configs.task1_config import (
    STL10_CLASSES,
    TRANSLATION_DIRECTIONS,
    TRANSLATION_PIXELS,
)


def validate_paired_features(clean_features, transformed_features):
    """Validate and return two aligned feature matrices."""
    clean_array = to_numpy(clean_features).astype(np.float64)
    transformed_array = to_numpy(transformed_features).astype(np.float64)

    if clean_array.ndim != 2 or transformed_array.ndim != 2:
        raise ValueError("Clean and transformed features must be two-dimensional.")
    if clean_array.shape != transformed_array.shape or clean_array.shape[0] == 0:
        raise ValueError("Paired feature matrices must have the same non-empty shape.")
    if not np.all(np.isfinite(clean_array)):
        raise ValueError("Clean features must contain only finite values.")
    if not np.all(np.isfinite(transformed_array)):
        raise ValueError("Transformed features must contain only finite values.")

    clean_norms = np.linalg.norm(clean_array, axis=1)
    transformed_norms = np.linalg.norm(transformed_array, axis=1)
    if np.any(clean_norms == 0) or np.any(transformed_norms == 0):
        raise ValueError("Cosine stability is undefined for a zero feature vector.")
    return clean_array, transformed_array


def validate_identifier_alignment(clean_identifiers, transformed_identifiers):
    """Verify that paired feature rows use the same image identifiers and order."""
    if clean_identifiers is None and transformed_identifiers is None:
        return
    if clean_identifiers is None or transformed_identifiers is None:
        raise ValueError("Both clean and transformed identifiers must be provided.")

    clean_list = [str(identifier) for identifier in clean_identifiers]
    transformed_list = [str(identifier) for identifier in transformed_identifiers]
    if len(clean_list) != len(transformed_list):
        raise ValueError("Clean and transformed identifier counts do not match.")
    if clean_list != transformed_list:
        mismatch_index = next(
            index
            for index, values in enumerate(zip(clean_list, transformed_list))
            if values[0] != values[1]
        )
        raise ValueError(
            f"Feature identifiers first differ at row {mismatch_index}: "
            f"{clean_list[mismatch_index]} != {transformed_list[mismatch_index]}."
        )


def calculate_cosine_similarities(clean_features, transformed_features):
    """Calculate one cosine similarity for every clean-transformed pair."""
    clean_array, transformed_array = validate_paired_features(
        clean_features,
        transformed_features,
    )
    dot_products = np.sum(clean_array * transformed_array, axis=1)
    clean_norms = np.linalg.norm(clean_array, axis=1)
    transformed_norms = np.linalg.norm(transformed_array, axis=1)
    similarities = dot_products / (clean_norms * transformed_norms)
    return np.clip(similarities, -1.0, 1.0)


def summarize_cosine_similarities(similarities):
    """Summarize a non-empty array of pairwise cosine similarities."""
    similarity_array = to_numpy(similarities).astype(np.float64)
    if similarity_array.ndim != 1 or similarity_array.size == 0:
        raise ValueError("Cosine similarities must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(similarity_array)):
        raise ValueError("Cosine similarities must contain only finite values.")

    return {
        "number_of_examples": int(similarity_array.size),
        "cosine_stability": float(np.mean(similarity_array)),
        "standard_deviation": float(np.std(similarity_array)),
        "median": float(np.median(similarity_array)),
        "minimum": float(np.min(similarity_array)),
        "maximum": float(np.max(similarity_array)),
    }


def calculate_feature_stability(clean_features, transformed_features):
    """Calculate the assignment's mean cosine representation stability."""
    similarities = calculate_cosine_similarities(
        clean_features,
        transformed_features,
    )
    return summarize_cosine_similarities(similarities)


def calculate_per_class_feature_stability(
    clean_features,
    transformed_features,
    labels,
    class_names=STL10_CLASSES,
):
    """Calculate mean cosine stability separately for each ground-truth class."""
    similarities = calculate_cosine_similarities(
        clean_features,
        transformed_features,
    )
    labels_array = to_numpy(labels).astype(np.int64)
    if labels_array.ndim != 1 or len(labels_array) != len(similarities):
        raise ValueError("Labels must match the number of paired feature rows.")
    if np.any(labels_array < 0) or np.any(labels_array >= len(class_names)):
        raise ValueError("A label is outside the supplied class-name range.")

    per_class_results = {}
    for class_index, class_name in enumerate(class_names):
        class_similarities = similarities[labels_array == class_index]
        if len(class_similarities) == 0:
            per_class_results[class_name] = {
                "number_of_examples": 0,
                "cosine_stability": None,
            }
        else:
            per_class_results[class_name] = summarize_cosine_similarities(
                class_similarities
            )
    return per_class_results


def summarize_feature_stability(
    clean_features,
    transformed_features,
    labels=None,
    clean_identifiers=None,
    transformed_identifiers=None,
    class_names=STL10_CLASSES,
):
    """Combine overall, alignment, and optional per-class stability checks."""
    validate_identifier_alignment(clean_identifiers, transformed_identifiers)
    clean_array, transformed_array = validate_paired_features(
        clean_features,
        transformed_features,
    )

    if clean_identifiers is not None and len(clean_identifiers) != len(clean_array):
        raise ValueError("Identifier count does not match the feature rows.")

    results = {
        "overall": calculate_feature_stability(clean_array, transformed_array),
    }
    if labels is not None:
        results["per_class"] = calculate_per_class_feature_stability(
            clean_array,
            transformed_array,
            labels,
            class_names,
        )
    return results


def get_translation_features(translated_features, displacement):
    """Read one displacement from a dictionary with integer or string keys."""
    if displacement in translated_features:
        return translated_features[displacement]
    if str(displacement) in translated_features:
        return translated_features[str(displacement)]
    raise ValueError(f"Translation features are missing displacement {displacement}.")


def summarize_translation_feature_stability(clean_features, translated_features):
    """Calculate translation stability by displacement and cardinal direction."""
    clean_array = to_numpy(clean_features)
    results = []

    for displacement in TRANSLATION_PIXELS:
        if displacement == 0:
            direction_features = {
                direction: clean_array for direction in TRANSLATION_DIRECTIONS
            }
        else:
            direction_features = get_translation_features(
                translated_features,
                displacement,
            )

        direction_results = {}
        all_similarities = []
        for direction in TRANSLATION_DIRECTIONS:
            if direction not in direction_features:
                raise ValueError(
                    f"Translation features for {displacement} pixels are missing "
                    f"{direction}."
                )
            similarities = calculate_cosine_similarities(
                clean_array,
                direction_features[direction],
            )
            direction_results[direction] = summarize_cosine_similarities(similarities)
            all_similarities.append(similarities)

        displacement_result = summarize_cosine_similarities(
            np.concatenate(all_similarities)
        )
        displacement_result["displacement"] = displacement
        displacement_result["direction_results"] = direction_results
        results.append(displacement_result)
    return results


def find_largest_feature_shifts(
    clean_features,
    transformed_features,
    image_identifiers,
    labels=None,
    top_count=10,
    class_names=STL10_CLASSES,
):
    """Return the examples with the lowest clean-transformed cosine similarity."""
    similarities = calculate_cosine_similarities(
        clean_features,
        transformed_features,
    )
    identifiers = [str(identifier) for identifier in image_identifiers]
    if len(identifiers) != len(similarities):
        raise ValueError("Image identifiers must match the number of feature pairs.")
    if not isinstance(top_count, int) or top_count < 1:
        raise ValueError("Top count must be a positive integer.")

    labels_array = None
    if labels is not None:
        labels_array = to_numpy(labels).astype(np.int64)
        if labels_array.ndim != 1 or len(labels_array) != len(similarities):
            raise ValueError("Labels must match the number of feature pairs.")

    selected_indices = np.argsort(similarities)[:min(top_count, len(similarities))]
    examples = []
    for index in selected_indices:
        example = {
            "row_index": int(index),
            "image_identifier": identifiers[index],
            "cosine_similarity": float(similarities[index]),
        }
        if labels_array is not None:
            class_index = int(labels_array[index])
            if class_index < 0 or class_index >= len(class_names):
                raise ValueError("A label is outside the supplied class-name range.")
            example["label_index"] = class_index
            example["class_name"] = class_names[class_index]
        examples.append(example)
    return examples


def compare_prediction_and_feature_stability(
    clean_features,
    transformed_features,
    clean_predictions,
    transformed_predictions,
):
    """Compare feature movement for stable and changed model predictions."""
    similarities = calculate_cosine_similarities(
        clean_features,
        transformed_features,
    )
    clean_array = to_numpy(clean_predictions).astype(np.int64)
    transformed_array = to_numpy(transformed_predictions).astype(np.int64)
    if clean_array.ndim != 1 or transformed_array.ndim != 1:
        raise ValueError("Predictions must be one-dimensional arrays.")
    if clean_array.shape != transformed_array.shape:
        raise ValueError("Clean and transformed predictions must have the same shape.")
    if len(clean_array) != len(similarities):
        raise ValueError("Predictions must match the number of feature pairs.")

    stable_mask = clean_array == transformed_array
    changed_mask = ~stable_mask
    stable_mean = None
    changed_mean = None
    if np.any(stable_mask):
        stable_mean = float(np.mean(similarities[stable_mask]))
    if np.any(changed_mask):
        changed_mean = float(np.mean(similarities[changed_mask]))

    return {
        "number_of_examples": int(len(similarities)),
        "stable_prediction_count": int(np.sum(stable_mask)),
        "changed_prediction_count": int(np.sum(changed_mask)),
        "prediction_consistency": float(np.mean(stable_mask)),
        "stable_prediction_cosine_stability": stable_mean,
        "changed_prediction_cosine_stability": changed_mean,
    }
