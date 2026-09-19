# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Project and visualize combined clean and transformed representations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from task1.analysis.feature_similarity import validate_paired_features
from task1.analysis.evaluate_bias import make_json_serializable, to_numpy
from task1.configs.task1_config import (
    FIGURES_DIR,
    REPRESENTATION_METHOD,
    SEED,
    STL10_CLASSES,
    TSNE_COMPONENTS,
    TSNE_INITIALIZATION,
    TSNE_LEARNING_RATE,
    TSNE_MAX_ITERATIONS,
    TSNE_METRIC,
    TSNE_NORMALIZE_FEATURES,
    TSNE_PERPLEXITY,
)


def validate_projection_inputs(
    clean_features,
    transformed_features,
    labels,
    image_identifiers,
    class_names,
):
    """Validate the paired features and metadata used by one projection."""
    clean_array, transformed_array = validate_paired_features(
        clean_features,
        transformed_features,
    )
    labels_array = to_numpy(labels).astype(np.int64)
    identifiers = [str(identifier) for identifier in image_identifiers]

    if labels_array.ndim != 1 or len(labels_array) != len(clean_array):
        raise ValueError("Labels must match the number of paired feature rows.")
    if len(identifiers) != len(clean_array):
        raise ValueError("Image identifiers must match the paired feature rows.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Image identifiers must be unique within the projection subset.")
    if np.any(labels_array < 0) or np.any(labels_array >= len(class_names)):
        raise ValueError("A label is outside the supplied class-name range.")
    return clean_array, transformed_array, labels_array, identifiers


def normalize_feature_rows(features):
    """L2-normalize every feature row before distance-based projection."""
    feature_array = np.asarray(features, dtype=np.float64)
    row_norms = np.linalg.norm(feature_array, axis=1, keepdims=True)
    if np.any(row_norms == 0):
        raise ValueError("Cannot normalize a zero feature vector.")
    return feature_array / row_norms


def fit_combined_tsne(
    clean_features,
    transformed_features,
    labels,
    image_identifiers,
    transformed_condition,
    class_names=STL10_CLASSES,
    perplexity=TSNE_PERPLEXITY,
    max_iterations=TSNE_MAX_ITERATIONS,
    normalize_features=TSNE_NORMALIZE_FEATURES,
    seed=SEED,
):
    """Fit one t-SNE projection to combined clean and transformed features."""
    if REPRESENTATION_METHOD != "tsne":
        raise ValueError("The current Task 1 configuration does not select t-SNE.")
    if not transformed_condition or transformed_condition == "clean":
        raise ValueError("Transformed condition must have a non-clean name.")

    clean_array, transformed_array, labels_array, identifiers = (
        validate_projection_inputs(
            clean_features,
            transformed_features,
            labels,
            image_identifiers,
            class_names,
        )
    )
    combined_features = np.vstack((clean_array, transformed_array))
    if normalize_features:
        combined_features = normalize_feature_rows(combined_features)

    number_of_points = len(combined_features)
    if perplexity <= 0 or perplexity >= number_of_points:
        raise ValueError("t-SNE perplexity must be positive and smaller than the point count.")
    if max_iterations < 250:
        raise ValueError("t-SNE requires at least 250 iterations.")

    projection_model = TSNE(
        n_components=TSNE_COMPONENTS,
        perplexity=perplexity,
        learning_rate=TSNE_LEARNING_RATE,
        init=TSNE_INITIALIZATION,
        max_iter=max_iterations,
        metric=TSNE_METRIC,
        random_state=seed,
    )
    projected_features = projection_model.fit_transform(combined_features)

    number_of_pairs = len(clean_array)
    combined_labels = np.concatenate((labels_array, labels_array))
    combined_identifiers = identifiers + identifiers
    conditions = ["clean"] * number_of_pairs + [
        transformed_condition
    ] * number_of_pairs
    pair_indices = list(range(number_of_pairs)) * 2
    class_labels = [class_names[class_index] for class_index in combined_labels]

    projection_data = pd.DataFrame(
        {
            "projection_x": projected_features[:, 0],
            "projection_y": projected_features[:, 1],
            "pair_index": pair_indices,
            "image_identifier": combined_identifiers,
            "class_index": combined_labels,
            "class_name": class_labels,
            "condition": conditions,
        }
    )
    projection_settings = {
        "method": "t-SNE",
        "seed": seed,
        "number_of_pairs": number_of_pairs,
        "number_of_points": number_of_points,
        "original_feature_dimension": int(combined_features.shape[1]),
        "components": TSNE_COMPONENTS,
        "perplexity": float(perplexity),
        "learning_rate": TSNE_LEARNING_RATE,
        "initialization": TSNE_INITIALIZATION,
        "max_iterations": int(max_iterations),
        "metric": TSNE_METRIC,
        "features_l2_normalized": bool(normalize_features),
        "clean_condition": "clean",
        "transformed_condition": transformed_condition,
    }
    return projection_data, projection_settings


def plot_representation_projection(
    projection_data,
    backbone_name,
    intervention_name,
    class_names=STL10_CLASSES,
    output_file=None,
):
    """Plot class colors and clean-transformed marker styles in one space."""
    required_columns = {
        "projection_x",
        "projection_y",
        "class_index",
        "class_name",
        "condition",
    }
    missing_columns = required_columns - set(projection_data.columns)
    if missing_columns:
        raise ValueError(f"Projection data is missing columns: {sorted(missing_columns)}")

    conditions = projection_data["condition"].drop_duplicates().tolist()
    if "clean" not in conditions or len(conditions) != 2:
        raise ValueError("Projection data must contain clean and one transformed condition.")

    transformed_condition = next(
        condition for condition in conditions if condition != "clean"
    )
    markers = {"clean": "o", transformed_condition: "^"}
    color_map = plt.get_cmap("tab10", len(class_names))
    figure, axis = plt.subplots(figsize=(10, 7))

    for condition in ("clean", transformed_condition):
        condition_data = projection_data[projection_data["condition"] == condition]
        point_colors = [
            color_map(int(class_index))
            for class_index in condition_data["class_index"]
        ]
        axis.scatter(
            condition_data["projection_x"],
            condition_data["projection_y"],
            c=point_colors,
            marker=markers[condition],
            s=32,
            alpha=0.70,
            edgecolors="none",
        )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_map(class_index),
            markeredgecolor="none",
            markersize=7,
            label=class_name,
        )
        for class_index, class_name in enumerate(class_names)
    ]
    condition_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[condition],
            linestyle="none",
            color="black",
            markersize=7,
            label=condition.replace("_", " ").title(),
        )
        for condition in ("clean", transformed_condition)
    ]
    class_legend = axis.legend(
        handles=class_handles,
        title="Ground-truth class",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    axis.add_artist(class_legend)
    axis.legend(
        handles=condition_handles,
        title="Condition",
        bbox_to_anchor=(1.02, 0.35),
        loc="upper left",
    )

    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.set_title(f"{backbone_name}: clean vs. {intervention_name}")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure, axis


def create_output_stem(backbone_name, intervention_name):
    """Create a stable lowercase filename stem for representation outputs."""
    combined_name = f"{backbone_name}_{intervention_name}_tsne"
    return "_".join(combined_name.lower().replace("-", "_").split())


def save_projection_results(
    projection_data,
    projection_settings,
    backbone_name,
    intervention_name,
    output_directory=FIGURES_DIR,
    class_names=STL10_CLASSES,
):
    """Save projection coordinates, settings, and the rendered figure."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = create_output_stem(backbone_name, intervention_name)
    coordinates_file = output_directory / f"{output_stem}_coordinates.csv"
    settings_file = output_directory / f"{output_stem}_settings.json"
    figure_file = output_directory / f"{output_stem}.png"

    projection_data.to_csv(coordinates_file, index=False)
    with settings_file.open("w", encoding="utf-8") as file:
        json.dump(make_json_serializable(projection_settings), file, indent=2)
    figure, _ = plot_representation_projection(
        projection_data,
        backbone_name,
        intervention_name,
        class_names=class_names,
        output_file=figure_file,
    )
    return {
        "coordinates_file": coordinates_file,
        "settings_file": settings_file,
        "figure_file": figure_file,
        "figure": figure,
    }
