# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Select and visualize confidently accepted near and far unknown examples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_accepted_unknown_failure_table(
    outputs,
    unknownness_scores,
    threshold,
    known_class_names,
    minimum_cases=3,
):
    """Select the lowest-unknownness mistakes that passed the known threshold."""
    scores = np.asarray(unknownness_scores, dtype=np.float64)
    predictions = np.asarray(outputs["known_logits"]).argmax(axis=1)
    table = pd.DataFrame(
        {
            "identifier": outputs["identifiers"],
            "dataset_index": outputs["dataset_indices"],
            "unknown_group": outputs["unknown_groups"],
            "unknown_class": outputs["class_names"],
            "predicted_known_index": predictions,
            "unknownness_score": scores,
            "threshold": float(threshold),
        }
    )
    table["predicted_known_class"] = table["predicted_known_index"].map(
        lambda index: known_class_names[int(index)]
    )
    accepted = table[table["unknownness_score"] <= float(threshold)].copy()
    accepted = accepted.sort_values("unknownness_score", ascending=True)
    selected_tables = []
    for group_name in ("near", "far"):
        group_table = accepted[accepted["unknown_group"] == group_name]
        if len(group_table) < int(minimum_cases):
            raise ValueError(
                f"Fewer than {minimum_cases} accepted {group_name} failures exist."
            )
        selected_tables.append(group_table.head(int(minimum_cases)))
    return pd.concat(selected_tables, ignore_index=True)


def plot_accepted_unknown_failures(base_dataset, failure_table, output_file):
    """Save a compact image grid for selected near and far MLS failures."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    number_of_examples = len(failure_table)
    columns = max(3, number_of_examples // 2)
    figure, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 6.4))
    axes = np.asarray(axes).reshape(-1)
    for axis, (_, row) in zip(axes, failure_table.iterrows()):
        image, _ = base_dataset[int(row["dataset_index"])]
        axis.imshow(image)
        axis.set_title(
            f"{row['unknown_group']}: {row['unknown_class']}\n"
            f"as {row['predicted_known_class']} ({row['unknownness_score']:.3f})"
        )
        axis.axis("off")
    for axis in axes[number_of_examples:]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
