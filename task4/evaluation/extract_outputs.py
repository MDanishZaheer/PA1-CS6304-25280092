# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Extract and cache identical logits and features for Task 4 scoring."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def create_evaluation_loader(dataset, configuration, generator_offset=0):
    """Build a deterministic non-shuffled evaluation loader."""
    training = configuration["training"]
    generator = torch.Generator()
    generator.manual_seed(int(configuration["seed"]) + int(generator_offset))
    number_of_workers = int(training["number_of_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=number_of_workers,
        pin_memory=bool(training["pin_memory"] and torch.cuda.is_available()),
        generator=generator,
        persistent_workers=number_of_workers > 0,
    )


@torch.inference_mode()
def collect_model_outputs(model, data_loader, device):
    """Collect known logits, optional dummy logits, features, and metadata."""
    model.eval()
    known_logit_batches = []
    dummy_logit_batches = []
    feature_batches = []
    labels = []
    identifiers = []
    dataset_indices = []
    class_names = []
    unknown_groups = []

    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        known_logits, dummy_logits, features = model.forward_with_dummy(images)
        known_logit_batches.append(known_logits.float().cpu())
        feature_batches.append(features.float().cpu())
        if dummy_logits is not None:
            dummy_logit_batches.append(dummy_logits.float().cpu())
        labels.extend(int(value) for value in batch["label"])
        identifiers.extend(str(value) for value in batch["identifier"])
        dataset_indices.extend(int(value) for value in batch["dataset_index"])
        class_names.extend(str(value) for value in batch["class_name"])
        if "unknown_group" in batch:
            unknown_groups.extend(str(value) for value in batch["unknown_group"])

    if not known_logit_batches:
        raise ValueError("Cannot extract outputs from an empty data loader.")
    known_logits = torch.cat(known_logit_batches).numpy()
    features = torch.cat(feature_batches).numpy()
    if dummy_logit_batches:
        dummy_logits = torch.cat(dummy_logit_batches).numpy()
    else:
        dummy_logits = np.empty((len(known_logits), 0), dtype=np.float32)
    if not unknown_groups:
        unknown_groups = [""] * len(known_logits)
    return {
        "known_logits": known_logits,
        "dummy_logits": dummy_logits,
        "features": features,
        "labels": np.asarray(labels, dtype=np.int64),
        "identifiers": np.asarray(identifiers, dtype=str),
        "dataset_indices": np.asarray(dataset_indices, dtype=np.int64),
        "class_names": np.asarray(class_names, dtype=str),
        "unknown_groups": np.asarray(unknown_groups, dtype=str),
    }


def save_output_cache(outputs, output_file):
    """Save model outputs as a compressed NumPy archive without pickle objects."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **outputs)
    return output_path


def save_feature_cache(outputs, output_file):
    """Save features, labels, and identifiers in the feature cache directory."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=outputs["features"],
        labels=outputs["labels"],
        identifiers=outputs["identifiers"],
        dataset_indices=outputs["dataset_indices"],
    )
    return output_path


def load_output_cache(output_file):
    """Load all arrays from a saved Task 4 output archive."""
    with np.load(Path(output_file), allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def create_prediction_table(outputs):
    """Create a compact table using predictions from known-class logits only."""
    predictions = np.asarray(outputs["known_logits"]).argmax(axis=1)
    return pd.DataFrame(
        {
            "identifier": outputs["identifiers"],
            "dataset_index": outputs["dataset_indices"],
            "label": outputs["labels"],
            "prediction": predictions,
            "class_name": outputs["class_names"],
            "unknown_group": outputs["unknown_groups"],
        }
    )
