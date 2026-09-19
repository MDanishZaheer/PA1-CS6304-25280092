# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Measure residual source-target information with logistic regression."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.inference_mode()
def extract_backbone_features(model, data_loader, device):
    """Extract deterministic pre-classifier features and image identifiers."""
    model.eval()
    feature_batches = []
    identifiers = []
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        features = model.backbone(images)
        feature_batches.append(features.float().cpu())
        identifiers.extend(str(value) for value in batch["identifier"])

    if not feature_batches:
        raise ValueError("Cannot extract features from an empty data loader.")
    return torch.cat(feature_batches).numpy(), identifiers


def balance_source_target_features(source_features, target_features, seed=6304):
    """Sample equal source and target feature counts with a fixed seed."""
    source_array = np.asarray(source_features, dtype=np.float32)
    target_array = np.asarray(target_features, dtype=np.float32)
    if source_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("Source and target features must be two-dimensional.")
    if source_array.shape[1] != target_array.shape[1]:
        raise ValueError("Source and target feature dimensions must match.")

    selected_count = min(len(source_array), len(target_array))
    if selected_count < 2:
        raise ValueError("Domain separability requires at least two examples per domain.")
    random_generator = np.random.default_rng(seed)
    source_indices = random_generator.choice(
        len(source_array),
        size=selected_count,
        replace=False,
    )
    target_indices = random_generator.choice(
        len(target_array),
        size=selected_count,
        replace=False,
    )
    return source_array[source_indices], target_array[target_indices]


def calculate_domain_separability(
    source_features,
    target_features,
    seed=6304,
    training_ratio=0.70,
    logistic_regression_c=1.0,
):
    """Train and evaluate the required balanced source-target domain probe."""
    balanced_source, balanced_target = balance_source_target_features(
        source_features,
        target_features,
        seed=seed,
    )
    features = np.concatenate([balanced_source, balanced_target], axis=0)
    domain_labels = np.concatenate(
        [
            np.zeros(len(balanced_source), dtype=np.int64),
            np.ones(len(balanced_target), dtype=np.int64),
        ]
    )
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        domain_labels,
        train_size=training_ratio,
        random_state=seed,
        stratify=domain_labels,
    )
    domain_probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(logistic_regression_c),
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )
    domain_probe.fit(train_features, train_labels)
    predictions = domain_probe.predict(test_features)
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "chance_accuracy": 0.5,
        "balanced_examples_per_domain": int(len(balanced_source)),
        "training_examples": int(len(train_labels)),
        "test_examples": int(len(test_labels)),
        "seed": int(seed),
        "logistic_regression_c": float(logistic_regression_c),
    }
