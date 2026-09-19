# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Measure information about the three observed PACS source domains."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task2.evaluation.domain_separability import extract_backbone_features


def balance_source_domain_features(features_by_domain, seed=6304):
    """Select equal feature counts from Photo, Art Painting, and Cartoon."""
    if len(features_by_domain) != 3:
        raise ValueError("Source-domain separability requires exactly three domains.")
    selected_count = min(len(features) for features in features_by_domain.values())
    if selected_count < 2:
        raise ValueError("Each source domain requires at least two feature vectors.")

    random_generator = np.random.default_rng(seed)
    balanced_features = {}
    for domain, features in features_by_domain.items():
        feature_array = np.asarray(features, dtype=np.float32)
        selected_indices = random_generator.choice(
            len(feature_array),
            size=selected_count,
            replace=False,
        )
        balanced_features[domain] = feature_array[selected_indices]
    return balanced_features


def calculate_source_domain_separability(
    features_by_domain,
    seed=6304,
    training_ratio=0.70,
    logistic_regression_c=1.0,
):
    """Train the required three-class balanced source-domain probe."""
    balanced_features = balance_source_domain_features(
        features_by_domain,
        seed=seed,
    )
    domains = list(balanced_features)
    feature_matrix = np.concatenate(
        [balanced_features[domain] for domain in domains],
        axis=0,
    )
    domain_labels = np.concatenate(
        [
            np.full(len(balanced_features[domain]), index, dtype=np.int64)
            for index, domain in enumerate(domains)
        ]
    )
    train_features, test_features, train_labels, test_labels = train_test_split(
        feature_matrix,
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
        "chance_accuracy": 1.0 / 3.0,
        "balanced_examples_per_domain": int(
            len(next(iter(balanced_features.values())))
        ),
        "training_examples": int(len(train_labels)),
        "test_examples": int(len(test_labels)),
        "domain_order": domains,
        "seed": int(seed),
        "logistic_regression_c": float(logistic_regression_c),
    }


def evaluate_source_domain_separability(
    model,
    validation_loaders,
    device,
    seed=6304,
    training_ratio=0.70,
    logistic_regression_c=1.0,
):
    """Extract source features on GPU and run the domain probe on CPU."""
    features_by_domain = {}
    for domain, data_loader in validation_loaders.items():
        features, _ = extract_backbone_features(model, data_loader, device)
        features_by_domain[domain] = features
    return calculate_source_domain_separability(
        features_by_domain,
        seed=seed,
        training_ratio=training_ratio,
        logistic_regression_c=logistic_regression_c,
    )
