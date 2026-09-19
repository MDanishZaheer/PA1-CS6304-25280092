# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create and reuse the fixed PACS protocol shared by Tasks 2 and 3."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from shared_task_2and3.pacs import (
    PACS_CLASSES,
    PACS_DOMAINS,
    PACSDomainDataset,
    create_evaluation_transform,
    create_training_transform,
    prepare_pacs_dataset,
    find_pacs_source_root,
    scan_domain_records,
)


SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"
SEED = 6304
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.20


def _class_counts(records):
    """Count the class labels in a collection of PACS records."""
    counts = Counter(record["class_name"] for record in records)
    return {class_name: int(counts[class_name]) for class_name in PACS_CLASSES}


def split_source_domain(records, seed=SEED):
    """Create the required stratified 80/20 split for one source domain."""
    if not np.isclose(TRAIN_RATIO + VALIDATION_RATIO, 1.0):
        raise ValueError("The source training and validation ratios must add to one.")
    if not records:
        raise ValueError("Cannot split an empty PACS source domain.")

    indices = np.arange(len(records))
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_RATIO,
        random_state=seed,
        stratify=labels,
    )
    train_records = [records[int(index)] for index in sorted(train_indices)]
    validation_records = [
        records[int(index)] for index in sorted(validation_indices)
    ]
    return train_records, validation_records


def build_pacs_protocol(image_root, seed=SEED):
    """Build the complete machine-readable PACS split description."""
    source_splits = {}
    for domain in SOURCE_DOMAINS:
        records = scan_domain_records(image_root, domain, include_labels=True)
        train_records, validation_records = split_source_domain(records, seed=seed)
        source_splits[domain] = {
            "all_count": len(records),
            "train_identifiers": [
                record["identifier"] for record in train_records
            ],
            "validation_identifiers": [
                record["identifier"] for record in validation_records
            ],
            "train_class_counts": _class_counts(train_records),
            "validation_class_counts": _class_counts(validation_records),
        }

    target_records = scan_domain_records(
        image_root,
        TARGET_DOMAIN,
        include_labels=False,
    )
    return {
        "dataset_name": "PACS",
        "seed": seed,
        "class_names": list(PACS_CLASSES),
        "source_domains": list(SOURCE_DOMAINS),
        "target_domain": TARGET_DOMAIN,
        "source_train_ratio": TRAIN_RATIO,
        "source_validation_ratio": VALIDATION_RATIO,
        "source_splits": source_splits,
        "target_adaptation": {
            "count": len(target_records),
            "identifiers": [record["identifier"] for record in target_records],
            "labels_exposed": False,
        },
    }


def save_pacs_protocol(protocol, split_file):
    """Save the shared PACS protocol as formatted JSON."""
    split_path = Path(split_file)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as file:
        json.dump(protocol, file, indent=2)


def load_pacs_protocol(split_file):
    """Load a previously saved PACS protocol."""
    with Path(split_file).open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_pacs_protocol(protocol, image_root):
    """Verify that saved identifiers still satisfy the assignment protocol."""
    if protocol.get("dataset_name") != "PACS":
        raise ValueError("The saved protocol belongs to a different dataset.")
    if protocol.get("seed") != SEED:
        raise ValueError("The saved PACS protocol uses a different seed.")
    if protocol.get("class_names") != list(PACS_CLASSES):
        raise ValueError("The saved PACS class order is incorrect.")
    if protocol.get("source_domains") != list(SOURCE_DOMAINS):
        raise ValueError("The saved source-domain order is incorrect.")
    if protocol.get("target_domain") != TARGET_DOMAIN:
        raise ValueError("The saved target domain is incorrect.")

    for domain in SOURCE_DOMAINS:
        available_records = scan_domain_records(image_root, domain, include_labels=True)
        available_identifiers = {
            record["identifier"] for record in available_records
        }
        domain_split = protocol["source_splits"][domain]
        train_identifiers = set(domain_split["train_identifiers"])
        validation_identifiers = set(domain_split["validation_identifiers"])
        if len(train_identifiers) != len(domain_split["train_identifiers"]):
            raise ValueError(f"The {domain} training split contains duplicates.")
        if len(validation_identifiers) != len(
            domain_split["validation_identifiers"]
        ):
            raise ValueError(f"The {domain} validation split contains duplicates.")
        if train_identifiers & validation_identifiers:
            raise ValueError(f"The {domain} source splits overlap.")
        if train_identifiers | validation_identifiers != available_identifiers:
            raise ValueError(f"The {domain} source splits do not cover the domain.")

        record_lookup = {
            record["identifier"]: record for record in available_records
        }
        training_records = [record_lookup[value] for value in train_identifiers]
        validation_records = [
            record_lookup[value] for value in validation_identifiers
        ]
        if _class_counts(training_records) != domain_split["train_class_counts"]:
            raise ValueError(f"The saved {domain} training class counts are incorrect.")
        if (
            _class_counts(validation_records)
            != domain_split["validation_class_counts"]
        ):
            raise ValueError(f"The saved {domain} validation class counts are incorrect.")
        for class_name in PACS_CLASSES:
            total_class_count = sum(
                record["class_name"] == class_name
                for record in available_records
            )
            validation_class_count = sum(
                record["class_name"] == class_name
                for record in validation_records
            )
            expected_validation_count = total_class_count * VALIDATION_RATIO
            if abs(validation_class_count - expected_validation_count) > 1:
                raise ValueError(f"The saved {domain} split is not stratified.")

    target_records = scan_domain_records(
        image_root,
        TARGET_DOMAIN,
        include_labels=False,
    )
    target_identifiers = {record["identifier"] for record in target_records}
    saved_target_identifiers = set(protocol["target_adaptation"]["identifiers"])
    if len(saved_target_identifiers) != len(
        protocol["target_adaptation"]["identifiers"]
    ):
        raise ValueError("The target adaptation identifiers contain duplicates.")
    if target_identifiers != saved_target_identifiers:
        raise ValueError("The saved target identifiers do not match the PACS dataset.")
    if protocol["target_adaptation"]["count"] != len(target_identifiers):
        raise ValueError("The saved target adaptation count is incorrect.")
    if protocol["target_adaptation"].get("labels_exposed") is not False:
        raise ValueError("The target adaptation protocol must not expose labels.")


def validate_pacs_source_protocol(protocol, image_root):
    """Validate only the Task 3 source portion without scanning Sketch."""
    if protocol.get("dataset_name") != "PACS":
        raise ValueError("The saved protocol belongs to a different dataset.")
    if protocol.get("seed") != SEED:
        raise ValueError("The saved PACS protocol uses a different seed.")
    if protocol.get("class_names") != list(PACS_CLASSES):
        raise ValueError("The saved PACS class order is incorrect.")
    if protocol.get("source_domains") != list(SOURCE_DOMAINS):
        raise ValueError("The saved source-domain order is incorrect.")
    if protocol.get("target_domain") != TARGET_DOMAIN:
        raise ValueError("The saved held-out domain is not Sketch.")
    if (
        not np.isclose(protocol.get("source_train_ratio", -1), TRAIN_RATIO)
        or not np.isclose(
            protocol.get("source_validation_ratio", -1),
            VALIDATION_RATIO,
        )
    ):
        raise ValueError("The saved protocol does not use the required 80/20 split.")

    for domain in SOURCE_DOMAINS:
        available_records = scan_domain_records(image_root, domain, include_labels=True)
        record_lookup = {
            record["identifier"]: record for record in available_records
        }
        available_identifiers = set(record_lookup)
        domain_split = protocol["source_splits"][domain]
        if domain_split.get("all_count") != len(available_records):
            raise ValueError(f"The saved {domain} total count is incorrect.")
        training_list = domain_split["train_identifiers"]
        validation_list = domain_split["validation_identifiers"]
        training_identifiers = set(training_list)
        validation_identifiers = set(validation_list)

        if len(training_identifiers) != len(training_list):
            raise ValueError(f"The {domain} training split contains duplicates.")
        if len(validation_identifiers) != len(validation_list):
            raise ValueError(f"The {domain} validation split contains duplicates.")
        if training_identifiers & validation_identifiers:
            raise ValueError(f"The {domain} source splits overlap.")
        if training_identifiers | validation_identifiers != available_identifiers:
            raise ValueError(f"The {domain} source splits do not cover the domain.")

        training_records = [record_lookup[value] for value in training_list]
        validation_records = [record_lookup[value] for value in validation_list]
        if _class_counts(training_records) != domain_split["train_class_counts"]:
            raise ValueError(f"The saved {domain} training counts are incorrect.")
        if (
            _class_counts(validation_records)
            != domain_split["validation_class_counts"]
        ):
            raise ValueError(f"The saved {domain} validation counts are incorrect.")
        for class_name in PACS_CLASSES:
            total_class_count = sum(
                record["class_name"] == class_name
                for record in available_records
            )
            validation_class_count = sum(
                record["class_name"] == class_name
                for record in validation_records
            )
            expected_validation_count = total_class_count * VALIDATION_RATIO
            if abs(validation_class_count - expected_validation_count) > 1:
                raise ValueError(f"The saved {domain} split is not stratified.")


def prepare_task3_source_protocol(dataset_directory, split_file):
    """Load the existing PACS sources and split without touching Sketch."""
    split_path = Path(split_file)
    if not split_path.exists():
        raise FileNotFoundError(
            "The shared PACS split does not exist. Run Task 2 data preparation first: "
            f"{split_path.resolve()}"
        )
    image_root = find_pacs_source_root(dataset_directory)
    protocol = load_pacs_protocol(split_path)
    validate_pacs_source_protocol(protocol, image_root)
    print(f"PACS source root: {image_root.resolve()}")
    print(f"Reused PACS protocol: {split_path.resolve()}")
    return image_root, protocol


def prepare_pacs_protocol(
    dataset_directory,
    download_directory,
    split_file,
    download=True,
    overwrite=False,
    require_official_counts=True,
):
    """Prepare PACS and create or reuse its validated shared split file."""
    image_root = prepare_pacs_dataset(
        dataset_directory=dataset_directory,
        download_directory=download_directory,
        download=download,
        require_official_counts=require_official_counts,
    )
    split_path = Path(split_file)
    if split_path.exists() and not overwrite:
        protocol = load_pacs_protocol(split_path)
        print("Using the existing shared PACS protocol.")
    else:
        protocol = build_pacs_protocol(image_root, seed=SEED)
        save_pacs_protocol(protocol, split_path)
        print("Created and saved the shared PACS protocol.")

    validate_pacs_protocol(protocol, image_root)
    print(f"Saved PACS protocol: {split_path.resolve()}")
    return image_root, protocol


def _select_records(all_records, selected_identifiers):
    """Select records in the exact order stored by the shared protocol."""
    record_lookup = {record["identifier"]: record for record in all_records}
    missing = [
        identifier
        for identifier in selected_identifiers
        if identifier not in record_lookup
    ]
    if missing:
        raise ValueError(f"The PACS protocol refers to missing images: {missing[:3]}")
    return [record_lookup[identifier] for identifier in selected_identifiers]


def build_task2_datasets(
    image_root,
    protocol,
    resize_size=256,
    crop_size=224,
    include_target_labels=False,
):
    """Build labeled source datasets and an unlabeled Task 2 target dataset."""
    training_transform = create_training_transform(resize_size, crop_size)
    evaluation_transform = create_evaluation_transform(resize_size, crop_size)
    datasets = {
        "source_train": {},
        "source_validation": {},
    }

    for domain in SOURCE_DOMAINS:
        all_records = scan_domain_records(image_root, domain, include_labels=True)
        domain_split = protocol["source_splits"][domain]
        training_records = _select_records(
            all_records,
            domain_split["train_identifiers"],
        )
        validation_records = _select_records(
            all_records,
            domain_split["validation_identifiers"],
        )
        datasets["source_train"][domain] = PACSDomainDataset(
            image_root,
            training_records,
            transform=training_transform,
            include_labels=True,
        )
        datasets["source_validation"][domain] = PACSDomainDataset(
            image_root,
            validation_records,
            transform=evaluation_transform,
            include_labels=True,
        )

    target_unlabeled_records = scan_domain_records(
        image_root,
        TARGET_DOMAIN,
        include_labels=False,
    )
    datasets["target_adaptation"] = PACSDomainDataset(
        image_root,
        target_unlabeled_records,
        transform=training_transform,
        include_labels=False,
    )
    datasets["target_diagnostic"] = PACSDomainDataset(
        image_root,
        target_unlabeled_records,
        transform=evaluation_transform,
        include_labels=False,
    )

    if include_target_labels:
        target_labeled_records = scan_domain_records(
            image_root,
            TARGET_DOMAIN,
            include_labels=True,
        )
        datasets["target_evaluation"] = PACSDomainDataset(
            image_root,
            target_labeled_records,
            transform=evaluation_transform,
            include_labels=True,
        )
    return datasets


def build_task3_source_datasets(
    image_root,
    protocol,
    resize_size=256,
    crop_size=224,
):
    """Build only the three labeled source domains required by Task 3."""
    training_transform = create_training_transform(resize_size, crop_size)
    evaluation_transform = create_evaluation_transform(resize_size, crop_size)
    datasets = {
        "source_train": {},
        "source_validation": {},
    }

    for domain in SOURCE_DOMAINS:
        all_records = scan_domain_records(image_root, domain, include_labels=True)
        domain_split = protocol["source_splits"][domain]
        training_records = _select_records(
            all_records,
            domain_split["train_identifiers"],
        )
        validation_records = _select_records(
            all_records,
            domain_split["validation_identifiers"],
        )
        datasets["source_train"][domain] = PACSDomainDataset(
            image_root,
            training_records,
            transform=training_transform,
            include_labels=True,
        )
        datasets["source_validation"][domain] = PACSDomainDataset(
            image_root,
            validation_records,
            transform=evaluation_transform,
            include_labels=True,
        )
    return datasets
