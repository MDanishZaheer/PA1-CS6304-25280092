# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate and record shape-texture cue-conflict images."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.transforms import functional as transform_functional

from task1.configs.task1_config import (
    ADAIN_BATCH_SIZE,
    ADAIN_DECODER_WEIGHTS_FILE,
    ADAIN_DECODER_WEIGHTS_URL,
    ADAIN_VGG_WEIGHTS_FILE,
    ADAIN_VGG_WEIGHTS_URL,
    CUE_CONFLICT_CANDIDATES_PER_DIRECTION,
    CUE_CONFLICT_CLASS_PAIRS,
    CUE_CONFLICT_IMAGE_DIR,
    CUE_CONFLICT_MIN_VALID_IMAGES,
    CUE_CONFLICT_REJECTION_RULE,
    CUE_CONFLICT_REVIEW_FILE,
    CUE_CONFLICT_STYLE_STRENGTH,
    CUE_CONFLICT_SUMMARY_FILE,
    PROJECT_ROOT,
    SEED,
    STL10_CLASSES,
)
from task1.data.make_subset import get_labels, prepare_stl10_splits
from task1.data.transforms import prepare_common_image


# AdaIN architecture adapted from https://github.com/naoto0804/pytorch-AdaIN
# Copyright (c) 2018 Naoto Inoue, released under the MIT License.
ADAIN_SOURCE = "https://github.com/naoto0804/pytorch-AdaIN"


def build_adain_decoder():
    """Build the decoder architecture expected by the public AdaIN weights."""
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 256, 3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3),
    )


def build_adain_vgg():
    """Build the normalized VGG architecture expected by the AdaIN weights."""
    return nn.Sequential(
        nn.Conv2d(3, 3, 1),
        nn.ReflectionPad2d(1),
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 256, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 512, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, ceil_mode=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),
    )


def calculate_channel_statistics(features, epsilon=1e-5):
    """Calculate spatial mean and standard deviation for each feature channel."""
    if features.ndim != 4:
        raise ValueError("AdaIN features must have batch, channel, height, and width.")
    batch_size, channel_count = features.shape[:2]
    flattened_features = features.reshape(batch_size, channel_count, -1)
    feature_variance = flattened_features.var(dim=2, unbiased=True) + epsilon
    feature_standard_deviation = feature_variance.sqrt().reshape(
        batch_size,
        channel_count,
        1,
        1,
    )
    feature_mean = flattened_features.mean(dim=2).reshape(
        batch_size,
        channel_count,
        1,
        1,
    )
    return feature_mean, feature_standard_deviation


def adaptive_instance_normalization(content_features, style_features):
    """Replace content channel statistics with the style channel statistics."""
    if content_features.shape[:2] != style_features.shape[:2]:
        raise ValueError("Content and style feature batches must have matching channels.")
    style_mean, style_standard_deviation = calculate_channel_statistics(style_features)
    content_mean, content_standard_deviation = calculate_channel_statistics(
        content_features
    )
    normalized_content = (
        content_features - content_mean
    ) / content_standard_deviation
    return normalized_content * style_standard_deviation + style_mean


def freeze_network(network):
    """Freeze a pretrained AdaIN network and place it in evaluation mode."""
    for parameter in network.parameters():
        parameter.requires_grad = False
    network.eval()
    return network


def download_weight_file(url, output_file):
    """Download one missing public weight file to the configured cache."""
    output_path = Path(output_file)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    torch.hub.download_url_to_file(url, str(temporary_path), progress=True)
    if not temporary_path.exists() or temporary_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded AdaIN weight file is empty: {temporary_path}")
    temporary_path.replace(output_path)
    return output_path


def prepare_adain_weights(download_missing=True):
    """Return local AdaIN weight paths, downloading only when requested."""
    required_weights = (
        (ADAIN_VGG_WEIGHTS_URL, ADAIN_VGG_WEIGHTS_FILE),
        (ADAIN_DECODER_WEIGHTS_URL, ADAIN_DECODER_WEIGHTS_FILE),
    )
    for url, weight_file in required_weights:
        weight_path = Path(weight_file)
        weight_is_missing = (
            not weight_path.exists() or weight_path.stat().st_size == 0
        )
        if weight_is_missing:
            if not download_missing:
                raise FileNotFoundError(f"AdaIN weight file is missing: {weight_file}")
            download_weight_file(url, weight_file)
    return ADAIN_VGG_WEIGHTS_FILE, ADAIN_DECODER_WEIGHTS_FILE


def load_adain_models(device, download_missing=True):
    """Load the pretrained VGG encoder and decoder on the selected device."""
    vgg_file, decoder_file = prepare_adain_weights(download_missing)
    complete_vgg = build_adain_vgg()
    decoder = build_adain_decoder()
    complete_vgg.load_state_dict(
        torch.load(vgg_file, map_location="cpu", weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(decoder_file, map_location="cpu", weights_only=True)
    )

    encoder = nn.Sequential(*list(complete_vgg.children())[:31])
    encoder = freeze_network(encoder).to(device)
    decoder = freeze_network(decoder).to(device)
    return encoder, decoder


@torch.inference_mode()
def apply_adain_style_transfer(
    encoder,
    decoder,
    content_images,
    style_images,
    style_strength=CUE_CONFLICT_STYLE_STRENGTH,
):
    """Stylize a batch while blending the result with its content features."""
    if content_images.ndim != 4 or style_images.ndim != 4:
        raise ValueError("Content and style images must be four-dimensional batches.")
    if content_images.shape != style_images.shape:
        raise ValueError("Content and style image batches must have the same shape.")
    if not 0.0 <= style_strength <= 1.0:
        raise ValueError("AdaIN style strength must be between zero and one.")

    content_features = encoder(content_images)
    style_features = encoder(style_images)
    transferred_features = adaptive_instance_normalization(
        content_features,
        style_features,
    )
    blended_features = (
        style_strength * transferred_features
        + (1.0 - style_strength) * content_features
    )
    return decoder(blended_features).clamp(0.0, 1.0)


def get_test_identifier_map(split_data):
    """Map every selected official test index to its saved identifier."""
    identifier_map = {
        int(record["official_index"]): str(record["identifier"])
        for record in split_data["test_image_identifiers"]
    }
    selected_indices = [int(index) for index in split_data["indices"]["test"]]
    if set(identifier_map) != set(selected_indices):
        raise ValueError("Saved test identifiers do not match the selected indices.")
    return identifier_map


def create_cue_conflict_pairs(test_labels, split_data):
    """Create balanced deterministic content-style pairs in both directions."""
    labels_array = np.asarray(test_labels, dtype=np.int64)
    selected_indices = np.asarray(split_data["indices"]["test"], dtype=np.int64)
    identifier_map = get_test_identifier_map(split_data)
    random_generator = np.random.default_rng(SEED)
    class_to_indices = {}

    for class_index, class_name in enumerate(STL10_CLASSES):
        class_to_indices[class_name] = selected_indices[
            labels_array[selected_indices] == class_index
        ]
        if len(class_to_indices[class_name]) == 0:
            raise ValueError(f"No selected test images are available for {class_name}.")

    records = []
    for first_class, second_class in CUE_CONFLICT_CLASS_PAIRS:
        class_pair = f"{first_class}__{second_class}"
        directions = (
            (first_class, second_class),
            (second_class, first_class),
        )
        for content_class, style_class in directions:
            content_pool = class_to_indices[content_class]
            style_pool = class_to_indices[style_class]
            content_indices = random_generator.choice(
                content_pool,
                size=CUE_CONFLICT_CANDIDATES_PER_DIRECTION,
                replace=len(content_pool) < CUE_CONFLICT_CANDIDATES_PER_DIRECTION,
            )
            style_indices = random_generator.choice(
                style_pool,
                size=CUE_CONFLICT_CANDIDATES_PER_DIRECTION,
                replace=len(style_pool) < CUE_CONFLICT_CANDIDATES_PER_DIRECTION,
            )

            for pair_number, (content_index, style_index) in enumerate(
                zip(content_indices, style_indices)
            ):
                content_index = int(content_index)
                style_index = int(style_index)
                conflict_id = (
                    f"shape_{content_class}_texture_{style_class}_{pair_number:03d}"
                )
                output_path = CUE_CONFLICT_IMAGE_DIR / f"{conflict_id}.png"
                records.append(
                    {
                        "conflict_id": conflict_id,
                        "class_pair": class_pair,
                        "direction": f"{content_class}_shape__{style_class}_texture",
                        "content_identifier": identifier_map[content_index],
                        "content_index": content_index,
                        "content_label_index": STL10_CLASSES.index(content_class),
                        "content_class": content_class,
                        "style_identifier": identifier_map[style_index],
                        "style_index": style_index,
                        "style_label_index": STL10_CLASSES.index(style_class),
                        "style_class": style_class,
                        "style_strength": CUE_CONFLICT_STYLE_STRENGTH,
                        "output_file": output_path.relative_to(PROJECT_ROOT).as_posix(),
                        "review_status": "pending",
                        "rejection_reason": "",
                    }
                )
    return records


def validate_candidate_metadata(review_data, require_images=False):
    """Validate candidate identities, labels, directions, and optional image files."""
    required_columns = {
        "conflict_id",
        "class_pair",
        "direction",
        "content_identifier",
        "content_index",
        "content_label_index",
        "content_class",
        "style_identifier",
        "style_index",
        "style_label_index",
        "style_class",
        "style_strength",
        "output_file",
        "review_status",
        "rejection_reason",
    }
    missing_columns = required_columns - set(review_data.columns)
    if missing_columns:
        raise ValueError(f"Cue-conflict metadata is missing: {sorted(missing_columns)}")
    if review_data["conflict_id"].duplicated().any():
        raise ValueError("Cue-conflict identifiers must be unique.")
    if review_data["output_file"].duplicated().any():
        raise ValueError("Cue-conflict output paths must be unique.")
    if (review_data["content_class"] == review_data["style_class"]).any():
        raise ValueError("Every cue conflict must use different content and style classes.")

    valid_statuses = {"pending", "accepted", "rejected"}
    normalized_statuses = review_data["review_status"].astype(str).str.strip().str.lower()
    recorded_statuses = set(normalized_statuses)
    if not recorded_statuses <= valid_statuses:
        raise ValueError("Review status must be pending, accepted, or rejected.")

    class_indices = {
        class_name: class_index
        for class_index, class_name in enumerate(STL10_CLASSES)
    }
    expected_content_labels = review_data["content_class"].map(class_indices)
    expected_style_labels = review_data["style_class"].map(class_indices)
    if expected_content_labels.isna().any() or expected_style_labels.isna().any():
        raise ValueError("Cue-conflict metadata contains an unknown STL-10 class.")
    if not np.array_equal(
        review_data["content_label_index"].to_numpy(dtype=np.int64),
        expected_content_labels.to_numpy(dtype=np.int64),
    ):
        raise ValueError("A cue-conflict content class has the wrong label index.")
    if not np.array_equal(
        review_data["style_label_index"].to_numpy(dtype=np.int64),
        expected_style_labels.to_numpy(dtype=np.int64),
    ):
        raise ValueError("A cue-conflict style class has the wrong label index.")
    if not np.allclose(
        review_data["style_strength"].to_numpy(dtype=float),
        CUE_CONFLICT_STYLE_STRENGTH,
    ):
        raise ValueError("Cue-conflict metadata contains the wrong AdaIN strength.")

    expected_candidates = (
        len(CUE_CONFLICT_CLASS_PAIRS)
        * 2
        * CUE_CONFLICT_CANDIDATES_PER_DIRECTION
    )
    if len(review_data) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} cue-conflict candidates, "
            f"but found {len(review_data)}."
        )

    expected_directions = set()
    for first_class, second_class in CUE_CONFLICT_CLASS_PAIRS:
        class_pair = f"{first_class}__{second_class}"
        expected_directions.add(
            (class_pair, f"{first_class}_shape__{second_class}_texture")
        )
        expected_directions.add(
            (class_pair, f"{second_class}_shape__{first_class}_texture")
        )
    recorded_directions = set(
        review_data[["class_pair", "direction"]].itertuples(index=False, name=None)
    )
    if recorded_directions != expected_directions:
        raise ValueError("Cue-conflict metadata does not contain the configured directions.")

    direction_counts = review_data.groupby(["class_pair", "direction"]).size()
    if not (direction_counts == CUE_CONFLICT_CANDIDATES_PER_DIRECTION).all():
        raise ValueError("Cue-conflict class pairs and directions are not balanced.")

    if require_images:
        missing_images = [
            output_file
            for output_file in review_data["output_file"]
            if not (PROJECT_ROOT / output_file).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"Missing {len(missing_images)} generated cue-conflict images."
            )


def create_review_summary(review_data):
    """Summarize visual-review counts across pairs and directions."""
    status_counts = review_data["review_status"].str.lower().value_counts()
    direction_summaries = {}
    for (class_pair, direction), group in review_data.groupby(
        ["class_pair", "direction"]
    ):
        group_counts = group["review_status"].str.lower().value_counts()
        direction_summaries[f"{class_pair}:{direction}"] = {
            "candidate_count": int(len(group)),
            "pending_count": int(group_counts.get("pending", 0)),
            "accepted_count": int(group_counts.get("accepted", 0)),
            "rejected_count": int(group_counts.get("rejected", 0)),
        }

    accepted_count = int(status_counts.get("accepted", 0))
    return {
        "seed": SEED,
        "source_implementation": ADAIN_SOURCE,
        "style_strength": CUE_CONFLICT_STYLE_STRENGTH,
        "visual_rejection_rule": CUE_CONFLICT_REJECTION_RULE,
        "minimum_required_valid_images": CUE_CONFLICT_MIN_VALID_IMAGES,
        "candidate_count": int(len(review_data)),
        "pending_count": int(status_counts.get("pending", 0)),
        "accepted_count": accepted_count,
        "rejected_count": int(status_counts.get("rejected", 0)),
        "minimum_valid_requirement_met": (
            accepted_count >= CUE_CONFLICT_MIN_VALID_IMAGES
        ),
        "pair_direction_counts": direction_summaries,
    }


def save_visual_review(
    review_data,
    review_file=CUE_CONFLICT_REVIEW_FILE,
    summary_file=CUE_CONFLICT_SUMMARY_FILE,
):
    """Save the editable review table and its machine-readable count summary."""
    validate_candidate_metadata(review_data)
    review_path = Path(review_file)
    summary_path = Path(summary_file)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    review_data.to_csv(review_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(create_review_summary(review_data), file, indent=2)


def load_visual_review(review_file=CUE_CONFLICT_REVIEW_FILE):
    """Load the cue-conflict review table without interpreting model outputs."""
    review_data = pd.read_csv(review_file, keep_default_na=False)
    review_data["review_status"] = review_data["review_status"].str.lower()
    validate_candidate_metadata(review_data)
    return review_data


def validate_completed_visual_review(review_data):
    """Require a completed visual-only review with at least 200 accepted images."""
    validate_candidate_metadata(review_data, require_images=True)
    pending_count = int(np.sum(review_data["review_status"].str.lower() == "pending"))
    if pending_count > 0:
        raise ValueError(f"Visual review still contains {pending_count} pending images.")

    rejected_rows = review_data[review_data["review_status"].str.lower() == "rejected"]
    if rejected_rows["rejection_reason"].str.strip().eq("").any():
        raise ValueError("Every rejected image must include a rejection reason.")

    accepted_count = int(
        np.sum(review_data["review_status"].str.lower() == "accepted")
    )
    if accepted_count < CUE_CONFLICT_MIN_VALID_IMAGES:
        raise ValueError(
            f"Only {accepted_count} valid conflicts were accepted; "
            f"at least {CUE_CONFLICT_MIN_VALID_IMAGES} are required."
        )
    return create_review_summary(review_data)


def load_accepted_cue_conflicts(review_file=CUE_CONFLICT_REVIEW_FILE):
    """Return only visually accepted conflicts after review validation."""
    review_data = load_visual_review(review_file)
    validate_completed_visual_review(review_data)
    accepted_mask = review_data["review_status"] == "accepted"
    return review_data[accepted_mask].reset_index(drop=True)


def select_device(device=None):
    """Use the requested device or select CUDA automatically when available."""
    selected_device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return selected_device


def load_candidate_image(test_dataset, record, image_role):
    """Load and verify one content or style image from the official test set."""
    index = int(record[f"{image_role}_index"])
    expected_label = int(record[f"{image_role}_label_index"])
    image, label = test_dataset[index]
    if int(label) != expected_label:
        raise ValueError(
            f"The {image_role} label for official test index {index} is incorrect."
        )
    common_image = prepare_common_image(image)
    return transform_functional.to_tensor(common_image)


def generate_cue_conflict_candidates(
    test_dataset,
    split_data,
    device=None,
    download_weights=True,
    overwrite=False,
):
    """Generate all balanced candidates and initialize their visual-review file."""
    if CUE_CONFLICT_REVIEW_FILE.exists() and not overwrite:
        review_data = load_visual_review(CUE_CONFLICT_REVIEW_FILE)
        validate_candidate_metadata(review_data, require_images=True)
        print("Using existing cue-conflict candidates and visual-review table.")
        return review_data

    selected_device = select_device(device)
    test_labels = get_labels(test_dataset)
    candidate_records = create_cue_conflict_pairs(test_labels, split_data)
    encoder, decoder = load_adain_models(selected_device, download_weights)
    CUE_CONFLICT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    for start_index in range(0, len(candidate_records), ADAIN_BATCH_SIZE):
        batch_records = candidate_records[start_index:start_index + ADAIN_BATCH_SIZE]
        content_images = torch.stack(
            [
                load_candidate_image(test_dataset, record, "content")
                for record in batch_records
            ]
        ).to(selected_device)
        style_images = torch.stack(
            [
                load_candidate_image(test_dataset, record, "style")
                for record in batch_records
            ]
        ).to(selected_device)
        generated_images = apply_adain_style_transfer(
            encoder,
            decoder,
            content_images,
            style_images,
        ).cpu()

        for record, generated_image in zip(batch_records, generated_images):
            output_path = PROJECT_ROOT / record["output_file"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            transform_functional.to_pil_image(generated_image).save(output_path)

        completed_count = min(start_index + ADAIN_BATCH_SIZE, len(candidate_records))
        print(f"Generated cue conflicts: {completed_count}/{len(candidate_records)}")

    review_data = pd.DataFrame(candidate_records)
    validate_candidate_metadata(review_data, require_images=True)
    save_visual_review(review_data)
    print(f"Device used for AdaIN: {selected_device}")
    print(f"Visual-review file: {CUE_CONFLICT_REVIEW_FILE}")
    print("All candidates are pending visual review before model evaluation.")
    return review_data


def main():
    """Prepare STL-10 and generate cue-conflict candidates when run as a module."""
    _, test_dataset, split_data = prepare_stl10_splits(
        download=True,
        overwrite=False,
    )
    generate_cue_conflict_candidates(
        test_dataset,
        split_data,
        device=None,
        download_weights=True,
        overwrite=False,
    )


if __name__ == "__main__":
    main()
