# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement PROSER classifier and data placeholder objectives."""

import torch
from torch.nn import functional as torch_functional

from task4.methods.manifold_mixup import create_layer2_data_placeholders


def collapse_dummy_logits(known_logits, dummy_logits):
    """Append the strongest of five dummy responses as one unknown logit."""
    if known_logits.ndim != 2 or dummy_logits is None or dummy_logits.ndim != 2:
        raise ValueError("PROSER requires two-dimensional known and dummy logits.")
    if len(known_logits) != len(dummy_logits) or dummy_logits.shape[1] < 1:
        raise ValueError("PROSER known and dummy logits must share a valid batch.")
    strongest_dummy = dummy_logits.max(dim=1, keepdim=True).values
    return torch.cat([known_logits, strongest_dummy], dim=1)


def calculate_classifier_placeholder_loss(
    known_logits,
    dummy_logits,
    labels,
    placeholder_weight=1.0,
):
    """Keep the true class first and the strongest dummy second for known data."""
    collapsed_logits = collapse_dummy_logits(known_logits, dummy_logits)
    number_of_known_classes = known_logits.shape[1]
    classification_loss = torch_functional.cross_entropy(
        collapsed_logits,
        labels,
    )
    masked_logits = collapsed_logits.clone()
    row_indices = torch.arange(len(labels), device=labels.device)
    masked_logits[row_indices, labels] = torch.finfo(masked_logits.dtype).min
    dummy_targets = torch.full_like(labels, fill_value=number_of_known_classes)
    placeholder_loss = torch_functional.cross_entropy(
        masked_logits,
        dummy_targets,
    )
    combined_loss = classification_loss + float(placeholder_weight) * placeholder_loss
    return combined_loss, classification_loss, placeholder_loss


def calculate_data_placeholder_loss(model, images, labels, configuration):
    """Train layer2 manifold-mixed representations toward the dummy class."""
    method = configuration["method"]
    layer2_features = model.forward_to_layer2(images)
    mixed_features, mixing_weight, number_of_pairs = (
        create_layer2_data_placeholders(
            layer2_features,
            labels,
            beta_distribution_alpha=method["beta_distribution_alpha"],
        )
    )
    final_features = model.forward_from_layer2(mixed_features)
    known_logits, dummy_logits = model.classify_features(final_features)
    collapsed_logits = collapse_dummy_logits(known_logits, dummy_logits)
    dummy_targets = torch.full(
        (len(collapsed_logits),),
        fill_value=known_logits.shape[1],
        dtype=torch.long,
        device=known_logits.device,
    )
    placeholder_loss = torch_functional.cross_entropy(
        collapsed_logits,
        dummy_targets,
    )
    return placeholder_loss, mixing_weight, number_of_pairs


def calculate_proser_loss(model, images, labels, configuration):
    """Split a mini-batch equally and combine both required PROSER losses."""
    if len(images) != len(labels) or len(images) < 4 or len(images) % 2 != 0:
        raise ValueError("PROSER requires an even mini-batch with at least four images.")
    half_batch = len(images) // 2
    classifier_images = images[:half_batch]
    classifier_labels = labels[:half_batch]
    mixup_images = images[half_batch:]
    mixup_labels = labels[half_batch:]
    known_logits, dummy_logits, _ = model.forward_with_dummy(classifier_images)
    classifier_loss, classification_loss, dummy_second_loss = (
        calculate_classifier_placeholder_loss(
            known_logits,
            dummy_logits,
            classifier_labels,
            placeholder_weight=configuration["method"][
                "classifier_placeholder_weight"
            ],
        )
    )
    data_placeholder_loss, mixing_weight, number_of_pairs = (
        calculate_data_placeholder_loss(
            model,
            mixup_images,
            mixup_labels,
            configuration,
        )
    )
    total_loss = (
        classifier_loss
        + float(configuration["method"]["data_placeholder_weight"])
        * data_placeholder_loss
    )
    return {
        "total_loss": total_loss,
        "classifier_placeholder_loss": classifier_loss,
        "classification_loss": classification_loss,
        "dummy_second_loss": dummy_second_loss,
        "data_placeholder_loss": data_placeholder_loss,
        "mixing_weight": float(mixing_weight.detach().item()),
        "number_of_mixup_pairs": int(number_of_pairs),
        "known_logits": known_logits,
        "known_labels": classifier_labels,
    }
