# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement Reciprocal Point Learning classification and open-space losses."""

import torch
from torch import nn
from torch.nn import functional as torch_functional


class ReciprocalPointHead(nn.Module):
    """Represent the extra-class space using learnable points for every class."""

    def __init__(
        self,
        number_of_classes,
        feature_dimension,
        number_of_reciprocal_points=1,
        temperature=1.0,
        margin_initial_value=0.0,
    ):
        super().__init__()
        self.number_of_classes = int(number_of_classes)
        self.feature_dimension = int(feature_dimension)
        self.number_of_reciprocal_points = int(number_of_reciprocal_points)
        self.temperature = float(temperature)
        if self.number_of_classes < 2:
            raise ValueError("RPL requires at least two known classes.")
        if self.feature_dimension < 1 or self.number_of_reciprocal_points < 1:
            raise ValueError("RPL feature and reciprocal-point counts must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("RPL temperature must be positive.")

        self.reciprocal_points = nn.Parameter(
            0.1
            * torch.randn(
                self.number_of_classes,
                self.number_of_reciprocal_points,
                self.feature_dimension,
            )
        )
        self.class_margins = nn.Parameter(
            torch.full(
                (self.number_of_classes,),
                fill_value=float(margin_initial_value),
            )
        )

    def squared_distances(self, features):
        """Return mean squared distances to every reciprocal point and class."""
        if features.ndim != 2 or features.shape[1] != self.feature_dimension:
            raise ValueError("RPL received a feature matrix with the wrong shape.")
        # Keep distance calculations in FP32 when the backbone uses CUDA autocast.
        feature_values = features.float()
        point_values = self.reciprocal_points.float()
        differences = feature_values[:, None, None, :] - point_values[None, :, :, :]
        point_distances = differences.square().mean(dim=3)
        class_distances = point_distances.mean(dim=2)
        return point_distances, class_distances

    def forward(self, features):
        """Use greater otherness from a class reciprocal point as its class logit."""
        _, class_distances = self.squared_distances(features)
        return class_distances / self.temperature


def calculate_rpl_loss(model, images, labels, configuration):
    """Combine reciprocal classification and bounded open-space regularization."""
    if len(images) != len(labels) or labels.ndim != 1:
        raise ValueError("RPL images and labels must have matching batch lengths.")
    if model.reciprocal_head is None:
        raise ValueError("RPL loss requires a model with a reciprocal-point head.")

    features = model.forward_features(images)
    point_distances, class_distances = model.reciprocal_head.squared_distances(
        features
    )
    known_logits = class_distances / model.reciprocal_head.temperature
    classification_loss = torch_functional.cross_entropy(known_logits, labels)

    row_indices = torch.arange(len(labels), device=labels.device)
    true_class_distances = point_distances[row_indices, labels]
    true_class_margins = model.reciprocal_head.class_margins[labels].unsqueeze(1)
    open_space_loss = torch_functional.mse_loss(
        true_class_distances,
        true_class_margins.expand_as(true_class_distances),
    )
    total_loss = classification_loss + float(
        configuration["method"]["open_space_weight"]
    ) * open_space_loss
    return {
        "total_loss": total_loss,
        "classification_loss": classification_loss,
        "open_space_loss": open_space_loss,
        "known_logits": known_logits,
        "known_labels": labels,
        "mean_reciprocal_distance": class_distances.mean(),
        "mean_true_class_margin": model.reciprocal_head.class_margins[
            labels
        ].mean(),
    }
