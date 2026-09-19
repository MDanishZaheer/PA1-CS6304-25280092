# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build Task 3 with the exact Task 2 ResNet-18 backbone implementation."""

from task2.models.backbone import (
    ResNet18Backbone,
    count_model_parameters,
    freeze_batch_norm_statistics,
)
from task3.configs.config_loader import get_output_paths


def build_backbone(configuration, device):
    """Build the checkpoint-compatible pretrained backbone for Task 3."""
    paths = get_output_paths(configuration)
    backbone = ResNet18Backbone(paths["model_cache_directory"])
    if backbone.feature_dimension != configuration["model"]["feature_dimension"]:
        raise ValueError("The configured feature dimension does not match ResNet-18.")
    backbone = backbone.to(device)
    total_parameters, trainable_parameters = count_model_parameters(backbone)
    print(f"Backbone: ResNet-18 ({configuration['model']['pretrained_weights']})")
    print(f"Feature dimension: {backbone.feature_dimension}")
    print(f"Trainable backbone parameters: {trainable_parameters:,}/{total_parameters:,}")
    return backbone


__all__ = [
    "ResNet18Backbone",
    "build_backbone",
    "count_model_parameters",
    "freeze_batch_norm_statistics",
]
