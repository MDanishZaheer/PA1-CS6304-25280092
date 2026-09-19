# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build the fully trainable ImageNet-pretrained ResNet-18 backbone."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Backbone(nn.Module):
    """Return the 512-dimensional feature before the ResNet classifier."""

    def __init__(self, model_cache_directory=None):
        super().__init__()
        if model_cache_directory is not None:
            cache_path = Path(model_cache_directory)
            cache_path.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(cache_path))

        weights = ResNet18_Weights.IMAGENET1K_V1
        network = resnet18(weights=weights)
        self.feature_dimension = int(network.fc.in_features)
        network.fc = nn.Identity()
        self.network = network

    def forward(self, images):
        return self.network(images)


def freeze_batch_norm_statistics(model):
    """Freeze BatchNorm running statistics while leaving gamma and beta trainable."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return model


def count_model_parameters(model):
    """Return total and trainable parameter counts for one PyTorch model."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def build_backbone(configuration, device):
    """Build the configured Task 2 backbone and move it to the selected device."""
    from task2.configs.config_loader import get_output_paths

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
