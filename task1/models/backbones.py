# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build frozen ResNet, ViT, and CLIP feature extractors."""

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as torch_functional
from torchvision import models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from torchvision.transforms import functional as transform_functional

from task1.configs.task1_config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    IMAGE_SIZE,
    OPENCLIP_CACHE_DIR,
    RESNET_MODEL_NAME,
    STL10_CLASSES,
    TORCH_HUB_DIR,
    VIT_MODEL_NAME,
)

try:
    import open_clip
except ImportError:
    open_clip = None


class ImageNormalizer:
    """Convert one common RGB PIL image to a normalized tensor."""

    def __init__(self, mean, standard_deviation):
        self.mean = tuple(float(value) for value in mean)
        self.standard_deviation = tuple(float(value) for value in standard_deviation)
        if len(self.mean) != 3 or len(self.standard_deviation) != 3:
            raise ValueError("Image normalization requires three values per statistic.")

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("Model normalization expects a PIL image.")
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Model input must be an RGB image of size {IMAGE_SIZE} by {IMAGE_SIZE}."
            )

        image_tensor = transform_functional.to_tensor(image)
        return transform_functional.normalize(
            image_tensor,
            mean=self.mean,
            std=self.standard_deviation,
        )


def freeze_model(model):
    """Freeze every parameter and place a pretrained model in evaluation mode."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def count_parameters(model):
    """Return total and trainable parameter counts for a model."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


class ResNet50Backbone(nn.Module):
    """Return the global-average-pooled ResNet-50 representation."""

    def __init__(self):
        super().__init__()
        TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(TORCH_HUB_DIR))

        weights = ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        self.feature_dimension = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.model = freeze_model(resnet)

        weight_transforms = weights.transforms()
        self.normalization = ImageNormalizer(
            weight_transforms.mean,
            weight_transforms.std,
        )
        self.model_name = RESNET_MODEL_NAME

    def forward(self, images):
        return self.model(images)


class ViTB16Backbone(nn.Module):
    """Return the final class-token representation from ViT-B/16."""

    def __init__(self):
        super().__init__()
        TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(TORCH_HUB_DIR))

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        vision_transformer = models.vit_b_16(weights=weights)
        self.feature_dimension = vision_transformer.hidden_dim
        vision_transformer.heads = nn.Identity()
        self.model = freeze_model(vision_transformer)

        weight_transforms = weights.transforms()
        self.normalization = ImageNormalizer(
            weight_transforms.mean,
            weight_transforms.std,
        )
        self.model_name = VIT_MODEL_NAME

    def forward(self, images):
        return self.model(images)


class OpenCLIPBackbone(nn.Module):
    """Return normalized OpenCLIP image and text representations."""

    def __init__(self):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "OpenCLIP is not installed. Install the open_clip_torch package "
                "in the ATML environment before building this backbone."
            )

        OPENCLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            cache_dir=str(OPENCLIP_CACHE_DIR),
        )
        self.model = freeze_model(clip_model)
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.feature_dimension = int(self.model.visual.output_dim)
        self.model_name = CLIP_MODEL_NAME

        preprocess_configuration = open_clip.get_model_preprocess_cfg(self.model)
        self.normalization = ImageNormalizer(
            preprocess_configuration["mean"],
            preprocess_configuration["std"],
        )

    def forward(self, images):
        image_features = self.model.encode_image(images)
        return torch_functional.normalize(image_features, dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts, device):
        """Tokenize prompts and return normalized text representations."""
        text_tokens = self.tokenizer(prompts).to(device)
        text_features = self.model.encode_text(text_tokens)
        return torch_functional.normalize(text_features, dim=-1)

    def get_logit_scale(self):
        """Return the learned scale used for zero-shot class similarities."""
        return self.model.logit_scale.exp().detach()


class LinearClassifierHead(nn.Module):
    """Map one frozen representation to the STL-10 class logits."""

    def __init__(self, feature_dimension, number_of_classes=len(STL10_CLASSES)):
        super().__init__()
        self.classifier = nn.Linear(feature_dimension, number_of_classes)

    def forward(self, features):
        return self.classifier(features)


def build_backbone(backbone_name, device):
    """Build one configured frozen backbone and move it to the device."""
    normalized_name = str(backbone_name).lower().replace("-", "_")
    if normalized_name in {"resnet", "resnet50"}:
        backbone = ResNet50Backbone()
    elif normalized_name in {"vit", "vit_b_16", "vitb16"}:
        backbone = ViTB16Backbone()
    elif normalized_name in {"clip", "openclip", "clip_vit_b_32", "vit_b_32"}:
        backbone = OpenCLIPBackbone()
    else:
        raise ValueError(f"Unknown backbone name: {backbone_name}")

    backbone = backbone.to(device)
    backbone.eval()
    total_parameters, trainable_parameters = count_parameters(backbone)
    print(f"Backbone: {backbone.model_name}")
    print(f"Feature dimension: {backbone.feature_dimension}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable backbone parameters: {trainable_parameters:,}")
    return backbone


def build_linear_head(backbone, device):
    """Build a trainable STL-10 linear head for a backbone's features."""
    classifier_head = LinearClassifierHead(backbone.feature_dimension)
    classifier_head = classifier_head.to(device)
    total_parameters, trainable_parameters = count_parameters(classifier_head)
    print(f"Linear-head parameters: {total_parameters:,}")
    print(f"Trainable linear-head parameters: {trainable_parameters:,}")
    return classifier_head
