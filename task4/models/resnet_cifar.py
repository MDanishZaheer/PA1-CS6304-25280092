# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Define a random-initialized ResNet-18 adapted to 32x32 CIFAR images."""

import torch
from torch import nn
from torchvision.models import resnet18

from task4.methods.rpl import ReciprocalPointHead


class CIFARResNet18(nn.Module):
    """Expose known logits, dummy logits, features, and the layer2 mix point."""

    def __init__(
        self,
        number_of_known_classes=10,
        number_of_dummy_classes=0,
        number_of_reciprocal_points=0,
        reciprocal_temperature=1.0,
        margin_initial_value=0.0,
    ):
        super().__init__()
        self.number_of_known_classes = int(number_of_known_classes)
        self.number_of_dummy_classes = int(number_of_dummy_classes)
        self.network = resnet18(
            weights=None,
            num_classes=self.number_of_known_classes,
        )
        self.network.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.network.conv1.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        self.network.maxpool = nn.Identity()
        self.feature_dimension = int(self.network.fc.in_features)
        self.reciprocal_head = None
        if int(number_of_reciprocal_points) > 0:
            self.network.fc = nn.Identity()
            self.reciprocal_head = ReciprocalPointHead(
                number_of_classes=self.number_of_known_classes,
                feature_dimension=self.feature_dimension,
                number_of_reciprocal_points=number_of_reciprocal_points,
                temperature=reciprocal_temperature,
                margin_initial_value=margin_initial_value,
            )
        if self.number_of_dummy_classes > 0:
            self.dummy_classifier = nn.Linear(
                self.feature_dimension,
                self.number_of_dummy_classes,
            )
        else:
            self.dummy_classifier = None

    def forward_to_layer2(self, images):
        """Run the CIFAR stem, layer1, and layer2 for manifold mixup."""
        features = self.network.conv1(images)
        features = self.network.bn1(features)
        features = self.network.relu(features)
        features = self.network.maxpool(features)
        features = self.network.layer1(features)
        return self.network.layer2(features)

    def forward_from_layer2(self, layer2_features):
        """Run layer3, layer4, and global pooling after the mix location."""
        features = self.network.layer3(layer2_features)
        features = self.network.layer4(features)
        features = self.network.avgpool(features)
        return torch.flatten(features, 1)

    def forward_features(self, images):
        """Return the 512-dimensional penultimate CIFAR representation."""
        return self.forward_from_layer2(self.forward_to_layer2(images))

    def classify_features(self, features):
        """Apply known and optional dummy classifiers to feature vectors."""
        if self.reciprocal_head is None:
            known_logits = self.network.fc(features)
        else:
            known_logits = self.reciprocal_head(features)
        dummy_logits = None
        if self.dummy_classifier is not None:
            dummy_logits = self.dummy_classifier(features)
        return known_logits, dummy_logits

    def forward_with_dummy(self, images):
        """Return known logits, dummy logits, and shared features."""
        features = self.forward_features(images)
        known_logits, dummy_logits = self.classify_features(features)
        return known_logits, dummy_logits, features

    def forward(self, images, return_features=False):
        """Return known-class logits and optionally penultimate features."""
        known_logits, _, features = self.forward_with_dummy(images)
        if return_features:
            return known_logits, features
        return known_logits


def count_model_parameters(model):
    """Return the total and trainable parameter counts."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def build_model(configuration, device):
    """Build the configured CIFAR ResNet-18 on the selected device."""
    model_configuration = configuration["model"]
    method = configuration["method"]
    model = CIFARResNet18(
        number_of_known_classes=model_configuration["number_of_known_classes"],
        number_of_dummy_classes=method["number_of_dummy_classes"],
        number_of_reciprocal_points=method.get(
            "number_of_reciprocal_points",
            0,
        ),
        reciprocal_temperature=method.get("reciprocal_temperature", 1.0),
        margin_initial_value=method.get("margin_initial_value", 0.0),
    ).to(device)
    if model.feature_dimension != model_configuration["feature_dimension"]:
        raise ValueError("The configured feature dimension does not match ResNet-18.")
    total_parameters, trainable_parameters = count_model_parameters(model)
    print(
        f"CIFAR ResNet-18 parameters: "
        f"{trainable_parameters:,}/{total_parameters:,} trainable"
    )
    return model
