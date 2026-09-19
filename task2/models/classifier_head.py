# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Define the seven-class PACS head and complete classification model."""

from torch import nn

from task2.models.backbone import build_backbone, count_model_parameters


class ClassifierHead(nn.Module):
    """Map a 512-dimensional ResNet feature to seven PACS class logits."""

    def __init__(self, feature_dimension=512, number_of_classes=7):
        super().__init__()
        self.linear = nn.Linear(feature_dimension, number_of_classes)

    def forward(self, features):
        return self.linear(features)


class PACSClassifier(nn.Module):
    """Combine the trainable ResNet-18 backbone and linear classifier head."""

    def __init__(self, backbone, classifier_head):
        super().__init__()
        self.backbone = backbone
        self.classifier_head = classifier_head

    def forward(self, images, return_features=False):
        features = self.backbone(images)
        logits = self.classifier_head(features)
        if return_features:
            return logits, features
        return logits


def build_classifier(configuration, device):
    """Build the complete Task 2 classifier on the selected device."""
    model_configuration = configuration["model"]
    backbone = build_backbone(configuration, device)
    classifier_head = ClassifierHead(
        feature_dimension=model_configuration["feature_dimension"],
        number_of_classes=model_configuration["number_of_classes"],
    )
    model = PACSClassifier(backbone, classifier_head).to(device)
    total_parameters, trainable_parameters = count_model_parameters(model)
    print(f"Complete classifier parameters: {trainable_parameters:,}/{total_parameters:,}")
    return model
