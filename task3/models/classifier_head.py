# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build the Task 2 checkpoint-compatible seven-class PACS classifier."""

from task2.models.classifier_head import ClassifierHead, PACSClassifier
from task3.models.backbone import build_backbone, count_model_parameters


def build_classifier(configuration, device):
    """Build the complete Task 3 classifier on the selected device."""
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


__all__ = ["ClassifierHead", "PACSClassifier", "build_classifier"]
