# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Train, save, and reuse linear heads on frozen backbone features."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task1.configs.task1_config import (
    CHECKPOINTS_DIR,
    HISTORIES_DIR,
    LINEAR_HEAD_BATCH_SIZE,
    LINEAR_HEAD_EARLY_STOPPING_PATIENCE,
    LINEAR_HEAD_LEARNING_RATE,
    LINEAR_HEAD_MAX_EPOCHS,
    LINEAR_HEAD_SELECTION_METRIC,
    LINEAR_HEAD_WEIGHT_DECAY,
    NUM_WORKERS,
    SEED,
    STL10_CLASSES,
)
from task1.models.backbones import LinearClassifierHead


def set_linear_head_seed(seed=SEED):
    """Set the random generators used during linear-head training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_training_device(device=None):
    """Use a requested device or automatically use CUDA when available."""
    selected_device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return selected_device


def validate_features_and_labels(
    features,
    labels,
    expected_feature_dimension=None,
):
    """Validate one feature matrix and its STL-10 labels."""
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)

    if feature_tensor.ndim != 2 or feature_tensor.shape[0] == 0:
        raise ValueError("Features must be a non-empty two-dimensional matrix.")
    if label_tensor.ndim != 1 or len(label_tensor) != len(feature_tensor):
        raise ValueError("Labels must be one-dimensional and match the features.")
    if not torch.isfinite(feature_tensor).all():
        raise ValueError("Features must contain only finite values.")
    if torch.any(label_tensor < 0) or torch.any(label_tensor >= len(STL10_CLASSES)):
        raise ValueError("A label is outside the STL-10 class range.")
    if (
        expected_feature_dimension is not None
        and feature_tensor.shape[1] != expected_feature_dimension
    ):
        raise ValueError("The feature dimension does not match the expected value.")
    return feature_tensor, label_tensor


def create_feature_loader(
    features,
    labels,
    batch_size=LINEAR_HEAD_BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    pin_memory=False,
):
    """Create a deterministic data loader for already extracted features."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    feature_tensor, label_tensor = validate_features_and_labels(features, labels)
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(feature_tensor, label_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        generator=data_generator,
    )


@torch.inference_mode()
def evaluate_linear_head(classifier_head, data_loader, device):
    """Calculate average loss and accuracy for one feature data loader."""
    classifier_head.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for feature_batch, label_batch in data_loader:
        feature_batch = feature_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)
        logits = classifier_head(feature_batch)
        total_loss += float(loss_function(logits, label_batch).item())
        total_correct += int((logits.argmax(dim=1) == label_batch).sum().item())
        total_examples += len(label_batch)

    if total_examples == 0:
        raise ValueError("Cannot evaluate a data loader containing no examples.")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "number_of_examples": total_examples,
    }


def make_safe_backbone_name(backbone_name):
    """Create a stable filename component from a backbone name."""
    safe_name = str(backbone_name).strip().lower().replace("-", "_").replace("/", "_")
    safe_name = "_".join(safe_name.split())
    if not safe_name:
        raise ValueError("Backbone name cannot be empty.")
    return safe_name


def get_linear_head_files(backbone_name):
    """Return the configured checkpoint and history paths for one backbone."""
    safe_name = make_safe_backbone_name(backbone_name)
    checkpoint_file = CHECKPOINTS_DIR / f"{safe_name}_linear_head_seed{SEED}.pt"
    history_file = HISTORIES_DIR / f"{safe_name}_linear_head_seed{SEED}.csv"
    return checkpoint_file, history_file


def save_linear_head_checkpoint(
    classifier_head,
    backbone_name,
    feature_dimension,
    best_epoch,
    best_validation_accuracy,
    checkpoint_file,
    maximum_epochs=LINEAR_HEAD_MAX_EPOCHS,
    early_stopping_patience=LINEAR_HEAD_EARLY_STOPPING_PATIENCE,
):
    """Save the best linear-head state and its required training settings."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "backbone_name": str(backbone_name),
        "feature_dimension": int(feature_dimension),
        "number_of_classes": len(STL10_CLASSES),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in classifier_head.state_dict().items()
        },
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_validation_accuracy),
        "selection_metric": LINEAR_HEAD_SELECTION_METRIC,
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": LINEAR_HEAD_LEARNING_RATE,
        "weight_decay": LINEAR_HEAD_WEIGHT_DECAY,
        "maximum_epochs": int(maximum_epochs),
        "early_stopping_patience": int(early_stopping_patience),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def train_linear_head(
    backbone_name,
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    device=None,
    checkpoint_file=None,
    history_file=None,
    max_epochs=LINEAR_HEAD_MAX_EPOCHS,
    patience=LINEAR_HEAD_EARLY_STOPPING_PATIENCE,
):
    """Train a linear head and restore the epoch with best validation accuracy."""
    if not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("Maximum epochs must be a positive integer.")
    if not isinstance(patience, int) or patience < 1:
        raise ValueError("Early-stopping patience must be a positive integer.")

    train_tensor, train_label_tensor = validate_features_and_labels(
        train_features,
        train_labels,
    )
    feature_dimension = int(train_tensor.shape[1])
    validation_tensor, validation_label_tensor = validate_features_and_labels(
        validation_features,
        validation_labels,
        expected_feature_dimension=feature_dimension,
    )
    if checkpoint_file is None or history_file is None:
        default_checkpoint, default_history = get_linear_head_files(backbone_name)
        checkpoint_file = checkpoint_file or default_checkpoint
        history_file = history_file or default_history

    selected_device = select_training_device(device)
    set_linear_head_seed(SEED)
    use_pinned_memory = selected_device.type == "cuda"
    train_loader = create_feature_loader(
        train_tensor,
        train_label_tensor,
        shuffle=True,
        pin_memory=use_pinned_memory,
    )
    train_evaluation_loader = create_feature_loader(
        train_tensor,
        train_label_tensor,
        shuffle=False,
        pin_memory=use_pinned_memory,
    )
    validation_loader = create_feature_loader(
        validation_tensor,
        validation_label_tensor,
        shuffle=False,
        pin_memory=use_pinned_memory,
    )

    classifier_head = LinearClassifierHead(feature_dimension).to(selected_device)
    optimizer = torch.optim.AdamW(
        classifier_head.parameters(),
        lr=LINEAR_HEAD_LEARNING_RATE,
        weight_decay=LINEAR_HEAD_WEIGHT_DECAY,
    )
    loss_function = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        classifier_head.train()
        for feature_batch, label_batch in train_loader:
            feature_batch = feature_batch.to(selected_device, non_blocking=True)
            label_batch = label_batch.to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier_head(feature_batch)
            loss = loss_function(logits, label_batch)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_linear_head(
            classifier_head,
            train_evaluation_loader,
            selected_device,
        )
        validation_metrics = evaluate_linear_head(
            classifier_head,
            validation_loader,
            selected_device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )

        validation_accuracy = validation_metrics["accuracy"]
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in classifier_head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"{backbone_name} epoch {epoch:02d}: "
            f"train accuracy={train_metrics['accuracy']:.4f}, "
            f"validation accuracy={validation_accuracy:.4f}"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    if best_state is None:
        raise RuntimeError("Linear-head training did not produce a valid checkpoint.")
    classifier_head.load_state_dict(best_state)
    classifier_head.eval()

    history_path = Path(history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_data = pd.DataFrame(history)
    history_data.to_csv(history_path, index=False)
    checkpoint = save_linear_head_checkpoint(
        classifier_head,
        backbone_name,
        feature_dimension,
        best_epoch,
        best_validation_accuracy,
        checkpoint_file,
        maximum_epochs=max_epochs,
        early_stopping_patience=patience,
    )
    print(
        f"Best {backbone_name} validation accuracy: "
        f"{best_validation_accuracy:.4f} at epoch {best_epoch}."
    )
    return classifier_head, history_data, checkpoint


def load_linear_head(
    backbone_name,
    feature_dimension,
    device=None,
    checkpoint_file=None,
):
    """Load and validate a previously selected linear-head checkpoint."""
    if checkpoint_file is None:
        checkpoint_file, _ = get_linear_head_files(backbone_name)
    checkpoint_path = Path(checkpoint_file)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    expected_settings = {
        "backbone_name": str(backbone_name),
        "feature_dimension": int(feature_dimension),
        "number_of_classes": len(STL10_CLASSES),
        "selection_metric": LINEAR_HEAD_SELECTION_METRIC,
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": LINEAR_HEAD_LEARNING_RATE,
        "weight_decay": LINEAR_HEAD_WEIGHT_DECAY,
        "maximum_epochs": LINEAR_HEAD_MAX_EPOCHS,
        "early_stopping_patience": LINEAR_HEAD_EARLY_STOPPING_PATIENCE,
    }
    for setting_name, expected_value in expected_settings.items():
        if checkpoint.get(setting_name) != expected_value:
            raise ValueError(
                f"The saved linear head has a different {setting_name} setting."
            )

    selected_device = select_training_device(device)
    classifier_head = LinearClassifierHead(feature_dimension).to(selected_device)
    classifier_head.load_state_dict(checkpoint["state_dict"])
    classifier_head.eval()
    return classifier_head, checkpoint


def train_or_load_linear_head(
    backbone_name,
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    device=None,
    overwrite=False,
):
    """Reuse a compatible head or train it using only train and validation data."""
    train_tensor, _ = validate_features_and_labels(train_features, train_labels)
    feature_dimension = int(train_tensor.shape[1])
    checkpoint_file, history_file = get_linear_head_files(backbone_name)

    if checkpoint_file.exists() and not overwrite:
        classifier_head, checkpoint = load_linear_head(
            backbone_name,
            feature_dimension,
            device=device,
            checkpoint_file=checkpoint_file,
        )
        if history_file.exists():
            history_data = pd.read_csv(history_file)
        else:
            history_data = pd.DataFrame()
        print(f"Loaded existing linear head: {checkpoint_file}")
        return classifier_head, history_data, checkpoint

    return train_linear_head(
        backbone_name,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        device=device,
        checkpoint_file=checkpoint_file,
        history_file=history_file,
    )


@torch.inference_mode()
def predict_linear_head(
    classifier_head,
    features,
    device=None,
    batch_size=LINEAR_HEAD_BATCH_SIZE,
):
    """Return ordered STL-10 logits for a matrix of frozen features."""
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    if feature_tensor.ndim != 2 or feature_tensor.shape[0] == 0:
        raise ValueError("Features must be a non-empty two-dimensional matrix.")
    if not torch.isfinite(feature_tensor).all():
        raise ValueError("Features must contain only finite values.")
    feature_dimension = classifier_head.classifier.in_features
    if feature_tensor.shape[1] != feature_dimension:
        raise ValueError("The features do not match the linear head dimension.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")

    selected_device = select_training_device(device)
    classifier_head = classifier_head.to(selected_device)
    classifier_head.eval()
    feature_loader = DataLoader(
        TensorDataset(feature_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=selected_device.type == "cuda",
    )
    logits = []
    for (feature_batch,) in feature_loader:
        feature_batch = feature_batch.to(selected_device, non_blocking=True)
        logits.append(classifier_head(feature_batch).cpu())
    return torch.cat(logits).numpy()
