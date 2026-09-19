# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Train Vanilla, GCSC, and PROSER using CIFAR-10 data only."""

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from task4.configs.config_loader import (
    create_output_directories,
    get_checkpoint_path,
    get_output_paths,
    save_config_snapshot,
)
from task4.methods.gcsc import calculate_gcsc_loss
from task4.methods.proser import calculate_proser_loss
from task4.methods.vanilla import calculate_vanilla_loss
from task4.models.resnet_cifar import build_model


EXPECTED_KNOWN_DATASET_KEYS = {
    "train",
    "train_evaluation",
    "validation",
    "test",
}


def set_random_seed(seed):
    """Set Python, NumPy, CPU, and CUDA random generators reproducibly."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_training_device(device=None):
    """Use a requested device or automatically select the available CUDA GPU."""
    selected_device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return selected_device


def seed_data_loader_worker(worker_id):
    """Seed NumPy and Python within every data-loading worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_known_only_datasets(datasets):
    """Reject any dataset mapping that could expose CIFAR-100 during training."""
    if set(datasets) != EXPECTED_KNOWN_DATASET_KEYS:
        raise ValueError(
            "Task 4 training requires only CIFAR-10 train, validation, and test views."
        )


def create_training_loader(dataset, configuration):
    """Create the reproducible shuffled CIFAR-10 optimization loader."""
    training = configuration["training"]
    generator = torch.Generator()
    generator.manual_seed(int(configuration["seed"]))
    number_of_workers = int(training["number_of_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=number_of_workers,
        pin_memory=bool(training["pin_memory"] and torch.cuda.is_available()),
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        persistent_workers=number_of_workers > 0,
    )


def create_validation_loader(dataset, configuration):
    """Create the deterministic CIFAR-10 validation loader."""
    training = configuration["training"]
    generator = torch.Generator()
    generator.manual_seed(int(configuration["seed"]) + 100)
    number_of_workers = int(training["number_of_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=number_of_workers,
        pin_memory=bool(training["pin_memory"] and torch.cuda.is_available()),
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        persistent_workers=number_of_workers > 0,
    )


@torch.inference_mode()
def evaluate_known_model(model, data_loader, device):
    """Calculate known-class accuracy and macro-F1 using only ten logits."""
    model.eval()
    labels = []
    predictions = []
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_labels = batch["label"].to(device, non_blocking=True)
        known_logits = model(images)
        labels.append(batch_labels.cpu())
        predictions.append(known_logits.argmax(dim=1).cpu())
    if not labels:
        raise ValueError("Cannot validate an empty CIFAR-10 loader.")
    labels = torch.cat(labels).numpy()
    predictions = torch.cat(predictions).numpy()
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "number_of_examples": int(len(labels)),
    }


def create_optimizer_and_scheduler(configuration, model):
    """Build the required SGD optimizer and cosine learning-rate schedule."""
    training = configuration["training"]
    optimizer = SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(training["maximum_epochs"]),
    )
    return optimizer, scheduler


def run_closed_set_training_epoch(
    configuration,
    model,
    optimizer,
    gradient_scaler,
    training_loader,
    device,
):
    """Run one Vanilla or GCSC optimization epoch."""
    method_name = configuration["method"]["name"]
    use_mixed_precision = bool(
        configuration["training"]["mixed_precision"] and device.type == "cuda"
    )
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    model.train()
    for batch in training_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
            known_logits = model(images)
            if method_name == "vanilla":
                loss = calculate_vanilla_loss(known_logits, labels)
            else:
                loss = calculate_gcsc_loss(known_logits, labels)
        gradient_scaler.scale(loss).backward()
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        total_loss += float(loss.detach().item()) * len(labels)
        total_correct += int((known_logits.argmax(dim=1) == labels).sum().item())
        total_examples += int(len(labels))
    return {
        "total_loss": total_loss / total_examples,
        "classification_loss": total_loss / total_examples,
        "training_accuracy": total_correct / total_examples,
    }


def run_proser_training_epoch(
    configuration,
    model,
    optimizer,
    gradient_scaler,
    training_loader,
    device,
):
    """Run one PROSER epoch with equal classifier and data-placeholder halves."""
    use_mixed_precision = bool(
        configuration["training"]["mixed_precision"] and device.type == "cuda"
    )
    metric_names = (
        "total_loss",
        "classifier_placeholder_loss",
        "classification_loss",
        "dummy_second_loss",
        "data_placeholder_loss",
        "mixing_weight",
        "number_of_mixup_pairs",
    )
    totals = {name: 0.0 for name in metric_names}
    total_correct = 0
    total_examples = 0
    number_of_batches = 0
    model.train()
    for batch in training_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
            loss_details = calculate_proser_loss(
                model,
                images,
                labels,
                configuration,
            )
        gradient_scaler.scale(loss_details["total_loss"]).backward()
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        for metric_name in metric_names:
            metric_value = loss_details[metric_name]
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().item()
            totals[metric_name] += float(metric_value)
        predictions = loss_details["known_logits"].argmax(dim=1)
        known_labels = loss_details["known_labels"]
        total_correct += int((predictions == known_labels).sum().item())
        total_examples += int(len(known_labels))
        number_of_batches += 1
    averaged = {
        name: total / number_of_batches
        for name, total in totals.items()
    }
    averaged["training_accuracy"] = total_correct / total_examples
    return averaged


def calculate_file_sha256(file_path):
    """Calculate the SHA-256 digest of one model checkpoint."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_proser_from_vanilla(model, configuration, device):
    """Load all Vanilla parameters while leaving five dummy weights random."""
    paths = get_output_paths(configuration)
    vanilla_checkpoint_file = paths["checkpoints_directory"] / "vanilla_best.pt"
    if not vanilla_checkpoint_file.exists():
        raise FileNotFoundError(
            "Train the Vanilla model before PROSER: "
            f"{vanilla_checkpoint_file.resolve()}"
        )
    checkpoint = torch.load(
        vanilla_checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    checkpoint_method = checkpoint.get("configuration", {}).get("method", {}).get(
        "name"
    )
    if checkpoint_method != "vanilla":
        raise ValueError("PROSER initialization checkpoint is not Vanilla.")
    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    expected_missing = {
        "dummy_classifier.weight",
        "dummy_classifier.bias",
    }
    if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
        raise ValueError("Vanilla and PROSER architectures are not checkpoint-compatible.")
    return vanilla_checkpoint_file, calculate_file_sha256(vanilla_checkpoint_file)


def save_training_checkpoint(
    checkpoint_file,
    configuration,
    epoch,
    model,
    optimizer,
    scheduler,
    validation_metrics,
    initialization_record,
):
    """Save a checkpoint selected exclusively by CIFAR-10 validation accuracy."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "configuration": configuration,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation": validation_metrics,
            "selection_metric": "validation_accuracy",
            "selection_value": float(validation_metrics["accuracy"]),
            "initialization": initialization_record,
            "cifar100_used_during_training": False,
        },
        checkpoint_path,
    )


def load_training_checkpoint(checkpoint_file, model, configuration, device):
    """Load a selected Task 4 checkpoint only when its configuration matches."""
    checkpoint = torch.load(
        Path(checkpoint_file),
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("configuration") != configuration:
        raise ValueError("The Task 4 checkpoint configuration does not match this run.")
    if checkpoint.get("selection_metric") != "validation_accuracy":
        raise ValueError("Task 4 checkpoints must use CIFAR-10 validation accuracy.")
    if checkpoint.get("cifar100_used_during_training") is not False:
        raise ValueError("A Task 4 checkpoint must certify no CIFAR-100 training use.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_task4_method(configuration, datasets, device=None):
    """Train one required Task 4 method without constructing CIFAR-100."""
    validate_known_only_datasets(datasets)
    paths = create_output_directories(configuration)
    selected_device = select_training_device(device)
    set_random_seed(int(configuration["seed"]))
    model = build_model(configuration, selected_device)
    method_name = configuration["method"]["name"]
    initialization_record = {"type": "random", "seed": int(configuration["seed"])}
    if method_name == "proser":
        vanilla_file, vanilla_digest = initialize_proser_from_vanilla(
            model,
            configuration,
            selected_device,
        )
        initialization_record = {
            "type": "selected_vanilla_checkpoint",
            "checkpoint_file": str(vanilla_file.resolve()),
            "checkpoint_sha256": vanilla_digest,
        }

    optimizer, scheduler = create_optimizer_and_scheduler(configuration, model)
    use_mixed_precision = bool(
        configuration["training"]["mixed_precision"]
        and selected_device.type == "cuda"
    )
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    training_loader = create_training_loader(datasets["train"], configuration)
    validation_loader = create_validation_loader(
        datasets["validation"],
        configuration,
    )
    run_name = configuration["method"]["run_name"]
    checkpoint_file = get_checkpoint_path(configuration)
    history_file = paths["histories_directory"] / f"{run_name}_history.csv"
    snapshot_file = paths["metrics_directory"] / f"{run_name}_config.json"
    save_config_snapshot(configuration, snapshot_file)

    history = []
    best_validation_accuracy = -float("inf")
    maximum_epochs = int(configuration["training"]["maximum_epochs"])
    print(f"Training Task 4 method: {method_name} on {selected_device}")
    for epoch_index in range(maximum_epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if method_name == "proser":
            training_metrics = run_proser_training_epoch(
                configuration,
                model,
                optimizer,
                gradient_scaler,
                training_loader,
                selected_device,
            )
        else:
            training_metrics = run_closed_set_training_epoch(
                configuration,
                model,
                optimizer,
                gradient_scaler,
                training_loader,
                selected_device,
            )
        validation_metrics = evaluate_known_model(
            model,
            validation_loader,
            selected_device,
        )
        history_row = {
            "epoch": epoch_index + 1,
            "learning_rate": learning_rate,
            **training_metrics,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(history_file, index=False)

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            save_training_checkpoint(
                checkpoint_file,
                configuration,
                epoch_index + 1,
                model,
                optimizer,
                scheduler,
                validation_metrics,
                initialization_record,
            )
            print(f"Saved improved checkpoint: {checkpoint_file.resolve()}")
        scheduler.step()
        print(
            f"Epoch {epoch_index + 1:03d}/{maximum_epochs}: "
            f"loss={training_metrics['total_loss']:.4f}, "
            f"validation accuracy={validation_metrics['accuracy']:.4f}"
        )

    checkpoint = load_training_checkpoint(
        checkpoint_file,
        model,
        configuration,
        selected_device,
    )
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "history": pd.DataFrame(history),
        "history_file": history_file,
        "device": selected_device,
    }


def train_or_load_task4_method(
    configuration,
    datasets,
    device=None,
    force_retrain=False,
):
    """Reuse an exact checkpoint or train the requested CIFAR-10 method."""
    validate_known_only_datasets(datasets)
    selected_device = select_training_device(device)
    paths = create_output_directories(configuration)
    checkpoint_file = get_checkpoint_path(configuration)
    run_name = configuration["method"]["run_name"]
    history_file = paths["histories_directory"] / f"{run_name}_history.csv"
    if force_retrain or not checkpoint_file.exists():
        return train_task4_method(configuration, datasets, selected_device)

    set_random_seed(int(configuration["seed"]))
    model = build_model(configuration, selected_device)
    checkpoint = load_training_checkpoint(
        checkpoint_file,
        model,
        configuration,
        selected_device,
    )
    history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
    print(f"Loaded Task 4 checkpoint: {checkpoint_file.resolve()}")
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "history": history,
        "history_file": history_file,
        "device": selected_device,
    }
