# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run the common source-only, DAN, DANN, and CDAN training pipeline."""

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task2.configs.config_loader import (
    create_output_directories,
    get_output_paths,
    save_config_snapshot,
)
from task2.evaluation.metrics import evaluate_source_validation
from task2.methods.cdan import calculate_cdan_loss
from task2.methods.dan import calculate_dan_loss
from task2.methods.dann import calculate_dann_loss, calculate_grl_strength
from task2.methods.source_only import calculate_source_only_loss
from task2.models.backbone import freeze_batch_norm_statistics
from task2.models.classifier_head import build_classifier
from task2.models.domain_discriminator import build_domain_discriminator


def set_random_seed(seed):
    """Set Python, NumPy, CPU, and CUDA random generators reproducibly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_training_device(device=None):
    """Use a requested device or automatically select CUDA when available."""
    selected_device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return selected_device


def seed_data_loader_worker(worker_id):
    """Seed NumPy and Python inside each data-loading worker process."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(
    dataset,
    batch_size,
    shuffle,
    drop_last,
    configuration,
    generator_seed,
):
    """Create a reproducible data loader using the shared training settings."""
    training = configuration["training"]
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    number_of_workers = int(training["number_of_workers"])
    pin_memory = bool(training["pin_memory"] and torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        persistent_workers=number_of_workers > 0,
    )


def create_task2_loaders(datasets, configuration):
    """Create domain-balanced source, target, and validation data loaders."""
    seed = int(configuration["seed"])
    training = configuration["training"]
    source_domains = configuration["dataset"]["source_domains"]
    source_batch_size = int(training["source_batch_size_per_domain"])

    source_training_loaders = {}
    for domain_index, domain in enumerate(source_domains):
        source_training_loaders[domain] = create_data_loader(
            datasets["source_train"][domain],
            batch_size=source_batch_size,
            shuffle=True,
            drop_last=True,
            configuration=configuration,
            generator_seed=seed + domain_index,
        )

    validation_loaders = {}
    for domain_index, domain in enumerate(source_domains):
        validation_loaders[domain] = create_data_loader(
            datasets["source_validation"][domain],
            batch_size=int(training["evaluation_batch_size"]),
            shuffle=False,
            drop_last=False,
            configuration=configuration,
            generator_seed=seed + 100 + domain_index,
        )

    target_loader = None
    if configuration["method"]["name"] != "source_only":
        target_loader = create_data_loader(
            datasets["target_adaptation"],
            batch_size=int(training["target_batch_size"]),
            shuffle=True,
            drop_last=True,
            configuration=configuration,
            generator_seed=seed + 200,
        )
    return source_training_loaders, validation_loaders, target_loader


def cycle_data_loader(data_loader):
    """Yield batches forever by restarting a finite loader when exhausted."""
    while True:
        for batch in data_loader:
            yield batch


def combine_source_batches(source_iterators, source_domains, device, expected_size):
    """Join one equally sized labeled batch from every source domain."""
    image_batches = []
    label_batches = []
    for domain in source_domains:
        batch = next(source_iterators[domain])
        if "label" not in batch:
            raise ValueError(f"Source domain {domain} did not provide class labels.")
        if len(batch["label"]) != expected_size:
            raise ValueError(f"Source domain {domain} did not provide {expected_size} images.")
        image_batches.append(batch["image"].to(device, non_blocking=True))
        label_batches.append(batch["label"].to(device, non_blocking=True))
    return torch.cat(image_batches, dim=0), torch.cat(label_batches, dim=0)


def calculate_method_loss(
    configuration,
    model,
    discriminator,
    source_images,
    source_labels,
    target_images,
    progress,
):
    """Calculate the selected method's loss through a common interface."""
    method = configuration["method"]
    method_name = method["name"]
    source_logits, source_features = model(source_images, return_features=True)

    if method_name == "source_only":
        return calculate_source_only_loss(source_logits, source_labels), 0.0

    target_logits, target_features = model(target_images, return_features=True)
    if method_name == "dan":
        loss_details = calculate_dan_loss(
            source_logits,
            source_labels,
            source_features.float(),
            target_features.float(),
            mmd_weight=method["mmd_weight"],
            bandwidth_multipliers=method["kernel_bandwidth_multipliers"],
        )
        return loss_details, 0.0

    grl_strength = calculate_grl_strength(
        progress,
        maximum_strength=method["grl_maximum_strength"],
    )
    if method_name == "dann":
        loss_details = calculate_dann_loss(
            source_logits,
            source_labels,
            source_features,
            target_features,
            discriminator,
            grl_strength,
            domain_loss_weight=method["domain_loss_weight"],
        )
    elif method_name == "cdan":
        loss_details = calculate_cdan_loss(
            source_logits,
            source_labels,
            source_features,
            target_logits,
            target_features,
            discriminator,
            grl_strength,
            domain_loss_weight=method["domain_loss_weight"],
        )
    else:
        raise ValueError(f"Unknown Task 2 method: {method_name}")
    return loss_details, grl_strength


def run_training_epoch(
    configuration,
    model,
    discriminator,
    optimizer,
    gradient_scaler,
    source_loaders,
    target_loader,
    device,
    epoch_index,
):
    """Run one source epoch using balanced and cyclic domain loaders."""
    training = configuration["training"]
    source_domains = configuration["dataset"]["source_domains"]
    source_batch_size = int(training["source_batch_size_per_domain"])
    largest_source_size = max(len(loader.dataset) for loader in source_loaders.values())
    steps_per_epoch = math.ceil(largest_source_size / source_batch_size)
    total_planned_steps = int(training["maximum_epochs"]) * steps_per_epoch

    source_iterators = {
        domain: cycle_data_loader(source_loaders[domain])
        for domain in source_domains
    }
    target_iterator = cycle_data_loader(target_loader) if target_loader is not None else None

    model.train()
    if configuration["model"]["freeze_batch_norm_statistics"]:
        freeze_batch_norm_statistics(model)
    if discriminator is not None:
        discriminator.train()

    totals = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "alignment_loss": 0.0,
        "domain_loss": 0.0,
        "domain_accuracy": 0.0,
        "grl_strength": 0.0,
        "gradient_norm_before_clipping": 0.0,
    }
    use_mixed_precision = bool(
        training["mixed_precision"] and device.type == "cuda"
    )
    gradient_clip_norm = float(training["gradient_clip_norm"])
    trainable_parameters = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
        if parameter.requires_grad
    ]

    for step_index in range(steps_per_epoch):
        source_images, source_labels = combine_source_batches(
            source_iterators,
            source_domains,
            device,
            expected_size=source_batch_size,
        )
        target_images = None
        if target_iterator is not None:
            target_batch = next(target_iterator)
            if "label" in target_batch:
                raise ValueError("Target labels were exposed during Task 2 adaptation.")
            if len(target_batch["image"]) != int(training["target_batch_size"]):
                raise ValueError("The target adaptation batch must contain 24 images.")
            target_images = target_batch["image"].to(device, non_blocking=True)

        global_step = epoch_index * steps_per_epoch + step_index
        progress = global_step / max(total_planned_steps - 1, 1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=use_mixed_precision,
        ):
            loss_details, grl_strength = calculate_method_loss(
                configuration,
                model,
                discriminator,
                source_images,
                source_labels,
                target_images,
                progress,
            )

        gradient_scaler.scale(loss_details["total_loss"]).backward()
        gradient_scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        gradient_scaler.step(optimizer)
        gradient_scaler.update()

        for loss_name in (
            "total_loss",
            "classification_loss",
            "alignment_loss",
            "domain_loss",
        ):
            totals[loss_name] += float(loss_details[loss_name].detach().item())
        if "domain_accuracy" in loss_details:
            totals["domain_accuracy"] += float(
                loss_details["domain_accuracy"].detach().item()
            )
        totals["grl_strength"] += float(grl_strength)
        totals["gradient_norm_before_clipping"] += float(
            gradient_norm.detach().item()
        )

    return {
        metric_name: metric_total / steps_per_epoch
        for metric_name, metric_total in totals.items()
    }


def create_optimizer(configuration, model, discriminator=None):
    """Create the common AdamW optimizer for all trainable method parameters."""
    parameters = list(model.parameters())
    if discriminator is not None:
        parameters.extend(discriminator.parameters())
    training = configuration["training"]
    return AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def save_training_checkpoint(
    checkpoint_file,
    configuration,
    epoch,
    model,
    discriminator,
    optimizer,
    validation_metrics,
):
    """Save the best source-selected model and all reproducibility information."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "configuration": configuration,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": (
            discriminator.state_dict() if discriminator is not None else None
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_validation": validation_metrics,
        "selection_metric": "mean_source_validation_macro_f1",
        "selection_value": validation_metrics["mean_macro_f1"],
    }
    torch.save(checkpoint, checkpoint_path)


def load_training_checkpoint(checkpoint_file, model, discriminator=None, device="cpu"):
    """Load a saved Task 2 model and optional domain discriminator."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if discriminator is not None and checkpoint["discriminator_state_dict"] is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    return checkpoint


def _history_row(epoch, training_metrics, validation_metrics):
    """Flatten one epoch's training and per-domain validation measurements."""
    row = {"epoch": int(epoch), **training_metrics}
    for domain, metrics in validation_metrics["domains"].items():
        row[f"{domain}_validation_accuracy"] = metrics["accuracy"]
        row[f"{domain}_validation_macro_f1"] = metrics["macro_f1"]
    row["mean_source_validation_accuracy"] = validation_metrics["mean_accuracy"]
    row["mean_source_validation_macro_f1"] = validation_metrics["mean_macro_f1"]
    return row


def train_task2_method(configuration, datasets, device=None):
    """Train one method and select its checkpoint without target labels."""
    create_output_directories(configuration)
    output_paths = get_output_paths(configuration)
    selected_device = select_training_device(device)
    set_random_seed(int(configuration["seed"]))

    model = build_classifier(configuration, selected_device)
    discriminator = build_domain_discriminator(configuration, selected_device)
    optimizer = create_optimizer(configuration, model, discriminator)
    use_mixed_precision = bool(
        configuration["training"]["mixed_precision"]
        and selected_device.type == "cuda"
    )
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    source_loaders, validation_loaders, target_loader = create_task2_loaders(
        datasets,
        configuration,
    )

    run_name = configuration["method"]["run_name"]
    checkpoint_file = output_paths["checkpoints_directory"] / f"{run_name}_best.pt"
    history_file = output_paths["histories_directory"] / f"{run_name}_history.csv"
    snapshot_file = output_paths["metrics_directory"] / f"{run_name}_config.json"
    save_config_snapshot(configuration, snapshot_file)

    history = []
    best_macro_f1 = -float("inf")
    epochs_without_improvement = 0
    maximum_epochs = int(configuration["training"]["maximum_epochs"])
    patience = int(configuration["training"]["early_stopping_patience"])

    print(f"Training method: {configuration['method']['name']}")
    print(f"Device: {selected_device}")
    for epoch_index in range(maximum_epochs):
        training_metrics = run_training_epoch(
            configuration,
            model,
            discriminator,
            optimizer,
            gradient_scaler,
            source_loaders,
            target_loader,
            selected_device,
            epoch_index,
        )
        validation_metrics = evaluate_source_validation(
            model,
            validation_loaders,
            selected_device,
        )
        history.append(
            _history_row(epoch_index + 1, training_metrics, validation_metrics)
        )
        pd.DataFrame(history).to_csv(history_file, index=False)

        current_macro_f1 = validation_metrics["mean_macro_f1"]
        print(
            f"Epoch {epoch_index + 1:02d}: "
            f"loss={training_metrics['total_loss']:.4f}, "
            f"mean source macro-F1={current_macro_f1:.4f}"
        )
        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            epochs_without_improvement = 0
            save_training_checkpoint(
                checkpoint_file,
                configuration,
                epoch_index + 1,
                model,
                discriminator,
                optimizer,
                validation_metrics,
            )
            print(f"Saved improved checkpoint: {checkpoint_file.resolve()}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch_index + 1} epochs.")
                break

    best_checkpoint = load_training_checkpoint(
        checkpoint_file,
        model,
        discriminator=discriminator,
        device=selected_device,
    )
    return {
        "model": model,
        "discriminator": discriminator,
        "checkpoint": best_checkpoint,
        "checkpoint_file": checkpoint_file,
        "history": pd.DataFrame(history),
        "history_file": history_file,
        "device": selected_device,
    }


def train_or_load_task2_method(
    configuration,
    datasets,
    device=None,
    force_retrain=False,
):
    """Reuse a selected checkpoint or train the configured method from scratch."""
    output_paths = create_output_directories(configuration)
    selected_device = select_training_device(device)
    run_name = configuration["method"]["run_name"]
    checkpoint_file = output_paths["checkpoints_directory"] / f"{run_name}_best.pt"
    history_file = output_paths["histories_directory"] / f"{run_name}_history.csv"

    if force_retrain or not checkpoint_file.exists():
        return train_task2_method(configuration, datasets, device=selected_device)

    set_random_seed(int(configuration["seed"]))
    model = build_classifier(configuration, selected_device)
    discriminator = build_domain_discriminator(configuration, selected_device)
    checkpoint = load_training_checkpoint(
        checkpoint_file,
        model,
        discriminator=discriminator,
        device=selected_device,
    )
    if checkpoint.get("configuration") != configuration:
        raise ValueError(
            "The existing checkpoint configuration does not match this run. "
            "Use a unique run name or set force_retrain=True."
        )
    history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
    print(f"Loaded checkpoint: {checkpoint_file.resolve()}")
    return {
        "model": model,
        "discriminator": discriminator,
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "history": history,
        "history_file": history_file,
        "device": selected_device,
    }
