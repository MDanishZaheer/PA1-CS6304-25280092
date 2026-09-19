# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Train DAN-DG and SAM through one source-only Task 3 pipeline."""

import math
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task2.training.train import (
    seed_data_loader_worker,
    select_training_device,
    set_random_seed,
)
from task3.configs.config_loader import (
    create_output_directories,
    get_method_checkpoint_path,
    get_output_paths,
    save_config_snapshot,
)
from task3.methods.dan_dg import calculate_dan_dg_loss
from task3.methods.erm import calculate_domain_balanced_erm_loss
from task3.methods.sam import build_sam_optimizer
from task3.models.backbone import freeze_batch_norm_statistics
from task3.models.classifier_head import build_classifier
from task3.selection.source_validation import (
    evaluate_source_validation,
    get_source_selection_value,
    load_erm_checkpoint,
)


def create_source_data_loader(
    dataset,
    batch_size,
    shuffle,
    drop_last,
    configuration,
    generator_seed,
):
    """Create one reproducible Task 3 source-domain data loader."""
    training = configuration["training"]
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    number_of_workers = int(training["number_of_workers"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=number_of_workers,
        pin_memory=bool(training["pin_memory"] and torch.cuda.is_available()),
        worker_init_fn=seed_data_loader_worker,
        generator=generator,
        persistent_workers=number_of_workers > 0,
    )


def create_task3_source_loaders(datasets, configuration):
    """Build training and validation loaders without a target-domain loader."""
    if set(datasets) != {"source_train", "source_validation"}:
        raise ValueError("Task 3 datasets must contain source domains only.")

    seed = int(configuration["seed"])
    training = configuration["training"]
    source_domains = configuration["dataset"]["source_domains"]
    expected_domains = set(source_domains)
    for split_name in ("source_train", "source_validation"):
        if set(datasets[split_name]) != expected_domains:
            raise ValueError(
                f"Task 3 {split_name} must contain exactly the three sources."
            )
    source_batch_size = int(training["source_batch_size_per_domain"])
    training_loaders = {}
    validation_loaders = {}

    for domain_index, domain in enumerate(source_domains):
        training_loaders[domain] = create_source_data_loader(
            datasets["source_train"][domain],
            batch_size=source_batch_size,
            shuffle=True,
            drop_last=True,
            configuration=configuration,
            generator_seed=seed + domain_index,
        )
        validation_loaders[domain] = create_source_data_loader(
            datasets["source_validation"][domain],
            batch_size=int(training["evaluation_batch_size"]),
            shuffle=False,
            drop_last=False,
            configuration=configuration,
            generator_seed=seed + 100 + domain_index,
        )
    return training_loaders, validation_loaders


def cycle_data_loader(data_loader):
    """Yield source batches forever by restarting a loader when exhausted."""
    while True:
        for batch in data_loader:
            yield batch


def collect_source_batches(
    source_iterators,
    source_domains,
    device,
    expected_batch_size,
):
    """Collect exactly eight labeled examples from each source domain."""
    images_by_domain = {}
    labels_by_domain = {}
    for domain in source_domains:
        batch = next(source_iterators[domain])
        if "label" not in batch:
            raise ValueError(f"Source domain {domain} did not provide labels.")
        if len(batch["label"]) != expected_batch_size:
            raise ValueError(
                f"Source domain {domain} did not provide {expected_batch_size} images."
            )
        images_by_domain[domain] = batch["image"].to(device, non_blocking=True)
        labels_by_domain[domain] = batch["label"].to(device, non_blocking=True)
    return images_by_domain, labels_by_domain


def forward_source_domains(model, images_by_domain):
    """Run one efficient forward pass and restore domain-separated outputs."""
    source_domains = list(images_by_domain)
    batch_sizes = [len(images_by_domain[domain]) for domain in source_domains]
    combined_images = torch.cat(
        [images_by_domain[domain] for domain in source_domains],
        dim=0,
    )
    combined_logits, combined_features = model(
        combined_images,
        return_features=True,
    )
    logits_by_domain = dict(
        zip(source_domains, torch.split(combined_logits, batch_sizes))
    )
    features_by_domain = dict(
        zip(source_domains, torch.split(combined_features, batch_sizes))
    )
    return logits_by_domain, features_by_domain


def run_dan_dg_training_epoch(
    configuration,
    model,
    optimizer,
    gradient_scaler,
    source_loaders,
    device,
):
    """Run one DAN-DG epoch using only pairwise source-domain alignment."""
    training = configuration["training"]
    method = configuration["method"]
    source_domains = configuration["dataset"]["source_domains"]
    batch_size = int(training["source_batch_size_per_domain"])
    steps_per_epoch = math.ceil(
        max(len(loader.dataset) for loader in source_loaders.values()) / batch_size
    )
    source_iterators = {
        domain: cycle_data_loader(loader)
        for domain, loader in source_loaders.items()
    }
    use_mixed_precision = bool(
        training["mixed_precision"] and device.type == "cuda"
    )
    totals = {"total_loss": 0.0, "classification_loss": 0.0, "mmd_loss": 0.0}

    model.train()
    freeze_batch_norm_statistics(model)
    for _ in range(steps_per_epoch):
        images_by_domain, labels_by_domain = collect_source_batches(
            source_iterators,
            source_domains,
            device,
            expected_batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
            logits_by_domain, features_by_domain = forward_source_domains(
                model,
                images_by_domain,
            )
            loss_details = calculate_dan_dg_loss(
                logits_by_domain,
                labels_by_domain,
                features_by_domain,
                mmd_weight=method["mmd_weight"],
                bandwidth_multipliers=method["kernel_bandwidth_multipliers"],
            )
        gradient_scaler.scale(loss_details["total_loss"]).backward()
        gradient_scaler.step(optimizer)
        gradient_scaler.update()

        for metric_name in totals:
            totals[metric_name] += float(
                loss_details[metric_name].detach().item()
            )
    return {
        metric_name: metric_total / steps_per_epoch
        for metric_name, metric_total in totals.items()
    }


def run_sam_training_epoch(
    configuration,
    model,
    optimizer,
    gradient_scaler,
    source_loaders,
    device,
):
    """Run one two-pass SAM epoch using the same balanced source batch twice."""
    training = configuration["training"]
    source_domains = configuration["dataset"]["source_domains"]
    batch_size = int(training["source_batch_size_per_domain"])
    steps_per_epoch = math.ceil(
        max(len(loader.dataset) for loader in source_loaders.values()) / batch_size
    )
    source_iterators = {
        domain: cycle_data_loader(loader)
        for domain, loader in source_loaders.items()
    }
    use_mixed_precision = bool(
        training["mixed_precision"] and device.type == "cuda"
    )
    totals = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "mmd_loss": 0.0,
        "sam_perturbed_loss": 0.0,
        "gradient_norm": 0.0,
    }

    model.train()
    freeze_batch_norm_statistics(model)
    for _ in range(steps_per_epoch):
        images_by_domain, labels_by_domain = collect_source_batches(
            source_iterators,
            source_domains,
            device,
            expected_batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
            first_logits, _ = forward_source_domains(model, images_by_domain)
            first_loss, _ = calculate_domain_balanced_erm_loss(
                first_logits,
                labels_by_domain,
            )
        gradient_scale = float(gradient_scaler.get_scale())
        gradient_scaler.scale(first_loss).backward()
        scaled_gradient_norm = optimizer.first_step(zero_grad=True)

        try:
            with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                second_logits, _ = forward_source_domains(model, images_by_domain)
                second_loss, _ = calculate_domain_balanced_erm_loss(
                    second_logits,
                    labels_by_domain,
                )
            gradient_scaler.scale(second_loss).backward()
        finally:
            optimizer.restore_parameters()

        gradient_scaler.step(optimizer.base_optimizer)
        gradient_scaler.update()
        totals["total_loss"] += float(second_loss.detach().item())
        totals["classification_loss"] += float(first_loss.detach().item())
        totals["sam_perturbed_loss"] += float(second_loss.detach().item())
        totals["gradient_norm"] += float(
            scaled_gradient_norm.detach().item() / gradient_scale
        )

    return {
        metric_name: metric_total / steps_per_epoch
        for metric_name, metric_total in totals.items()
    }


def create_method_optimizer(configuration, model):
    """Build AdamW directly for DAN-DG or wrapped by SAM for SAM."""
    method_name = configuration["method"]["name"]
    training = configuration["training"]
    if method_name == "dan_dg":
        return AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    if method_name == "sam":
        return build_sam_optimizer(configuration, model)
    raise ValueError("Task 3 ERM is load-only and must not create an optimizer.")


def save_task3_checkpoint(
    checkpoint_file,
    configuration,
    epoch,
    model,
    optimizer,
    validation_metrics,
):
    """Save a best Task 3 checkpoint selected only from source validation."""
    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_state = optimizer.state_dict()
    torch.save(
        {
            "configuration": configuration,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "source_validation": validation_metrics,
            "selection_metric": "mean_source_validation_macro_f1",
            "selection_value": get_source_selection_value(validation_metrics),
            "sketch_used_during_training": False,
        },
        checkpoint_path,
    )


def load_task3_checkpoint(checkpoint_file, model, configuration, device):
    """Load a Task 3 checkpoint only when its configuration matches exactly."""
    checkpoint = torch.load(
        Path(checkpoint_file),
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("configuration") != configuration:
        raise ValueError("The Task 3 checkpoint configuration does not match this run.")
    if checkpoint.get("sketch_used_during_training") is not False:
        raise ValueError("A Task 3 checkpoint must certify that Sketch was not used.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _create_history_row(epoch, training_metrics, validation_metrics):
    """Flatten training and source-validation measurements for one epoch."""
    row = {"epoch": int(epoch), **training_metrics}
    for domain, metrics in validation_metrics["domains"].items():
        row[f"{domain}_validation_accuracy"] = metrics["accuracy"]
        row[f"{domain}_validation_macro_f1"] = metrics["macro_f1"]
    row["mean_source_validation_accuracy"] = validation_metrics["mean_accuracy"]
    row["mean_source_validation_macro_f1"] = validation_metrics["mean_macro_f1"]
    row["worst_source_validation_accuracy"] = validation_metrics["worst_accuracy"]
    row["worst_source_validation_macro_f1"] = validation_metrics["worst_macro_f1"]
    return row


def load_task2_erm_baseline(configuration, device=None):
    """Build and load the unchanged Task 2 Source-only model for Task 3."""
    selected_device = select_training_device(device)
    set_random_seed(int(configuration["seed"]))
    model = build_classifier(configuration, selected_device)
    checkpoint_file = get_method_checkpoint_path(configuration)
    checkpoint = load_erm_checkpoint(checkpoint_file, model, selected_device)
    history_file = get_output_paths(configuration)["task2_erm_history"]
    history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
    print(f"Loaded unchanged Task 2 ERM checkpoint: {checkpoint_file.resolve()}")
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "history": history,
        "history_file": history_file,
        "device": selected_device,
        "reused_from_task2": True,
    }


def train_task3_method(configuration, datasets, device=None):
    """Train DAN-DG or SAM and select a checkpoint without Sketch."""
    if configuration["method"]["name"] == "erm":
        raise ValueError("Task 3 ERM must be loaded from Task 2, not retrained.")
    output_paths = create_output_directories(configuration)
    selected_device = select_training_device(device)
    set_random_seed(int(configuration["seed"]))
    model = build_classifier(configuration, selected_device)
    optimizer = create_method_optimizer(configuration, model)
    use_mixed_precision = bool(
        configuration["training"]["mixed_precision"]
        and selected_device.type == "cuda"
    )
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    training_loaders, validation_loaders = create_task3_source_loaders(
        datasets,
        configuration,
    )

    run_name = configuration["method"]["run_name"]
    checkpoint_file = get_method_checkpoint_path(configuration)
    history_file = output_paths["histories_directory"] / f"{run_name}_history.csv"
    snapshot_file = output_paths["metrics_directory"] / f"{run_name}_config.json"
    save_config_snapshot(configuration, snapshot_file)

    history = []
    best_macro_f1 = -float("inf")
    epochs_without_improvement = 0
    maximum_epochs = int(configuration["training"]["maximum_epochs"])
    patience = int(configuration["training"]["early_stopping_patience"])
    method_name = configuration["method"]["name"]
    print(f"Training Task 3 method: {method_name}")
    print(f"Device: {selected_device}")

    for epoch_index in range(maximum_epochs):
        if method_name == "dan_dg":
            training_metrics = run_dan_dg_training_epoch(
                configuration,
                model,
                optimizer,
                gradient_scaler,
                training_loaders,
                selected_device,
            )
        else:
            training_metrics = run_sam_training_epoch(
                configuration,
                model,
                optimizer,
                gradient_scaler,
                training_loaders,
                selected_device,
            )
        validation_metrics = evaluate_source_validation(
            model,
            validation_loaders,
            selected_device,
        )
        history.append(
            _create_history_row(epoch_index + 1, training_metrics, validation_metrics)
        )
        pd.DataFrame(history).to_csv(history_file, index=False)

        current_macro_f1 = get_source_selection_value(validation_metrics)
        print(
            f"Epoch {epoch_index + 1:02d}: "
            f"loss={training_metrics['total_loss']:.4f}, "
            f"mean source macro-F1={current_macro_f1:.4f}"
        )
        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            epochs_without_improvement = 0
            save_task3_checkpoint(
                checkpoint_file,
                configuration,
                epoch_index + 1,
                model,
                optimizer,
                validation_metrics,
            )
            print(f"Saved improved checkpoint: {checkpoint_file.resolve()}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch_index + 1} epochs.")
                break

    best_checkpoint = load_task3_checkpoint(
        checkpoint_file,
        model,
        configuration,
        selected_device,
    )
    return {
        "model": model,
        "checkpoint": best_checkpoint,
        "checkpoint_file": checkpoint_file,
        "history": pd.DataFrame(history),
        "history_file": history_file,
        "device": selected_device,
        "reused_from_task2": False,
    }


def train_or_load_task3_method(
    configuration,
    datasets,
    device=None,
    force_retrain=False,
):
    """Load ERM, reuse a Task 3 checkpoint, or train a missing method."""
    if configuration["method"]["name"] == "erm":
        return load_task2_erm_baseline(configuration, device=device)

    output_paths = create_output_directories(configuration)
    selected_device = select_training_device(device)
    checkpoint_file = get_method_checkpoint_path(configuration)
    history_file = (
        output_paths["histories_directory"]
        / f"{configuration['method']['run_name']}_history.csv"
    )
    if force_retrain or not checkpoint_file.exists():
        return train_task3_method(configuration, datasets, selected_device)

    set_random_seed(int(configuration["seed"]))
    model = build_classifier(configuration, selected_device)
    checkpoint = load_task3_checkpoint(
        checkpoint_file,
        model,
        configuration,
        selected_device,
    )
    history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
    print(f"Loaded Task 3 checkpoint: {checkpoint_file.resolve()}")
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_file": checkpoint_file,
        "history": history,
        "history_file": history_file,
        "device": selected_device,
        "reused_from_task2": False,
    }
