# Task 2 - Unsupervised Domain Adaptation

This folder implements the PACS Source-only, DAN, DANN, and CDAN experiments through one shared training and evaluation pipeline.

## Protocol

- Labeled source domains: `photo`, `art_painting`, and `cartoon`.
- Unlabeled adaptation target: `sketch`.
- Each source domain uses a stratified 80/20 train/validation split with seed `6304`.
- The reusable PACS loader and split protocol are stored in `shared_task_2and3/`.
- Checkpoints are selected only with mean macro-F1 across the three source validation domains.
- Every update contains 8 images from each source domain. Adaptation methods additionally receive 24 unlabeled Sketch images.
- The complete ResNet-18 is fine-tuned while its ImageNet BatchNorm running means and variances remain frozen.
- Every method uses full-precision training and the same global gradient-clipping norm of 5 for numerical stability.
- Sketch labels are loaded only after the experiment-lock file has fixed all configurations and checkpoint hashes.

PACS is downloaded automatically from the public [`flwrlabs/pacs`](https://huggingface.co/datasets/flwrlabs/pacs) Hugging Face dataset. Images are stored below `Data/pacs/`, and the validated download manifest is saved at `Data/task2_cache/downloads/pacs_huggingface_manifest.json`.

## Configuration files

`configs/base.yaml` contains settings shared by every method. Each method YAML contains only its objective-specific values. `configs/config_loader.py` merges and validates them before use.

## Recommended execution

Run `scripts/run_task2.ipynb` from the project or notebook directory. The notebook is divided into short stages for:

1. Runtime and configuration checks.
2. PACS download, validation, and saved source splits.
3. Source-only, DAN, DANN, and CDAN training.
4. The DAN alignment-strength study at weights 0.1, 1, and 10.
5. Experiment locking before target-label access.
6. Final metrics, domain separability, per-class analysis, confusions, and plots.

Fill in the hypothesis placeholders before displaying any final target-label results. Set the notebook's final unlock variable to `True` only after all checkpoints are trained and the lock file is created.

## Saved outputs

- `results/checkpoints/`: source-selected model checkpoints. `source_only_best.pt` is reused unchanged by Task 3.
- `results/histories/`: per-epoch classification, alignment, domain, and validation measurements.
- `results/metrics/`: configuration snapshots, the experiment lock, final summary, and class-level tables.
- `results/predictions/`: final Sketch predictions and selected failure-case identifiers.
- `results/figures/`: training curves and target confusion matrices.

The notebook and code prepare evidence for the assignment, but the final report interpretation and prose must be written by the student.
