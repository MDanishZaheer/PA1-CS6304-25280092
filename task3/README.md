# Task 3 - Domain Generalization

This folder implements the shared ERM baseline, target-free DAN-DG, and non-adaptive SAM on the PACS source domains.

## Required prerequisites

Run the Task 2 data-preparation and Source-only training stages first. Task 3 requires:

- `Data/pacs/` containing PACS.
- `shared_task_2and3/splits/pacs_sketch_seed6304.json` containing the fixed source splits.
- `task2/results/checkpoints/source_only_best.pt` as the unchanged ERM baseline.

Task 3 does not download PACS, regenerate the protocol, or retrain ERM. Its source loader validates and constructs only Photo, Art Painting, and Cartoon. Sketch is loaded exclusively by `evaluation/evaluate_sketch.py` after the experiment lock has been created.

## Methods

- `ERM`: loads the Task 2 Source-only checkpoint unchanged.
- `DAN-DG`: averages the exact Task 2 MMD implementation across the three unordered source-domain pairs. The main weight is `lambda_DG = 1`.
- `SAM`: performs normalized non-adaptive parameter ascent followed by an AdamW update using `rho = 0.05` for the main comparison.

Every trainable method uses the same ResNet-18 initialization, transforms, balanced 8+8+8 source batch, optimizer settings, BatchNorm policy, epoch budget, source-only checkpoint criterion, and seed as Task 2.

## Controlled study

The notebook uses the bounded SAM study `rho = {0.01, 0.05, 0.1}`. The main SAM run supplies the middle setting, so only the 0.01 and 0.1 variants require extra training.

## Recommended execution

Run `scripts/run_task3.ipynb` after Task 2. The notebook separates:

1. Configuration and source-only protocol validation.
2. Reuse of Task 2 ERM.
3. DAN-DG and SAM training.
4. Source-only mean/worst metrics, source-domain separability, and sharpness.
5. Controlled SAM training.
6. Configuration and checkpoint locking.
7. One explicit final Sketch-label evaluation.
8. Class-level evidence and the Task 2 DAN versus Task 3 DAN-DG comparison.

Replace the hypothesis placeholders before unlocking Sketch. Final target results are for analysis only and must not be used to revise Task 3 settings.

## Saved outputs

- `results/checkpoints/`: source-selected DAN-DG and SAM checkpoints.
- `results/histories/`: classification, MMD, SAM, and source-validation histories.
- `results/metrics/`: locks, summaries, diagnostics, class changes, and cross-task comparisons.
- `results/predictions/`: final Sketch predictions and selected failure identifiers.
- `results/figures/`: training curves and Sketch confusion matrices.

The code prepares the required evidence, but the student must write the final report interpretation and prose.
