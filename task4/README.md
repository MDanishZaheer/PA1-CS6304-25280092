# Task 4 - Open-Set Recognition

This folder implements the CIFAR-10/CIFAR-100 open-set recognition study with Vanilla, GCSC, PROSER, and the optional RPL extension.

## Data boundary

CIFAR-10 supplies all training, checkpoint selection, score design, Mahalanobis statistics, calibration, and rejection thresholds. The official CIFAR-10 training partition is split in a stratified manner into 90% training and 10% validation using seed 6304. The complete CIFAR-10 test partition is used for final closed-set accuracy.

CIFAR-100 is evaluation-only. Its archive is downloaded beside CIFAR-10 during the initial data-preparation block, but the download helper neither constructs a dataset nor returns images, labels, classes, or targets. `evaluation/evaluate_osr.py` dynamically imports and constructs the fixed near/far unknown groups only after the experiment lock has validated every checkpoint, known-output cache, score implementation, threshold, unknown-group protocol, and student-written hypothesis.

The fixed groups contain 800 images each:

- Near: bus, pickup truck, motorcycle, tractor, wolf, fox, leopard, camel.
- Far: bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe.

## Models and training

- `Vanilla`: random-initialized CIFAR ResNet-18 trained with cross-entropy.
- `GCSC`: the same initialization and optimization, adding only `RandAugment(num_ops=2, magnitude=9)`.
- `RPL`: random-initialized CIFAR ResNet-18 trained with reciprocal-distance classification and a learnable open-space margin.
- `PROSER`: initialized from the selected Vanilla checkpoint, adds five dummy classifiers, and fine-tunes for 50 epochs with classifier and layer2 manifold-mixup data placeholders.

The CIFAR ResNet-18 uses a 3x3 stride-1 first convolution and no initial max-pooling. Vanilla, GCSC, and RPL use SGD with learning rate 0.1 for 100 epochs. PROSER uses learning rate 0.001 for 50 epochs. All use momentum 0.9, weight decay 0.0005, cosine decay, batch size 128, and seed 6304. Checkpoints are selected only by CIFAR-10 validation accuracy.

RPL uses one 512-dimensional reciprocal point per known class, temperature 1, and open-space weight 0.1. Its implementation independently follows the classification and open-space equations from Chen et al., [*Learning Open Set Network with Discriminative Reciprocal Points*](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf) (ECCV 2020), together with the [authors' published RPL defaults](https://github.com/ma-xu/Open-Set-Recognition/tree/master/OSR/ARPL). Both MLS and the paper-style MSP score are calibrated using CIFAR-10 validation only.

The PROSER implementation follows the classifier-placeholder and data-placeholder equations in Zhou, Ye, and Zhan, *Learning Placeholders for Open-Set Recognition* (CVPR 2021), and independently adapts them to the required CIFAR ResNet-18. The primary references are the [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Learning_Placeholders_for_Open-Set_Recognition_CVPR_2021_paper.html) and [authors' reference implementation](https://github.com/LAMDA-CL/CVPR21-Proser).

## Scores and calibration

The frozen Vanilla outputs are scored with MSP, MLS, Energy, and shared-diagonal Mahalanobis distance. GCSC, RPL, and PROSER use MLS for the common trained-model comparison. RPL additionally uses its paper-style MSP score, while PROSER also uses its calibrated strongest-dummy probability minus maximum-known probability score.

Every score is expressed as unknownness, where larger means more novel. Each rejection threshold is the 95th percentile of that score on CIFAR-10 validation data. PROSER's dummy bias is also calibrated using only CIFAR-10 validation logits so 95% of known validation examples remain recognized as known.

## Running the task

Run `scripts/run_task4.ipynb` from inside the PA_1 project. The notebook is divided into:

1. Configuration and runtime checks.
2. CIFAR-10 split preparation and file-only CIFAR-100 download.
3. Vanilla, GCSC, RPL, and PROSER training.
4. CIFAR-10 output extraction and closed-set metrics.
5. Mahalanobis fitting and validation-only threshold calibration.
6. Student hypotheses and immutable experiment locking.
7. One explicit final CIFAR-100 evaluation.
8. Required score, trained-model, distribution, and failure evidence.

Replace all hypothesis placeholders before creating the lock. Do not use final CIFAR-100 results to change the models, scores, or thresholds.

## Outputs

- `results/checkpoints/`: selected model states.
- `results/histories/`: per-epoch losses, accuracy, and learning rates.
- `results/metrics/`: configurations, thresholds, locks, and final tables.
- `results/predictions/`: CIFAR-10 predictions and final unknown scores.
- `results/figures/`: training curves, score distributions, and failure images.
- `Data/task4_cache/`: large feature, logit, and Mahalanobis arrays excluded from Git.

The code produces reproducible evidence, while final report claims and interpretation remain the student's responsibility.
