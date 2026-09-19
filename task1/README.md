# Task 1 - Inductive Biases and Feature Representations

The Python modules contain reusable experiment logic. The notebook in `scripts/` coordinates the complete workflow, displays verification checks, and compiles the saved results.

## Structure

```text
task1/
  configs/                    Experiment settings
  data/
    make_subset.py            Deterministic STL-10 splits and test subset
    make_cue_conflicts.py     Shape-texture conflict generation
    transforms.py             Color, translation, and patch interventions
  models/
    backbones.py              Frozen ResNet, ViT, and CLIP wrappers
    linear_probe.py           Linear-head training, selection, and reuse
  analysis/
    evaluate_bias.py          Prediction metrics and bias evaluation
    feature_similarity.py     Clean-transformed cosine stability
    representation.py        t-SNE or UMAP visualization
  scripts/
    run_task1.ipynb           Experiment runner and result compilation
  licenses/                   Third-party license notices
  results/                    Saved metrics, predictions, tables, and figures
```

Raw STL-10 files are stored in `../Data/stl10/`. The notebook should import and call the reusable modules rather than duplicate their implementations.

## Running Task 1

Open `scripts/run_task1.ipynb` with the `ATML` kernel and run its cells from top to bottom. Replace the four hypothesis placeholders before interpreting any results. The cue-conflict section deliberately pauses after generating candidates: complete the visual-only decisions in `results/cue_conflicts/visual_review.csv`, then rerun that cell to continue evaluation. Cached features and compatible linear-head checkpoints are reused automatically.

## Third-party attribution

`data/make_cue_conflicts.py` adapts the network architecture and uses the public pretrained weights from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN), copyright (c) 2018 Naoto Inoue. It is used under the MIT License; the license notice is preserved in `licenses/pytorch_adain_LICENSE.txt`.
