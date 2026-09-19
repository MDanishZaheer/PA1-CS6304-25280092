# Data

This directory stores datasets used by the assignment.

- `stl10/` is the download location for the raw STL-10 dataset.
- `pacs/` is the extraction location for the PACS dataset used by Tasks 2 and 3.
- `task2_cache/downloads/` stores the PACS archive downloaded by the shared loader.
- `task2_cache/features/` is available for reusable Task 2 feature caches.
- `task2_cache/model_weights/` stores pretrained ResNet-18 weights downloaded by PyTorch.
- `task3_cache/features/` is available for source-only diagnostic feature caches.
- `cifar10/` stores the CIFAR-10 known-class dataset used by Task 4.
- `cifar100/` stores CIFAR-100, which Task 4 loads only after final locking.
- `task4_cache/features/` stores deterministic penultimate feature arrays.
- `task4_cache/logits/` stores known and final unknown model-output arrays.
- `task4_cache/mahalanobis/` stores Vanilla class means and covariance values.
- Generated images and large dataset files should remain outside version control.
- Reproducible split indices and small metadata files belong with the relevant task code.
