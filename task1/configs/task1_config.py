# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Stores shared paths and experiment settings for Task 1."""

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
STL10_DIR = DATA_DIR / "stl10"
TASK1_DIR = PROJECT_ROOT / "task1"
RESULTS_DIR = TASK1_DIR / "results"
SPLITS_DIR = RESULTS_DIR / "splits"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
HISTORIES_DIR = RESULTS_DIR / "histories"
CACHE_DIR = DATA_DIR / "task1_cache"
FEATURE_CACHE_DIR = CACHE_DIR / "features"
CUE_CONFLICT_DIR = CACHE_DIR / "cue_conflicts"
MODEL_CACHE_DIR = CACHE_DIR / "model_weights"
TORCH_HUB_DIR = MODEL_CACHE_DIR / "torch"
OPENCLIP_CACHE_DIR = MODEL_CACHE_DIR / "open_clip"
ADAIN_WEIGHTS_DIR = MODEL_CACHE_DIR / "adain"
ADAIN_VGG_WEIGHTS_FILE = ADAIN_WEIGHTS_DIR / "vgg_normalised.pth"
ADAIN_DECODER_WEIGHTS_FILE = ADAIN_WEIGHTS_DIR / "decoder.pth"
ADAIN_VGG_WEIGHTS_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/"
    "v0.0.0/vgg_normalised.pth"
)
ADAIN_DECODER_WEIGHTS_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/"
    "v0.0.0/decoder.pth"
)
CUE_CONFLICT_IMAGE_DIR = CUE_CONFLICT_DIR / "candidates"
CUE_CONFLICT_RESULTS_DIR = RESULTS_DIR / "cue_conflicts"
CUE_CONFLICT_REVIEW_FILE = CUE_CONFLICT_RESULTS_DIR / "visual_review.csv"
CUE_CONFLICT_SUMMARY_FILE = CUE_CONFLICT_RESULTS_DIR / "review_summary.json"

# Reproducibility and data settings fixed by the assignment
SEED = 6304
DATASET_NAME = "STL10"
STL10_CLASSES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.20
TEST_SUBSET_SIZE = 500
TEST_IMAGES_PER_CLASS = TEST_SUBSET_SIZE // len(STL10_CLASSES)
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# Pretrained backbone settings fixed by the assignment
RESNET_MODEL_NAME = "resnet50"
RESNET_WEIGHTS = "IMAGENET1K_V2"
VIT_MODEL_NAME = "vit_b_16"
VIT_WEIGHTS = "IMAGENET1K_V1"
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_PROMPT_TEMPLATE = "a photo of a {}."

# Linear classifier-head training settings
FEATURE_BATCH_SIZE = 32
LINEAR_HEAD_BATCH_SIZE = 64
LINEAR_HEAD_MAX_EPOCHS = 50
LINEAR_HEAD_LEARNING_RATE = 1e-3
LINEAR_HEAD_WEIGHT_DECAY = 1e-4
LINEAR_HEAD_EARLY_STOPPING_PATIENCE = 5
LINEAR_HEAD_SELECTION_METRIC = "validation_accuracy"
NUM_WORKERS = 0

# Color-intervention design choice
COLOR_INTERVENTIONS = ("grayscale", "hue_rotation")
HUE_ROTATION_DEGREES = 90.0

# Shape-texture cue-conflict design choices
CUE_CONFLICT_CLASS_PAIRS = (
    ("airplane", "cat"),
    ("bird", "car"),
    ("deer", "ship"),
    ("dog", "truck"),
    ("horse", "monkey"),
)
CUE_CONFLICT_STYLE_STRENGTH = 0.8
CUE_CONFLICT_CANDIDATES_PER_DIRECTION = 30
CUE_CONFLICT_MIN_VALID_IMAGES = 200
ADAIN_BATCH_SIZE = 8
CUE_CONFLICT_REJECTION_RULE = (
    "Reject an image if the content object is no longer visually identifiable, "
    "the stylization contains severe artifacts, or the output is not valid RGB."
)

# Spatial-intervention settings 
TRANSLATION_PIXELS = (0, 8, 16, 32)
TRANSLATION_DIRECTIONS = ("up", "down", "left", "right")
TRANSLATION_PADDING_MODE = "reflect"
PATCH_GRID_SIZE = 4

# Representation-analysis design choice
REPRESENTATION_INTERVENTIONS = (
    "grayscale",
    "cue_conflict",
    "translation",
    "patch_shuffle",
)
REPRESENTATION_METHOD = "tsne"
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30.0
TSNE_LEARNING_RATE = "auto"
TSNE_INITIALIZATION = "pca"
TSNE_MAX_ITERATIONS = 1000
TSNE_NORMALIZE_FEATURES = True
TSNE_METRIC = "euclidean"

# Machine-readable outputs used by later files
SPLIT_FILE = SPLITS_DIR / f"stl10_splits_seed{SEED}.json"
CONFIG_SNAPSHOT_FILE = RESULTS_DIR / "task1_config_snapshot.json"
OUTPUT_DIRECTORIES = (
    STL10_DIR,
    SPLITS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
    HISTORIES_DIR,
    FEATURE_CACHE_DIR,
    CUE_CONFLICT_DIR,
    TORCH_HUB_DIR,
    OPENCLIP_CACHE_DIR,
    ADAIN_WEIGHTS_DIR,
    CUE_CONFLICT_IMAGE_DIR,
    CUE_CONFLICT_RESULTS_DIR,
)
