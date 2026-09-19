# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Apply color, translation, and patch-shuffle interventions."""

import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import functional as transform_functional

from task1.configs.task1_config import (
    HUE_ROTATION_DEGREES,
    IMAGE_SIZE,
    PATCH_GRID_SIZE,
    SEED,
    TRANSLATION_DIRECTIONS,
    TRANSLATION_PADDING_MODE,
    TRANSLATION_PIXELS,
)


def prepare_common_image(image, image_size=IMAGE_SIZE):
    """Convert an input image to the common square RGB representation."""
    if isinstance(image, Image.Image):
        common_image = image.convert("RGB")
    else:
        image_array = np.asarray(image)
        if image_array.dtype != np.uint8:
            raise ValueError("NumPy images must use unsigned 8-bit pixel values.")
        common_image = Image.fromarray(image_array).convert("RGB")

    expected_size = (image_size, image_size)
    if common_image.size != expected_size:
        common_image = common_image.resize(expected_size, Image.Resampling.BICUBIC)
    return common_image


def apply_grayscale(image):
    """Remove color while keeping a three-channel RGB output."""
    common_image = prepare_common_image(image)
    return ImageOps.grayscale(common_image).convert("RGB")


def apply_hue_rotation(image, degrees=HUE_ROTATION_DEGREES):
    """Rotate hue by a fixed number of degrees without changing geometry."""
    common_image = prepare_common_image(image)
    wrapped_degrees = float(degrees) % 360.0
    hue_factor = wrapped_degrees / 360.0
    if hue_factor > 0.5:
        hue_factor -= 1.0
    return transform_functional.adjust_hue(common_image, hue_factor)


def apply_translation(image, displacement, direction):
    """Translate an image using reflection padding and a shifted crop."""
    common_image = prepare_common_image(image)
    displacement = int(displacement)
    direction = str(direction).lower()

    if displacement not in TRANSLATION_PIXELS:
        raise ValueError(f"Displacement must be one of {TRANSLATION_PIXELS}.")
    if direction not in TRANSLATION_DIRECTIONS:
        raise ValueError(f"Direction must be one of {TRANSLATION_DIRECTIONS}.")
    if displacement == 0:
        return common_image.copy()

    image_array = np.asarray(common_image)
    padded_array = np.pad(
        image_array,
        ((displacement, displacement), (displacement, displacement), (0, 0)),
        mode=TRANSLATION_PADDING_MODE,
    )
    row_start = displacement
    column_start = displacement

    if direction == "up":
        row_start += displacement
    elif direction == "down":
        row_start -= displacement
    elif direction == "left":
        column_start += displacement
    elif direction == "right":
        column_start -= displacement

    row_end = row_start + IMAGE_SIZE
    column_end = column_start + IMAGE_SIZE
    translated_array = padded_array[row_start:row_end, column_start:column_end]
    return Image.fromarray(translated_array)


def create_patch_permutation(image_index, grid_size=PATCH_GRID_SIZE, seed=SEED):
    """Create one deterministic non-identity patch permutation for an image."""
    if not isinstance(image_index, (int, np.integer)) or image_index < 0:
        raise ValueError("Image index must be a non-negative integer.")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("Patch grid size must be an integer of at least two.")

    patch_count = grid_size * grid_size
    identity_permutation = np.arange(patch_count)
    seed_sequence = np.random.SeedSequence([int(seed), int(image_index)])
    random_generator = np.random.default_rng(seed_sequence)
    permutation = random_generator.permutation(patch_count)

    while np.array_equal(permutation, identity_permutation):
        permutation = random_generator.permutation(patch_count)
    return permutation.tolist()


def apply_patch_permutation(image, permutation, grid_size=PATCH_GRID_SIZE):
    """Rearrange image patches according to a recorded permutation."""
    common_image = prepare_common_image(image)
    image_array = np.asarray(common_image)
    patch_count = grid_size * grid_size
    permutation_array = np.asarray(permutation, dtype=np.int64)

    if IMAGE_SIZE % grid_size != 0:
        raise ValueError("Image size must be divisible by the patch grid size.")
    if permutation_array.shape != (patch_count,):
        raise ValueError(f"Patch permutation must contain {patch_count} indices.")
    if sorted(permutation_array.tolist()) != list(range(patch_count)):
        raise ValueError("Patch permutation must contain every patch index once.")
    if np.array_equal(permutation_array, np.arange(patch_count)):
        raise ValueError("Patch permutation must not be the identity ordering.")

    patch_size = IMAGE_SIZE // grid_size
    shuffled_array = np.empty_like(image_array)

    for destination_index, source_index in enumerate(permutation_array):
        destination_row = destination_index // grid_size
        destination_column = destination_index % grid_size
        source_row = int(source_index) // grid_size
        source_column = int(source_index) % grid_size

        source_row_start = source_row * patch_size
        source_column_start = source_column * patch_size
        destination_row_start = destination_row * patch_size
        destination_column_start = destination_column * patch_size

        shuffled_array[
            destination_row_start:destination_row_start + patch_size,
            destination_column_start:destination_column_start + patch_size,
        ] = image_array[
            source_row_start:source_row_start + patch_size,
            source_column_start:source_column_start + patch_size,
        ]

    return Image.fromarray(shuffled_array)


def apply_patch_shuffle(image, image_index):
    """Create and apply the fixed patch permutation for one official image index."""
    permutation = create_patch_permutation(image_index)
    shuffled_image = apply_patch_permutation(image, permutation)
    return shuffled_image, permutation
