# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Download, verify, transform, and load the PACS image domains."""

import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights
from torchvision.transforms import v2


PACS_CLASSES = (
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
)
PACS_DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
PACS_SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
PACS_EXPECTED_DOMAIN_COUNTS = {
    "photo": 1670,
    "art_painting": 2048,
    "cartoon": 2344,
    "sketch": 3929,
}
PACS_HUGGING_FACE_DATASET = "flwrlabs/pacs"
PACS_HUGGING_FACE_CONFIG = "default"
PACS_HUGGING_FACE_SPLIT = "train"
PACS_HUGGING_FACE_PAGE_SIZE = 100
PACS_HUGGING_FACE_WORKERS = 4
PACS_HUGGING_FACE_PAGE_DELAY = (0.5, 1.5)
DOWNLOAD_MAX_ATTEMPTS = 8
DOWNLOAD_MAX_RETRY_DELAY = 60.0
RETRYABLE_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
DOWNLOAD_USER_AGENT = "Mozilla/5.0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def create_training_transform(resize_size=256, crop_size=224):
    """Build the required random training transform with ImageNet normalization."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    normalization = weights.transforms()
    return v2.Compose(
        [
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.RandomCrop((crop_size, crop_size)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )


def create_evaluation_transform(resize_size=256, crop_size=224):
    """Build the deterministic validation and final-evaluation transform."""
    weights = ResNet18_Weights.IMAGENET1K_V1
    normalization = weights.transforms()
    return v2.Compose(
        [
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.CenterCrop((crop_size, crop_size)),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )


def _hugging_face_rows_url(offset, length):
    """Return one Hugging Face dataset-viewer page URL for PACS."""
    query = urllib.parse.urlencode(
        {
            "dataset": PACS_HUGGING_FACE_DATASET,
            "config": PACS_HUGGING_FACE_CONFIG,
            "split": PACS_HUGGING_FACE_SPLIT,
            "offset": int(offset),
            "length": int(length),
        }
    )
    return f"https://datasets-server.huggingface.co/rows?{query}"


def _read_json_url(url):
    """Read public JSON with jittered retries for rate limits and server errors."""
    for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": DOWNLOAD_USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception as error:
            if attempt == DOWNLOAD_MAX_ATTEMPTS - 1 or not _is_retryable_error(error):
                raise
            retry_delay = _calculate_retry_delay(error, attempt)
            print(
                "Hugging Face metadata request was rate limited or unavailable; "
                f"retrying in {retry_delay:.1f} seconds..."
            )
            time.sleep(retry_delay)

    raise RuntimeError("Hugging Face metadata could not be downloaded.")


def _is_retryable_error(error):
    """Return whether a network or response error may succeed on a later attempt."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_HTTP_CODES
    return isinstance(
        error,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ),
    )


def _calculate_retry_delay(error, attempt):
    """Combine Retry-After or exponential backoff with random jitter."""
    retry_after = None
    if isinstance(error, urllib.error.HTTPError):
        retry_after = error.headers.get("Retry-After")

    try:
        base_delay = float(retry_after)
    except (TypeError, ValueError):
        base_delay = float(2 ** attempt)

    jitter = random.uniform(0.25, 1.25)
    return min(DOWNLOAD_MAX_RETRY_DELAY, base_delay + jitter)


def _is_valid_image_file(image_path):
    """Return whether a path contains an image that Pillow can verify."""
    image_path = Path(image_path)
    if not image_path.is_file():
        return False
    try:
        with Image.open(image_path) as image_file:
            image_file.verify()
    except Exception:
        return False
    return True


def _download_hugging_face_image(download_job):
    """Download one PACS image atomically and return whether work was required."""
    image_url, image_path = download_job
    image_path = Path(image_path)
    if _is_valid_image_file(image_path):
        return False

    image_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = image_path.with_suffix(image_path.suffix + ".part")
    partial_path.unlink(missing_ok=True)

    for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
        try:
            request = urllib.request.Request(
                image_url,
                headers={"User-Agent": DOWNLOAD_USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)

            if not _is_valid_image_file(partial_path):
                raise RuntimeError(f"Invalid image returned for {image_path.name}.")

            partial_path.replace(image_path)
            return True
        except Exception as error:
            partial_path.unlink(missing_ok=True)
            if attempt == DOWNLOAD_MAX_ATTEMPTS - 1 or not _is_retryable_error(error):
                raise
            time.sleep(_calculate_retry_delay(error, attempt))

    return False


def download_hugging_face_pacs(dataset_directory, download_directory):
    """Download the complete public Hugging Face PACS dataset into class folders."""
    dataset_path = Path(dataset_directory)
    cache_path = Path(download_directory)
    dataset_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    expected_total = sum(PACS_EXPECTED_DOMAIN_COUNTS.values())
    completed_rows = 0
    downloaded_images = 0

    for offset in range(0, expected_total, PACS_HUGGING_FACE_PAGE_SIZE):
        page_length = min(PACS_HUGGING_FACE_PAGE_SIZE, expected_total - offset)
        payload = _read_json_url(_hugging_face_rows_url(offset, page_length))
        reported_total = int(payload.get("num_rows_total", -1))
        if reported_total != expected_total:
            raise ValueError(
                "The Hugging Face PACS row count changed: "
                f"expected {expected_total}, received {reported_total}."
            )

        download_jobs = []
        for row_item in payload.get("rows", []):
            row_index = int(row_item["row_idx"])
            row = row_item["row"]
            domain = str(row["domain"])
            label = int(row["label"])
            if domain not in PACS_DOMAINS:
                raise ValueError(f"Unexpected PACS domain from Hugging Face: {domain}")
            if not 0 <= label < len(PACS_CLASSES):
                raise ValueError(f"Unexpected PACS label from Hugging Face: {label}")

            image_url = str(row["image"]["src"])
            class_name = PACS_CLASSES[label]
            image_path = dataset_path / domain / class_name / f"{row_index:05d}.jpg"
            download_jobs.append((image_url, image_path))

        if len(download_jobs) != page_length:
            raise RuntimeError(
                f"Hugging Face returned {len(download_jobs)} of {page_length} requested rows."
            )

        with ThreadPoolExecutor(max_workers=PACS_HUGGING_FACE_WORKERS) as executor:
            downloaded_images += sum(executor.map(_download_hugging_face_image, download_jobs))

        completed_rows += len(download_jobs)
        print(f"Prepared {completed_rows}/{expected_total} PACS images...")
        time.sleep(random.uniform(*PACS_HUGGING_FACE_PAGE_DELAY))

    manifest = {
        "dataset": PACS_HUGGING_FACE_DATASET,
        "config": PACS_HUGGING_FACE_CONFIG,
        "split": PACS_HUGGING_FACE_SPLIT,
        "total_images": expected_total,
        "domain_counts": PACS_EXPECTED_DOMAIN_COUNTS,
    }
    manifest_path = cache_path / "pacs_huggingface_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded {downloaded_images} new PACS images from Hugging Face.")
    print(f"Saved PACS manifest: {manifest_path.resolve()}")
    return dataset_path


def find_pacs_image_root(search_directory):
    """Find the directory whose immediate children are the four PACS domains."""
    search_path = Path(search_directory)
    candidates = [search_path]
    if search_path.exists():
        candidates.extend(path for path in search_path.rglob("*") if path.is_dir())

    required_domains = set(PACS_DOMAINS)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        child_directories = {
            child.name for child in candidate.iterdir() if child.is_dir()
        }
        if required_domains.issubset(child_directories):
            return candidate
    raise FileNotFoundError(
        f"Could not find the four PACS domain folders below {search_path.resolve()}."
    )


def find_pacs_source_root(search_directory):
    """Find the PACS root without descending into the held-out Sketch domain."""
    search_path = Path(search_directory)
    required_domains = set(PACS_SOURCE_DOMAINS)
    candidates = [(search_path, 0)]
    visited_directories = set()

    while candidates:
        candidate, depth = candidates.pop(0)
        if not candidate.is_dir():
            continue
        resolved_candidate = candidate.resolve()
        if resolved_candidate in visited_directories:
            continue
        visited_directories.add(resolved_candidate)

        child_directories = [child for child in candidate.iterdir() if child.is_dir()]
        child_names = {child.name for child in child_directories}
        if required_domains.issubset(child_names):
            return candidate

        if depth < 4:
            candidates.extend(
                (child, depth + 1)
                for child in child_directories
                if child.name.lower() != "sketch"
            )
    raise FileNotFoundError(
        "Could not find the Photo, Art Painting, and Cartoon folders below "
        f"{search_path.resolve()}. Run the Task 2 data-preparation stage first."
    )


def scan_domain_records(image_root, domain, include_labels=True):
    """Create deterministic image records for one PACS domain."""
    if domain not in PACS_DOMAINS:
        raise ValueError(f"Unknown PACS domain: {domain}")

    root_path = Path(image_root)
    domain_path = root_path / domain
    records = []
    for class_index, class_name in enumerate(PACS_CLASSES):
        class_directory = domain_path / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Missing PACS class directory: {class_directory}")

        image_paths = sorted(
            path
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            relative_path = image_path.relative_to(root_path).as_posix()
            if include_labels:
                identifier = relative_path
            else:
                path_digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
                identifier = f"{domain}_{path_digest[:16]}"
            record = {
                "identifier": identifier,
                "relative_path": relative_path,
                "domain": domain,
            }
            if include_labels:
                record["label"] = class_index
                record["class_name"] = class_name
            records.append(record)
    return records


def verify_pacs_dataset(image_root, require_official_counts=True):
    """Check the PACS domains, classes, and official image counts."""
    root_path = find_pacs_image_root(image_root)
    domain_counts = {
        domain: len(scan_domain_records(root_path, domain, include_labels=False))
        for domain in PACS_DOMAINS
    }
    if require_official_counts and domain_counts != PACS_EXPECTED_DOMAIN_COUNTS:
        raise ValueError(
            "PACS image counts do not match the official 9,991-image dataset: "
            f"{domain_counts}"
        )
    return root_path, domain_counts


def prepare_pacs_dataset(
    dataset_directory,
    download_directory,
    download=True,
    require_official_counts=True,
):
    """Reuse or download PACS, then return its verified image-root directory."""
    dataset_path = Path(dataset_directory)
    try:
        image_root = find_pacs_image_root(dataset_path)
    except FileNotFoundError:
        if not download:
            raise

        download_hugging_face_pacs(
            dataset_directory=dataset_path,
            download_directory=download_directory,
        )
        image_root = find_pacs_image_root(dataset_path)

    image_root, domain_counts = verify_pacs_dataset(
        image_root,
        require_official_counts=require_official_counts,
    )
    print(f"PACS image root: {image_root.resolve()}")
    print(f"PACS domain counts: {domain_counts}")
    return image_root


class PACSDomainDataset(Dataset):
    """Load PACS records while optionally keeping class labels hidden."""

    def __init__(self, image_root, records, transform=None, include_labels=True):
        self.image_root = Path(image_root)
        self.transform = transform
        self.include_labels = bool(include_labels)
        self.records = []

        for original_record in records:
            record = {
                "identifier": str(original_record["identifier"]),
                "relative_path": str(original_record["relative_path"]),
                "domain": str(original_record["domain"]),
            }
            if self.include_labels:
                if "label" not in original_record:
                    raise ValueError("A labeled PACS dataset requires labels in every record.")
                record["label"] = int(original_record["label"])
                record["class_name"] = str(original_record["class_name"])
            self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.image_root / record["relative_path"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "domain": record["domain"],
            "identifier": record["identifier"],
        }
        if self.include_labels:
            sample["label"] = record["label"]
        return sample
