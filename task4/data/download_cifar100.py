# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Download CIFAR-100 files without constructing or exposing its dataset."""

from pathlib import Path

from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


def _cifar100_files_are_ready(data_directory):
    """Check every official CIFAR-100 train, test, and metadata file."""
    data_root = Path(data_directory)
    extracted_root = data_root / CIFAR100.base_folder
    required_files = list(CIFAR100.train_list) + list(CIFAR100.test_list)
    required_files.append(
        (CIFAR100.meta["filename"], CIFAR100.meta["md5"])
    )
    return all(
        check_integrity(extracted_root / filename, checksum)
        for filename, checksum in required_files
    )


def download_cifar100_files(data_directory, download=True):
    """Make CIFAR-100 available on disk without returning images or labels."""
    data_root = Path(data_directory)
    data_root.mkdir(parents=True, exist_ok=True)
    if _cifar100_files_are_ready(data_root):
        print(f"CIFAR-100 files already available: {data_root.resolve()}")
        return data_root.resolve()
    if not bool(download):
        raise RuntimeError(
            "CIFAR-100 files are missing and dataset downloading is disabled."
        )

    download_and_extract_archive(
        CIFAR100.url,
        download_root=str(data_root),
        filename=CIFAR100.filename,
        md5=CIFAR100.tgz_md5,
    )
    if not _cifar100_files_are_ready(data_root):
        raise RuntimeError("The downloaded CIFAR-100 files failed integrity checks.")
    print(
        "Downloaded CIFAR-100 files without constructing its evaluation dataset: "
        f"{data_root.resolve()}"
    )
    return data_root.resolve()
