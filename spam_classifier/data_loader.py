from __future__ import annotations

import pathlib
import urllib.request

import pandas as pd

try:
    from .paths import DATASET_DIRNAME, DATASET_FILENAME
except ImportError:  # pragma: no cover - direct execution fallback
    from paths import DATASET_DIRNAME, DATASET_FILENAME

DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-"
    "Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def dataset_path(project_root: pathlib.Path) -> pathlib.Path:
    """Return the expected dataset path inside the project."""
    return project_root / DATASET_DIRNAME / DATASET_FILENAME


def download_dataset(destination: pathlib.Path) -> pathlib.Path:
    """Download the dataset to the destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, destination)
    print(f"Saved dataset to {destination}")
    return destination


def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load the dataset into a DataFrame with named columns."""
    column_names = ["label", "message"]
    df = pd.read_csv(csv_path, header=None, names=column_names)
    return df


def ensure_dataset(project_root: pathlib.Path) -> pathlib.Path:
    """Ensure the dataset exists locally, downloading if necessary."""
    csv_path = dataset_path(project_root)
    if not csv_path.exists():
        download_dataset(csv_path)
    return csv_path


def get_dataset(project_root: pathlib.Path) -> pd.DataFrame:
    """Return the dataset as a DataFrame, downloading it if missing."""
    csv_path = ensure_dataset(project_root)
    return load_dataset(csv_path)
