from __future__ import annotations

import pathlib
import urllib.request

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-"
    "Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)
OUTPUT_FILENAME = "sms_spam.csv"


def download_dataset(destination: pathlib.Path) -> None:
    """Download the SMS spam dataset to the given destination path."""
    print(f"Downloading dataset from {DATA_URL}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, destination)
    print(f"Saved dataset to {destination}")


def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load the downloaded dataset into a pandas DataFrame."""
    column_names = ["label", "message"]
    df = pd.read_csv(csv_path, header=None, names=column_names)
    return df


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parent
    csv_path = project_root / OUTPUT_FILENAME

    download_dataset(csv_path)

    df = load_dataset(csv_path)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataFrame info:")
    df.info()


if __name__ == "__main__":
    main()
