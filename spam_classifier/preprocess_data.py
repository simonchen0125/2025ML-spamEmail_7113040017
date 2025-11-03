from __future__ import annotations

import pathlib

from joblib import dump
from scipy.sparse import save_npz

try:
    from .data_loader import get_dataset
    from .paths import (
        ARTIFACTS_DIRNAME,
        FEATURES_FILENAME,
        LABELS_FILENAME,
        VECTORIZER_FILENAME,
    )
    from .preprocessor import prepare_features
except ImportError:  # pragma: no cover - direct execution fallback
    from data_loader import get_dataset
    from paths import ARTIFACTS_DIRNAME, FEATURES_FILENAME, LABELS_FILENAME, VECTORIZER_FILENAME
    from preprocessor import prepare_features


def ensure_artifacts_dir(project_root: pathlib.Path) -> pathlib.Path:
    """Ensure the artifacts directory exists and return its path."""
    artifacts_dir = project_root / ARTIFACTS_DIRNAME
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parent
    artifacts_dir = ensure_artifacts_dir(project_root)

    print("Loading dataset for preprocessing...")
    df = get_dataset(project_root)

    print("Cleaning and vectorizing messages...")
    vectorizer, features, labels = prepare_features(df)

    vectorizer_path = artifacts_dir / VECTORIZER_FILENAME
    features_path = artifacts_dir / FEATURES_FILENAME
    labels_path = artifacts_dir / LABELS_FILENAME

    dump(vectorizer, vectorizer_path)
    save_npz(features_path, features)
    labels.to_csv(labels_path, index=False)

    print("Preprocessing complete:")
    print(f"- Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"- Features saved to: {features_path}")
    print(f"- Labels saved to:   {labels_path}")
    print(f"- Vectorizer saved to: {vectorizer_path}")


if __name__ == "__main__":
    main()
