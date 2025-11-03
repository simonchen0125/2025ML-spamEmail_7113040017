from __future__ import annotations

import pathlib

import pandas as pd
from joblib import load
from scipy.sparse import load_npz

try:
    from .paths import (
        ARTIFACTS_DIRNAME,
        FEATURES_FILENAME,
        LABELS_FILENAME,
        NB_MODEL_FILENAME,
        VECTORIZER_FILENAME,
    )
    from .trainer import evaluate_model, save_artifacts, split_dataset, train_naive_bayes
except ImportError:  # pragma: no cover - direct execution fallback
    from paths import (
        ARTIFACTS_DIRNAME,
        FEATURES_FILENAME,
        LABELS_FILENAME,
        NB_MODEL_FILENAME,
        VECTORIZER_FILENAME,
    )
    from trainer import evaluate_model, save_artifacts, split_dataset, train_naive_bayes


def main() -> None:
    """Train and evaluate the Multinomial Naive Bayes spam classifier."""
    project_root = pathlib.Path(__file__).resolve().parent
    artifacts_dir = project_root / ARTIFACTS_DIRNAME

    features_path = artifacts_dir / FEATURES_FILENAME
    labels_path = artifacts_dir / LABELS_FILENAME
    vectorizer_path = artifacts_dir / VECTORIZER_FILENAME
    model_path = artifacts_dir / NB_MODEL_FILENAME

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Run preprocess_data.py before training Naive Bayes."
        )

    print("Loading preprocessed features and labels...")
    features = load_npz(features_path)
    labels = pd.read_csv(labels_path)["label"]
    vectorizer = load(vectorizer_path)

    print("Splitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    print("Training Naive Bayes classifier...")
    model = train_naive_bayes(X_train, y_train)

    print("Evaluating model...")
    result = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy: {result.accuracy:.4f}")
    print("\nClassification Report:")
    print(result.classification_report)
    print("\nConfusion Matrix:")
    print(result.confusion_matrix)

    print("\nSaving artifacts...")
    save_artifacts(model, vectorizer, model_path, vectorizer_path)
    print(f"Saved Naive Bayes model to {model_path}")


if __name__ == "__main__":
    main()
