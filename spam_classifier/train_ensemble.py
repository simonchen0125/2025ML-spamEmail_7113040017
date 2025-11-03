from __future__ import annotations

import pathlib

import pandas as pd
from joblib import load
from scipy.sparse import load_npz
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

try:
    from .paths import (
        ARTIFACTS_DIRNAME,
        ENSEMBLE_MODEL_FILENAME,
        FEATURES_FILENAME,
        LABELS_FILENAME,
        VECTORIZER_FILENAME,
    )
    from .trainer import evaluate_model, save_artifacts, split_dataset
except ImportError:  # pragma: no cover - direct execution fallback
    from paths import (
        ARTIFACTS_DIRNAME,
        ENSEMBLE_MODEL_FILENAME,
        FEATURES_FILENAME,
        LABELS_FILENAME,
        VECTORIZER_FILENAME,
    )
    from trainer import evaluate_model, save_artifacts, split_dataset


def build_ensemble() -> VotingClassifier:
    """Construct a soft-voting ensemble using logistic regression and Naive Bayes."""
    estimators = [
        (
            "logreg",
            LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        ("nb", MultinomialNB()),
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


def main() -> None:
    """Train and evaluate a soft-voting ensemble classifier."""
    project_root = pathlib.Path(__file__).resolve().parent
    artifacts_dir = project_root / ARTIFACTS_DIRNAME

    features_path = artifacts_dir / FEATURES_FILENAME
    labels_path = artifacts_dir / LABELS_FILENAME
    vectorizer_path = artifacts_dir / VECTORIZER_FILENAME
    model_path = artifacts_dir / ENSEMBLE_MODEL_FILENAME

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Run preprocess_data.py before training the ensemble."
        )

    print("Loading preprocessed features and labels...")
    features = load_npz(features_path)
    labels = pd.read_csv(labels_path)["label"]
    vectorizer = load(vectorizer_path)

    print("Splitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    print("Training soft-voting ensemble (LogReg + Naive Bayes)...")
    ensemble = build_ensemble()
    ensemble.fit(X_train, y_train)

    print("Evaluating ensemble...")
    result = evaluate_model(ensemble, X_test, y_test)

    print(f"\nAccuracy: {result.accuracy:.4f}")
    print("\nClassification Report:")
    print(result.classification_report)
    print("\nConfusion Matrix:")
    print(result.confusion_matrix)

    print("\nSaving artifacts...")
    save_artifacts(ensemble, vectorizer, model_path, vectorizer_path)
    print(f"Saved ensemble model to {model_path}")


if __name__ == "__main__":
    main()
