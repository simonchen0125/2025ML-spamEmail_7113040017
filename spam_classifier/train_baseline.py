from __future__ import annotations

import pathlib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    # When executed as part of the package
    from .download_dataset import OUTPUT_FILENAME, download_dataset, load_dataset
except ImportError:  # pragma: no cover - fallback for direct script execution
    from download_dataset import OUTPUT_FILENAME, download_dataset, load_dataset


def ensure_dataset(project_root: pathlib.Path) -> pathlib.Path:
    """Ensure the dataset CSV exists, downloading it if necessary."""
    csv_path = project_root / OUTPUT_FILENAME
    if not csv_path.exists():
        download_dataset(csv_path)
    return csv_path


def build_pipeline() -> Pipeline:
    """Create the text vectorizer + linear SVM pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("svm", LinearSVC(random_state=42)),
        ]
    )


def print_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Pretty-print the confusion matrix for ham/spam predictions."""
    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="Actual"),
        columns=pd.Index(labels, name="Predicted"),
    )
    print("\nConfusion Matrix:")
    print(matrix_df)


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parent
    csv_path = ensure_dataset(project_root)
    df = load_dataset(csv_path)

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()
