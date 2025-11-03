from __future__ import annotations

import pathlib

try:
    from .data_loader import get_dataset
    from .preprocessor import prepare_features
    from .trainer import evaluate_model, save_artifacts, split_dataset, train_svm
except ImportError:  # pragma: no cover - support direct script execution
    from data_loader import get_dataset
    from preprocessor import prepare_features
    from trainer import evaluate_model, save_artifacts, split_dataset, train_svm


def main() -> None:
    project_root = pathlib.Path(__file__).resolve().parent

    print("Loading dataset...")
    df = get_dataset(project_root)

    print("Preprocessing messages...")
    vectorizer, features, labels = prepare_features(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    print("Training Linear SVM baseline...")
    model = train_svm(X_train, y_train)

    print("Evaluating model...")
    result = evaluate_model(model, X_test, y_test)

    print(f"\nAccuracy: {result.accuracy:.4f}")
    print("\nClassification Report:")
    print(result.classification_report)
    print("\nConfusion Matrix:")
    print(result.confusion_matrix)

    model_path = project_root / "svm_model.pkl"
    vectorizer_path = project_root / "vectorizer.pkl"
    save_artifacts(model, vectorizer, model_path, vectorizer_path)


if __name__ == "__main__":
    main()
