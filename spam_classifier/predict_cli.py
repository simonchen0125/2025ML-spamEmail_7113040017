from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
from joblib import load

try:
    from .paths import (
        ARTIFACTS_DIRNAME,
        ENSEMBLE_MODEL_FILENAME,
        LOGREG_MODEL_FILENAME,
        MODEL_FILENAME,
        NB_MODEL_FILENAME,
        VECTORIZER_FILENAME,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from paths import (
        ARTIFACTS_DIRNAME,
        ENSEMBLE_MODEL_FILENAME,
        LOGREG_MODEL_FILENAME,
        MODEL_FILENAME,
        NB_MODEL_FILENAME,
        VECTORIZER_FILENAME,
    )


MODEL_CHOICES = {
    "svm": MODEL_FILENAME,
    "logreg": LOGREG_MODEL_FILENAME,
    "nb": NB_MODEL_FILENAME,
    "ensemble": ENSEMBLE_MODEL_FILENAME,
}


def resolve_model_filename(choice: str | None) -> str:
    """Return the artifact filename for the requested model."""
    if choice is None:
        return MODEL_FILENAME
    choice = choice.lower()
    if choice not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{choice}'. Choose from: {', '.join(MODEL_CHOICES)}")
    return MODEL_CHOICES[choice]


def load_artifacts(project_root: pathlib.Path, model_choice: str | None):
    artifacts_dir = project_root / ARTIFACTS_DIRNAME
    model_filename = resolve_model_filename(model_choice)
    model_path = artifacts_dir / model_filename
    vectorizer_path = artifacts_dir / VECTORIZER_FILENAME

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Model or vectorizer artifacts are missing. Train the requested model first."
        )

    model = load(model_path)
    vectorizer = load(vectorizer_path)
    return model, vectorizer


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict SMS spam/ham labels using the trained models."
    )
    parser.add_argument(
        "messages",
        nargs="*",
        help="One or more SMS messages to classify. If omitted, messages are read from stdin.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CHOICES),
        help="Select which trained model to use (default: svm).",
    )
    return parser.parse_args(argv)


def read_messages(args: argparse.Namespace) -> list[str]:
    if args.messages:
        return args.messages
    print("Enter SMS messages (Ctrl+D to finish):", file=sys.stderr)
    return [line.strip() for line in sys.stdin if line.strip()]


def main(argv: list[str] | None = None) -> None:
    project_root = pathlib.Path(__file__).resolve().parent
    args = parse_args(argv or sys.argv[1:])
    messages = read_messages(args)

    if not messages:
        print("No messages provided.", file=sys.stderr)
        return

    model, vectorizer = load_artifacts(project_root, args.model)
    features = vectorizer.transform(messages)
    predictions = model.predict(features)
    score_label = None
    scores = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        scores = proba[:, 1]
        score_label = "probability"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        score_label = "decision_score"

    results = pd.DataFrame({"message": messages, "prediction": predictions})
    if scores is not None:
        results[score_label] = scores
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
