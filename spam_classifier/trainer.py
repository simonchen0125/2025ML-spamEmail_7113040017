from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from joblib import dump
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


@dataclass
class TrainingResult:
    model: ClassifierMixin
    accuracy: float
    classification_report: str
    confusion_matrix: pd.DataFrame
    X_test: csr_matrix
    y_test: pd.Series
    y_pred: pd.Series


def split_dataset(
    features: csr_matrix, labels: pd.Series, *, test_size: float = 0.2, random_state: int = 42
) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
    """Split features and labels into train and test sets."""
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


def train_svm(X_train: csr_matrix, y_train: pd.Series) -> LinearSVC:
    """Train a baseline linear SVM classifier."""
    model = LinearSVC(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train: csr_matrix, y_train: pd.Series) -> LogisticRegression:
    """Train a logistic regression classifier with basic regularization."""
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train: csr_matrix, y_train: pd.Series) -> MultinomialNB:
    """Train a Multinomial Naive Bayes classifier."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: ClassifierMixin, X_test: csr_matrix, y_test: pd.Series) -> TrainingResult:
    """Evaluate the trained model and collect metrics."""
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="Actual"),
        columns=pd.Index(labels, name="Predicted"),
    )
    return TrainingResult(
        model=model,
        accuracy=accuracy,
        classification_report=report,
        confusion_matrix=matrix_df,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
    )


def save_artifacts(
    model: ClassifierMixin,
    vectorizer: TfidfVectorizer,
    model_path: pathlib.Path,
    vectorizer_path: pathlib.Path,
) -> None:
    """Persist the trained model and vectorizer."""
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vectorizer_path}")
