from __future__ import annotations

import re
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from scipy.sparse import csr_matrix

STOPWORDS = set(ENGLISH_STOP_WORDS)
PUNCTUATION_REGEX = re.compile(r"[^\w\s]")


def clean_message(message: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and drop stopwords."""
    lowered = message.lower()
    no_punct = PUNCTUATION_REGEX.sub(" ", lowered)
    tokens = [token for token in no_punct.split() if token and token not in STOPWORDS]
    return " ".join(tokens)


def preprocess_messages(df: pd.DataFrame) -> pd.Series:
    """Return cleaned message text as a Series."""
    return df["message"].astype(str).map(clean_message)


def vectorize_messages(messages: pd.Series) -> Tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer on the messages."""
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(messages)
    return vectorizer, features


def prepare_features(df: pd.DataFrame) -> Tuple[TfidfVectorizer, csr_matrix, pd.Series]:
    """Clean, vectorize, and return (vectorizer, features, labels)."""
    cleaned_messages = preprocess_messages(df)
    vectorizer, features = vectorize_messages(cleaned_messages)
    labels = df["label"]
    return vectorizer, features, labels
