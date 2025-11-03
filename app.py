from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from scipy.sparse import load_npz

from spam_classifier.data_loader import get_dataset
from spam_classifier.paths import (
    ARTIFACTS_DIRNAME,
    ENSEMBLE_MODEL_FILENAME,
    FEATURES_FILENAME,
    LABELS_FILENAME,
    LOGREG_MODEL_FILENAME,
    MODEL_FILENAME,
    NB_MODEL_FILENAME,
    VECTORIZER_FILENAME,
)
from spam_classifier.trainer import TrainingResult, evaluate_model, split_dataset

BASE_DIR = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = BASE_DIR / "spam_classifier"
ARTIFACTS_DIR = PACKAGE_DIR / ARTIFACTS_DIRNAME

MODEL_FILES = {
    "SVM (LinearSVC)": MODEL_FILENAME,
    "Logistic Regression": LOGREG_MODEL_FILENAME,
    "Naive Bayes": NB_MODEL_FILENAME,
    "Ensemble (LogReg + Naive Bayes)": ENSEMBLE_MODEL_FILENAME,
}


@st.cache_resource(show_spinner=False)
def load_vectorizer_and_models() -> Tuple:
    """Load the shared TF-IDF vectorizer and any available trained models."""
    vectorizer_path = ARTIFACTS_DIR / VECTORIZER_FILENAME
    if not vectorizer_path.exists():
        st.error(
            "TF-IDF vectorizer missing. Run the preprocessing script before starting the app."
        )
        st.stop()

    vectorizer = load(vectorizer_path)
    models: Dict[str, object] = {}
    missing: List[str] = []

    for display_name, filename in MODEL_FILES.items():
        model_path = ARTIFACTS_DIR / filename
        if model_path.exists():
            models[display_name] = load(model_path)
        else:
            missing.append(display_name)

    if not models:
        st.error(
            "No trained models were found. Please run the training scripts before launching the app."
        )
        st.stop()

    return vectorizer, models, missing


@st.cache_resource(show_spinner=False)
def load_evaluation_data():
    """Load cached TF-IDF features and labels for on-demand evaluation."""
    features_path = ARTIFACTS_DIR / FEATURES_FILENAME
    labels_path = ARTIFACTS_DIR / LABELS_FILENAME

    if not features_path.exists() or not labels_path.exists():
        return None, None

    features = load_npz(features_path)
    labels = pd.read_csv(labels_path)["label"]
    return features, labels


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame | None:
    """Load the raw dataset for exploration."""
    try:
        return get_dataset(PACKAGE_DIR)
    except Exception as exc:  # pragma: no cover - defensive
        st.warning(f"Unable to load dataset: {exc}")
        return None


def compute_metrics(models: Dict[str, object], features, labels) -> Dict[str, TrainingResult]:
    """Evaluate each loaded model on a held-out split."""
    if features is None or labels is None:
        return {}

    _, X_test, _, y_test = split_dataset(features, labels)
    metrics: Dict[str, TrainingResult] = {}
    for display_name, model in models.items():
        try:
            metrics[display_name] = evaluate_model(model, X_test, y_test)
        except Exception as exc:  # pragma: no cover - defensive
            st.warning(f"Unable to evaluate {display_name}: {exc}")
    return metrics


def run_predictions(model, vectorizer, messages: List[str]):
    """Generate predictions and scores for the provided messages."""
    features = vectorizer.transform(messages)
    predictions = model.predict(features)

    score_label = None
    scores = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        scores = proba[:, 1]
        score_label = "Probability"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        score_label = "Decision Score"
    return predictions, scores, score_label


def compute_top_tokens(
    features, labels: pd.Series, vectorizer, target_label: str, *, top_n: int = 15
) -> pd.DataFrame:
    """Return the top TF-IDF tokens for the given label."""
    if features is None or labels is None or vectorizer is None:
        return pd.DataFrame()

    mask = (labels == target_label).to_numpy()
    if mask.sum() == 0:
        return pd.DataFrame()

    subset = features[mask]
    mean_scores = np.asarray(subset.mean(axis=0)).ravel()
    tokens = vectorizer.get_feature_names_out()
    df = pd.DataFrame({"token": tokens, "score": mean_scores})
    return df.sort_values("score", ascending=False).head(top_n)


def render_sidebar(metrics: Dict[str, TrainingResult], missing_models: List[str]) -> None:
    """Render sidebar with metrics and housekeeping information."""
    st.sidebar.header("Model Metrics")
    if not metrics:
        st.sidebar.info("Run preprocessing and training to enable live metrics.")
    else:
        for name, result in metrics.items():
            st.sidebar.markdown(f"**{name}**")
            st.sidebar.metric("Accuracy", f"{result.accuracy:.4f}")
            st.sidebar.table(result.confusion_matrix)
            with st.sidebar.expander("Classification report"):
                st.text(result.classification_report)

    if missing_models:
        st.sidebar.caption(
            "Unavailable models: " + ", ".join(missing_models) + ". Train them to enable selection."
        )


def render_probability_chart(scores: np.ndarray, predictions: List[str], label: str) -> None:
    """Show a histogram for probability / decision scores."""
    if scores is None or len(scores) < 2:
        return

    chart_df = pd.DataFrame({"score": scores, "prediction": predictions})
    chart = (
        alt.Chart(chart_df)
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=20), title=label),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("prediction:N", title="Prediction"),
            tooltip=["prediction", alt.Tooltip("count():Q", title="Count")],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def render_dataset_explorer(
    raw_dataset: pd.DataFrame | None,
    features,
    labels: pd.Series | None,
    vectorizer,
) -> None:
    """Display dataset-level insights."""
    st.subheader("Dataset Exploration")
    if raw_dataset is None:
        st.info("Dataset preview unavailable. Run data_ingestion.py to download the dataset.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.write("樣本數量 (Samples per label)")
        counts_df = raw_dataset["label"].value_counts().reset_index()
        counts_df.columns = ["label", "count"]
        chart = (
            alt.Chart(counts_df)
            .mark_bar()
            .encode(x=alt.X("label:N", title="Label"), y=alt.Y("count:Q", title="Count"))
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.write("訊息長度分佈 (Message length)")
        length_df = raw_dataset.assign(length=raw_dataset["message"].str.len())
        length_chart = (
            alt.Chart(length_df)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X("length:Q", bin=alt.Bin(maxbins=30), title="Characters"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("count():Q", title="Count")],
            )
        )
        st.altair_chart(length_chart, use_container_width=True)

    if features is None or labels is None:
        st.info("Run preprocessing to enable token-level insights.")
        return

    spam_top = compute_top_tokens(features, labels, vectorizer, "spam")
    ham_top = compute_top_tokens(features, labels, vectorizer, "ham")
    col_spam, col_ham = st.columns(2)
    with col_spam:
        st.write("垃圾簡訊關鍵詞 (Top spam tokens)")
        st.dataframe(spam_top.reset_index(drop=True), use_container_width=True)
    with col_ham:
        st.write("正常簡訊關鍵詞 (Top ham tokens)")
        st.dataframe(ham_top.reset_index(drop=True), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Spam SMS Classifier", layout="wide")
    st.title("簡訊垃圾郵件分類器")
    st.write(
        "這是一個使用 SVM 機器學習模型來偵測垃圾簡訊的應用程式。請在下方輸入一則英文訊息來進行測試。"
    )

    vectorizer, models, missing_models = load_vectorizer_and_models()
    features, labels = load_evaluation_data()
    raw_dataset = load_raw_dataset()

    metrics = compute_metrics(models, features, labels)
    render_sidebar(metrics, missing_models)

    model_names = list(models.keys())
    selected_model_name = st.selectbox("Select model", model_names)
    selected_model = models[selected_model_name]

    st.subheader("Single Prediction")
    message_input = st.text_area("請在此輸入訊息：", height=150)

    if st.button("Predict", key="single_predict"):
        if not message_input.strip():
            st.warning("Please enter a message before predicting.")
        else:
            preds, scores, score_label = run_predictions(selected_model, vectorizer, [message_input])
            prediction = preds[0]
            if prediction.lower() == "spam":
                st.error("Prediction: Spam")
            else:
                st.success("Prediction: Ham (Not Spam)")
            if scores is not None:
                st.caption(f"{score_label}: {scores[0]:.4f}")

    st.divider()
    st.subheader("Batch Prediction")
    col_input, col_upload = st.columns([2, 1])

    with col_input:
        batch_messages_raw = st.text_area(
            "Batch messages (one per line)", key="batch_input", height=150
        )
    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV with a 'message' column", type=["csv"])

    if st.button("Batch Predict", key="batch_predict"):
        messages: List[str] = []
        if batch_messages_raw:
            messages.extend(
                [line.strip() for line in batch_messages_raw.splitlines() if line.strip()]
            )
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Unable to read uploaded file: {exc}")
                uploaded_df = None

            if uploaded_df is not None:
                if "message" not in uploaded_df.columns:
                    st.error("Uploaded CSV must contain a 'message' column.")
                else:
                    messages.extend(uploaded_df["message"].dropna().astype(str).tolist())

        if not messages:
            st.warning("Please provide at least one message for batch prediction.")
        else:
            preds, scores, score_label = run_predictions(selected_model, vectorizer, messages)
            results_df = pd.DataFrame({"message": messages, "prediction": preds})
            if scores is not None:
                column_name = score_label.lower().replace(" ", "_")
                results_df[column_name] = scores
            st.dataframe(results_df, use_container_width=True)
            if scores is not None:
                render_probability_chart(scores, preds.tolist(), score_label)

    st.divider()
    render_dataset_explorer(raw_dataset, features, labels, vectorizer)


if __name__ == "__main__":
    main()
