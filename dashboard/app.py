"""Streamlit dashboard for the Zambia cyber-legislation sentiment project.

Run from the repository root:
    streamlit run dashboard/app.py
"""

from pathlib import Path
import base64
import importlib.util
import json
import os
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import light_clean, preprocess


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_preprocessed_3472.csv"
MODEL_ROOT = PROJECT_ROOT / "models"
COMPARISON_DIR = PROJECT_ROOT / "results" / "model_comparison"
LDA_DIR = PROJECT_ROOT / "results" / "lda"

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
RAW_RECORDS = 24_853
VALID_EXTRACTED_COMMENTS = 24_852
KEYWORD_ELIGIBLE_COMMENTS = 12_925
FINAL_CORPUS_COMMENTS = 3_472
SPLIT_DESCRIPTION = "70/15/15 stratified split, random seed 42"
COLORS = {
    "NEGATIVE": "#F16D6D",
    "NEUTRAL": "#F0EE91",
    "POSITIVE": "#65DB6F",
}
GREEN_SCALE = "Greens"
GREEN_SEQUENCE = ["#2CA25F", "#99D8C9", "#006D2C", "#C7E9C0"]
THEMES = {
    "Cyber legislation": r"\bcyber(?:\s+security|\s+crime|crime|crimes|\s+law|\s+bill)",
    "Surveillance and privacy": r"\b(?:surveillance|privacy|spy|spying|interception|monitor|monitored)\b",
    "Freedom of expression": r"\b(?:freedom of expression|free speech|human rights|digital rights|gag|silenc\w*|censor\w*)\b",
    "Authoritarianism": r"\b(?:dictator\w*|authoritarian\w*|tyranny|tyrant|police state|oppress\w*|totalitarian|regime)\b",
    "Penalties and enforcement": r"\b(?:imprisonment|jail|arrested|penalty|prison|enforcement)\b",
    "Legal process": r"\b(?:constitutional|unconstitutional|assent|enacted|gazette|high court|section\s+\d+)\b",
}


def apply_pyldavis_green_theme(html):
    """Inject a small script that recolours pyLDAvis' default blue/red elements."""
    injection = f"""
<style>
  body {{
    background: #ffffff;
  }}
</style>
<script>
(function () {{
  const inactiveTopic = "{GREEN_SEQUENCE[3]}";
  const activeTopic = "{GREEN_SEQUENCE[0]}";
  const overallBar = "{GREEN_SEQUENCE[1]}";
  const selectedBar = "{COLORS["POSITIVE"]}";
  const darkGreen = "{GREEN_SEQUENCE[2]}";

  function normalise(value) {{
    return (value || "").replace(/\\s+/g, "").toLowerCase();
  }}

  function isBlue(value) {{
    return value.includes("#1f77b4")
      || value.includes("#aec7e8")
      || value.includes("rgb(31,119,180)")
      || value.includes("31,119,180")
      || value.includes("steelblue")
      || value.includes("lightblue");
  }}

  function isRed(value) {{
    return value.includes("#d62728")
      || value.includes("#d95f02")
      || value.includes("rgb(214,39,40)")
      || value.includes("214,39,40")
      || value.includes("217,95,2")
      || value === "red";
  }}

  function paint(element, color) {{
    element.setAttribute("fill", color);
    element.style.fill = color;
  }}

  function recolourPyLDAvis() {{
    document.querySelectorAll("svg circle").forEach(function (element) {{
      const fill = normalise(element.getAttribute("fill") || element.style.fill);
      const klass = normalise(element.getAttribute("class"));
      if (isRed(fill) || klass.includes("selected")) {{
        paint(element, activeTopic);
      }} else if (isBlue(fill)) {{
        paint(element, inactiveTopic);
      }}
      const stroke = normalise(element.getAttribute("stroke") || element.style.stroke);
      if (isBlue(stroke) || isRed(stroke)) {{
        element.setAttribute("stroke", darkGreen);
        element.style.stroke = darkGreen;
      }}
    }});

    document.querySelectorAll("svg rect").forEach(function (element) {{
      const fill = normalise(element.getAttribute("fill") || element.style.fill);
      if (isRed(fill)) {{
        paint(element, selectedBar);
      }} else if (isBlue(fill)) {{
        paint(element, overallBar);
      }}
    }});

    document.querySelectorAll("svg path").forEach(function (element) {{
      const fill = normalise(element.getAttribute("fill") || element.style.fill);
      const stroke = normalise(element.getAttribute("stroke") || element.style.stroke);
      if (isRed(fill)) {{
        paint(element, activeTopic);
      }} else if (isBlue(fill)) {{
        paint(element, inactiveTopic);
      }}
      if (isBlue(stroke) || isRed(stroke)) {{
        element.setAttribute("stroke", darkGreen);
        element.style.stroke = darkGreen;
      }}
    }});
  }}

  function startThemeWatcher() {{
    recolourPyLDAvis();
    let attempts = 0;
    const timer = setInterval(function () {{
      recolourPyLDAvis();
      attempts += 1;
      if (attempts > 20) clearInterval(timer);
    }}, 350);
    if (document.body) {{
      new MutationObserver(recolourPyLDAvis).observe(
        document.body,
        {{ childList: true, subtree: true, attributes: true }}
      );
    }}
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", startThemeWatcher);
  }} else {{
    startThemeWatcher();
  }}
}})();
</script>
"""
    if "</body>" in html:
        return html.replace("</body>", f"{injection}</body>", 1)
    return f"{html}{injection}"

st.set_page_config(
    page_title="Zambia Cyber-Legislation Sentiment",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_corpus(path):
    frame = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    text_column = "text_original" if "text_original" in frame.columns else "text"
    label_column = "label" if "label" in frame.columns else "human_label"
    date_column = "date_utc" if "date_utc" in frame.columns else "date"

    required = {text_column, label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    frame["text_display"] = frame[text_column].fillna("").astype(str)
    frame["sentiment"] = frame[label_column].astype(str)
    frame["date_parsed"] = pd.to_datetime(
        frame.get(date_column),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    for theme, pattern in THEMES.items():
        frame[theme] = frame["text_display"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    return frame


@st.cache_data(show_spinner=False)
def load_csv_if_present(path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    path = MODEL_ROOT / "svm" / "tfidf_svm.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def load_word2vec_model():
    svm_path = MODEL_ROOT / "svm" / "w2v_svm.joblib"
    vectors_path = MODEL_ROOT / "svm" / "word2vec_sg300.joblib"
    if not svm_path.exists() or not vectors_path.exists():
        return None
    return joblib.load(svm_path), joblib.load(vectors_path)


@st.cache_resource(show_spinner=False)
def load_cnn_lstm_model():
    network_path = MODEL_ROOT / "cnn_lstm_svm" / "cnn_lstm.keras"
    head_path = MODEL_ROOT / "cnn_lstm_svm" / "cnn_lstm_svm_head.joblib"
    if not network_path.exists() or not head_path.exists():
        return None
    try:
        from tensorflow.keras.models import Model, load_model
    except ImportError:
        return None
    network = load_model(network_path)
    bundle = joblib.load(head_path)
    extractor = Model(network.input, network.get_layer("features").output)
    return extractor, bundle


@st.cache_resource(show_spinner=False)
def load_distilbert_model():
    model_path = MODEL_ROOT / "distilbert" / "model"
    if not (model_path / "config.json").exists():
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


def available_predictors():
    predictors = {}
    if (MODEL_ROOT / "svm" / "tfidf_svm.joblib").exists():
        predictors["TF-IDF + SVM"] = "tfidf"
    if (
        (MODEL_ROOT / "svm" / "w2v_svm.joblib").exists()
        and (MODEL_ROOT / "svm" / "word2vec_sg300.joblib").exists()
    ):
        predictors["Word2Vec + SVM"] = "word2vec"
    if (
        (MODEL_ROOT / "cnn_lstm_svm" / "cnn_lstm.keras").exists()
        and (MODEL_ROOT / "cnn_lstm_svm" / "cnn_lstm_svm_head.joblib").exists()
    ):
        predictors["CNN-LSTM + SVM"] = "cnn_lstm"
    distilbert_runtime_available = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )
    if (
        (MODEL_ROOT / "distilbert" / "model" / "config.json").exists()
        and distilbert_runtime_available
    ):
        predictors["DistilBERT"] = "distilbert"
    return predictors


def predict_tfidf(text):
    model = load_tfidf_model()
    if model is None:
        raise RuntimeError("TF-IDF model artifact was not found.")
    return str(model.predict([preprocess(text)])[0]), None


def predict_word2vec(text):
    loaded = load_word2vec_model()
    if loaded is None:
        raise RuntimeError("Word2Vec-SVM artifacts were not found.")
    model, word2vec = loaded
    cleaned = preprocess(text)
    vectors = [
        word2vec.wv[word]
        for word in cleaned.split()
        if word in word2vec.wv
    ]
    document_vector = (
        np.mean(vectors, axis=0)
        if vectors
        else np.zeros(word2vec.vector_size)
    )
    return str(model.predict(document_vector.reshape(1, -1))[0]), None


def predict_cnn_lstm(text):
    loaded = load_cnn_lstm_model()
    if loaded is None:
        raise RuntimeError(
            "CNN-LSTM-SVM artifacts were not found or TensorFlow is unavailable."
        )
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    extractor, bundle = loaded
    sequence = pad_sequences(
        bundle["tokenizer"].texts_to_sequences([preprocess(text)]),
        maxlen=bundle["maxlen"],
        padding="post",
        truncating="post",
    )
    features = extractor.predict(sequence, verbose=0)
    prediction_id = int(bundle["svm"].predict(features)[0])
    return str(bundle["id_to_label"][prediction_id]), None


def predict_distilbert(text):
    loaded = load_distilbert_model()
    if loaded is None:
        raise RuntimeError(
            "DistilBERT artifacts were not found or Transformers/PyTorch are unavailable."
        )
    import torch

    tokenizer, model, device = loaded
    encoded = tokenizer(
        light_clean(text),
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
    values = probabilities.detach().cpu().numpy()
    prediction_id = int(values.argmax())
    label = str(model.config.id2label[prediction_id])
    scores = {
        str(model.config.id2label[index]): float(probability)
        for index, probability in enumerate(values)
    }
    return label, scores


def classify_text(model_kind, text):
    functions = {
        "tfidf": predict_tfidf,
        "word2vec": predict_word2vec,
        "cnn_lstm": predict_cnn_lstm,
        "distilbert": predict_distilbert,
    }
    return functions[model_kind](text)


def reset_live_prediction():
    """Clear the live-prediction input and output widgets."""
    st.session_state["live_comment"] = ""
    st.session_state.pop("live_prediction", None)
    st.session_state.pop("live_probabilities", None)
    st.session_state.pop("live_model_name", None)


def reset_filters():
    """Reset sidebar filters to the full corpus."""
    st.session_state["sentiment_filter"] = LABELS
    st.session_state["theme_filter"] = []
    if "default_date_range" in st.session_state:
        st.session_state["date_filter"] = st.session_state["default_date_range"]


def show_user_error(message, error):
    """Show a presentation-safe error message with optional technical detail."""
    st.error(message)
    with st.expander("Technical details"):
        st.exception(error)


st.title("Public Sentiment on Zambia's 2025 Cyber Security Legislation")
st.caption(
    "Aggregate analysis of 3,472 anonymised Facebook annotated comments."
)

if not DATA_PATH.exists():
    st.error(f"Processed corpus not found: {DATA_PATH}")
    st.stop()

try:
    corpus = load_corpus(DATA_PATH)
except Exception as error:
    show_user_error("The processed corpus could not be loaded.", error)
    st.stop()


st.sidebar.header("Filters")
dated = corpus["date_parsed"].dropna()
selected_dates = None
if not dated.empty:
    minimum_date = dated.min().date()
    maximum_date = dated.max().date()
    st.session_state.setdefault("default_date_range", (minimum_date, maximum_date))

st.sidebar.button("Reset all filters", on_click=reset_filters)
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    LABELS,
    default=LABELS,
    key="sentiment_filter",
)
selected_themes = st.sidebar.multiselect(
    "Theme contains any",
    list(THEMES),
    key="theme_filter",
)

if not dated.empty:
    selected_dates = st.sidebar.date_input(
        "Comment date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
        key="date_filter",
    )

filtered = corpus[corpus["sentiment"].isin(selected_sentiments)].copy()
if selected_themes:
    filtered = filtered[filtered[selected_themes].any(axis=1)]
if selected_dates and isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    local_dates = filtered["date_parsed"].dt.date
    filtered = filtered[local_dates.between(start_date, end_date)]


about_tab, overview_tab, themes_tab, models_tab, topics_tab, predictor_tab = st.tabs(
    ["About", "Overview", "Themes", "Models", "Topics", "Live prediction"]
)


with about_tab:
    st.subheader("What this dashboard demonstrates")
    st.write(
        "This dashboard summarises a thesis NLP pipeline for analysing public "
        "Facebook-comment sentiment toward Zambia's 2025 cyber security "
        "legislation. It is designed for aggregate analysis and model "
        "demonstration, and was not used to profile individuals."
    )

    columns = st.columns(4)
    columns[0].metric("Raw Apify records", f"{RAW_RECORDS:,}")
    columns[1].metric("Valid comments", f"{VALID_EXTRACTED_COMMENTS:,}")
    columns[2].metric("Keyword-eligible", f"{KEYWORD_ELIGIBLE_COMMENTS:,}")
    columns[3].metric("Final corpus", f"{FINAL_CORPUS_COMMENTS:,}")

    st.markdown("#### Methodology snapshot")
    methodology_rows = pd.DataFrame(
        [
            {
                "Stage": "Post discovery",
                "Implementation": (
                    "Month-by-month Google searches with custom date filters "
                    "for public Facebook posts discussing cyber security in Zambia."
                ),
            },
            {
                "Stage": "Comment extraction",
                "Implementation": (
                    "Apify Facebook Comments Scraper; parent post links were "
                    "tracked in the sampling spreadsheet."
                ),
            },
            {
                "Stage": "Annotation target",
                "Implementation": (
                    "The researcher's final labeling scheme was three-way sentiment: NEGATIVE, NEUTRAL, POSITIVE. "
                    "sentiment label."
                ),
            },
            {
                "Stage": "Model split",
                "Implementation": SPLIT_DESCRIPTION,
            },
            {
                "Stage": "Model selection",
                "Implementation": (
                    "Macro-F1 on the same held-out test set; TF-IDF + SVM was "
                    "the strongest candidate."
                ),
            },
            {
                "Stage": "Topic modelling",
                "Implementation": (
                    "LDA is exploratory and manually interpreted; it is not "
                    "used to score supervised sentiment models."
                ),
            },
        ]
    )
    st.dataframe(methodology_rows, width="stretch", hide_index=True)

    st.markdown("#### Important limitations")
    st.info(
        "The corpus is Facebook-only and purposively sampled. Google ranking, "
        "public-page visibility, keyword filtering, comment availability, "
        "sarcasm, and code-switching can all affect what is represented. "
        "The findings should therefore be interpreted as evidence from the "
        "collected Facebook discourse, not as a nationally representative survey."
    )


with overview_tab:
    total = len(filtered)
    columns = st.columns(4)
    columns[0].metric("Filtered comments", f"{total:,}")
    for column, label in zip(columns[1:], LABELS):
        share = filtered["sentiment"].eq(label).mean() * 100 if total else 0
        column.metric(label.title(), f"{share:.1f}%")

    left, right = st.columns(2)
    with left:
        counts = (
            filtered["sentiment"]
            .value_counts()
            .reindex(LABELS, fill_value=0)
            .rename_axis("sentiment")
            .reset_index(name="comments")
        )
        figure = px.bar(
            counts,
            x="sentiment",
            y="comments",
            color="sentiment",
            color_discrete_map=COLORS,
            text_auto=",",
            title="Sentiment distribution",
        )
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, width="stretch")

    with right:
        temporal = filtered.dropna(subset=["date_parsed"]).copy()
        if temporal.empty:
            st.info("No dated comments match the current filters.")
        else:
            temporal["week"] = (
                temporal["date_parsed"]
                .dt.tz_convert(None)
                .dt.to_period("W")
                .dt.start_time
            )
            weekly = (
                temporal.groupby(["week", "sentiment"])
                .size()
                .reset_index(name="comments")
            )
            figure = px.line(
                weekly,
                x="week",
                y="comments",
                color="sentiment",
                color_discrete_map=COLORS,
                markers=True,
                category_orders={"sentiment": LABELS},
                title="Weekly sentiment volume",
            )
            st.plotly_chart(figure, width="stretch")

    st.subheader("Comment explorer")
    display_columns = [
        column
        for column in ["date_parsed", "sentiment", "text_display", "postTitle", "_doc_id"]
        if column in filtered.columns
    ]
    st.dataframe(
        filtered[display_columns].rename(
            columns={"date_parsed": "date", "text_display": "comment"}
        ),
        width="stretch",
        height=380,
        hide_index=True,
    )
    if not filtered.empty and "_doc_id" in filtered.columns:
        st.caption(
            "For long comments, select a record below and open it in the full-text viewer."
        )
        explorer = filtered.copy()
        explorer["comment_preview"] = (
            explorer["text_display"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.slice(0, 140)
        )
        explorer["comment_picker_label"] = (
            explorer["_doc_id"].astype(str)
            + " | "
            + explorer["sentiment"].astype(str)
            + " | "
            + explorer["comment_preview"]
        )
        selected_comment_label = st.selectbox(
            "Select a comment to read fully",
            explorer["comment_picker_label"].tolist(),
            index=None,
            placeholder="Choose a comment...",
        )

        def show_full_comment(row):
            st.write(f"Document ID: `{row['_doc_id']}`")
            st.write(f"Sentiment: `{row['sentiment']}`")
            if pd.notna(row.get("date_parsed")):
                st.write(f"Date: {row['date_parsed']}")
            if pd.notna(row.get("postTitle")):
                st.write("Parent post:")
                st.info(str(row["postTitle"]))
            st.text_area(
                "Full comment",
                value=str(row.get("text_display", "")),
                height=320,
                disabled=True,
            )

        if selected_comment_label:
            selected_row = explorer[
                explorer["comment_picker_label"].eq(selected_comment_label)
            ].iloc[0]
            if hasattr(st, "dialog"):
                @st.dialog("Full comment")
                def full_comment_dialog():
                    show_full_comment(selected_row)

                if st.button("Open full comment"):
                    full_comment_dialog()
            else:
                with st.expander("Full selected comment", expanded=True):
                    show_full_comment(selected_row)


with themes_tab:
    if filtered.empty:
        st.info("No comments match the current filters.")
    else:
        theme_counts = pd.DataFrame({
            "theme": list(THEMES),
            "comments": [int(filtered[theme].sum()) for theme in THEMES],
        }).sort_values("comments", ascending=False)
        theme_counts["percentage"] = theme_counts["comments"] / len(filtered) * 100

        figure = px.bar(
            theme_counts,
            x="comments",
            y="theme",
            orientation="h",
            color="comments",
            color_continuous_scale=GREEN_SCALE,
            text_auto=",",
            title="Theme prevalence",
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figure, width="stretch")

        rows = []
        for theme in THEMES:
            subset = filtered[filtered[theme]]
            for label in LABELS:
                rows.append({
                    "theme": theme,
                    "sentiment": label,
                    "comments": int(subset["sentiment"].eq(label).sum()),
                })
        figure = px.bar(
            pd.DataFrame(rows),
            x="theme",
            y="comments",
            color="sentiment",
            color_discrete_map=COLORS,
            category_orders={"sentiment": LABELS},
            barmode="stack",
            title="Sentiment composition by theme",
        )
        figure.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(figure, width="stretch")


with models_tab:
    metrics = load_csv_if_present(COMPARISON_DIR / "comparison_metrics.csv")
    if metrics is None or metrics.empty:
        st.info(
            "Model comparison results are unavailable. "
            "Run `python src/06_evaluate_compare.py` after training/importing models."
        )
    else:
        st.dataframe(metrics.round(3), width="stretch", hide_index=True)
        if {"model", "macro_f1"}.issubset(metrics.columns):
            best_model = metrics.sort_values("macro_f1", ascending=False).iloc[0]
            st.success(
                f"Selected model: **{best_model['model']}** "
                f"(held-out macro-F1 = {best_model['macro_f1']:.3f})."
            )
            st.caption(
                "Macro-F1 is used because it gives equal importance to NEGATIVE, "
                "NEUTRAL and POSITIVE performance, which is more appropriate than "
                "accuracy alone for an imbalanced sentiment corpus."
            )
        main_metrics = metrics.melt(
            id_vars="model",
            value_vars=["accuracy", "macro_f1"],
            var_name="metric",
            value_name="score",
        )
        figure = px.bar(
            main_metrics,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            color_discrete_sequence=GREEN_SEQUENCE[:2],
            text_auto=".3f",
            range_y=[0, 1],
            title="Held-out test performance",
        )
        st.plotly_chart(figure, width="stretch")

        class_columns = [
            column
            for column in ["f1_negative", "f1_neutral", "f1_positive"]
            if column in metrics.columns
        ]
        if class_columns:
            class_metrics = metrics.melt(
                id_vars="model",
                value_vars=class_columns,
                var_name="class",
                value_name="f1",
            )
            class_metrics["class"] = (
                class_metrics["class"]
                .str.replace("f1_", "", regex=False)
                .str.upper()
            )
            figure = px.bar(
                class_metrics,
                x="model",
                y="f1",
                color="class",
                barmode="group",
                color_discrete_map=COLORS,
                text_auto=".3f",
                range_y=[0, 1],
                title="Per-class F1 scores",
            )
            st.plotly_chart(figure, width="stretch")

        confusion_image = COMPARISON_DIR / "comparison_confusion_matrices.png"
        if confusion_image.exists():
            st.image(str(confusion_image), caption="Held-out confusion matrices")

        st.markdown("#### Interpretation")
        st.write(
            "TF-IDF + SVM is the preferred deployment model because it achieved "
            "the highest held-out macro-F1 while remaining lightweight, fast and "
            "easier to explain than the neural candidates. DistilBERT remained "
            "competitive, but the simpler sparse model performed better on this "
            "relatively small, domain-specific corpus."
        )


with topics_tab:
    topics = load_csv_if_present(LDA_DIR / "topics_named.csv")
    if topics is None:
        topics = load_csv_if_present(LDA_DIR / "topics.csv")
    assignments = load_csv_if_present(LDA_DIR / "document_topics_named.csv")
    if assignments is None:
        assignments = load_csv_if_present(LDA_DIR / "document_topics.csv")
    topic_sentiment = load_csv_if_present(LDA_DIR / "topic_sentiment_named.csv")
    if topic_sentiment is None:
        topic_sentiment = load_csv_if_present(LDA_DIR / "topic_sentiment_counts.csv")

    if topics is None or assignments is None:
        st.info("LDA outputs are unavailable. Run `python src/07_lda_topics.py`.")
    else:
        st.subheader("Topic keywords")
        topic_name_map = {}
        if {"topic", "topic_name"}.issubset(topics.columns):
            topic_name_map = topics.set_index("topic")["topic_name"].to_dict()
            topic_columns = [
                column
                for column in ["topic_name", "top_words", "topic"]
                if column in topics.columns
            ]
            st.dataframe(
                topics[topic_columns],
                width="stretch",
                hide_index=True,
                column_config={"topic": "Topic ID"},
            )
        else:
            st.dataframe(topics, width="stretch", hide_index=True)

        if "topic_name" not in assignments.columns and topic_name_map:
            assignments["topic_name"] = assignments["dominant_topic"].map(
                topic_name_map
            )
        prevalence_column = (
            "topic_name" if "topic_name" in assignments.columns else "dominant_topic"
        )
        prevalence = (
            assignments[prevalence_column]
            .value_counts()
            .rename_axis("topic_name")
            .reset_index(name="comments")
        )
        figure = px.bar(
            prevalence,
            x="topic_name",
            y="comments",
            color="comments",
            color_continuous_scale=GREEN_SCALE,
            text_auto=",",
            title="Dominant topic prevalence",
            labels={"topic_name": "Topic", "comments": "Comments"},
        )
        figure.update_xaxes(tickangle=-25)
        st.plotly_chart(figure, width="stretch")

        if topic_sentiment is not None:
            if "topic_name" not in topic_sentiment.columns and topic_name_map:
                topic_id_column = next(
                    (
                        column
                        for column in ["dominant_topic", "topic"]
                        if column in topic_sentiment.columns
                    ),
                    None,
                )
                if topic_id_column is not None:
                    topic_sentiment["topic_name"] = topic_sentiment[
                        topic_id_column
                    ].map(topic_name_map)

            topic_column = (
                "topic_name"
                if "topic_name" in topic_sentiment.columns
                else topic_sentiment.columns[0]
            )
            sentiment_columns = [
                label for label in LABELS if label in topic_sentiment.columns
            ]
            sentiment_long = topic_sentiment.melt(
                id_vars=topic_column,
                value_vars=sentiment_columns,
                var_name="sentiment",
                value_name="comments",
            ).rename(columns={topic_column: "topic_name"})
            figure = px.bar(
                sentiment_long,
                x="topic_name",
                y="comments",
                color="sentiment",
                color_discrete_map=COLORS,
                category_orders={"sentiment": LABELS},
                barmode="stack",
                title="Topic and sentiment",
                labels={"topic_name": "Topic", "comments": "Comments"},
            )
            figure.update_xaxes(tickangle=-25)
            st.plotly_chart(figure, width="stretch")

        st.subheader("Representative comments by topic")
        st.caption(
            "These examples are selected from comments where LDA assigned a "
            "relatively high probability to the selected dominant topic."
        )
        topic_options = prevalence["topic_name"].tolist()
        if not topic_options:
            st.info("No topic assignments are available for representative examples.")
        else:
            selected_topic = st.selectbox(
                "Choose a topic",
                topic_options,
                key="representative_topic",
            )
            topic_examples = assignments[
                assignments[prevalence_column].astype(str).eq(str(selected_topic))
            ].copy()
            if "topic_probability" in topic_examples.columns:
                topic_examples = topic_examples.sort_values(
                    "topic_probability",
                    ascending=False,
                )
            example_columns = [
                column
                for column in [
                    "_doc_id",
                    "label",
                    "topic_probability",
                    "text_original",
                ]
                if column in topic_examples.columns
            ]
            if topic_examples.empty or not example_columns:
                st.info("No representative examples are available for this topic.")
            else:
                examples = topic_examples[example_columns].head(5).rename(
                    columns={
                        "_doc_id": "Document ID",
                        "label": "Sentiment",
                        "topic_probability": "Topic probability",
                        "text_original": "Comment",
                    }
                )
                st.dataframe(examples, width="stretch", hide_index=True)

        visualisation = LDA_DIR / "lda_visualisation.html"
        if visualisation.exists():
            st.subheader("Interactive LDA visualisation")
            st.caption(
                "Select a topic bubble to inspect its most relevant terms. "
                "Use the relevance slider to change the term ranking."
            )

            visualisation_html = visualisation.read_text(encoding="utf-8")
            topic_order_match = re.search(
                r'"topic\.order"\s*:\s*\[([^\]]+)\]',
                visualisation_html,
            )
            if topic_order_match and topic_name_map:
                topic_order = [
                    int(value.strip())
                    for value in topic_order_match.group(1).split(",")
                ]
                pyldavis_key = pd.DataFrame(
                    {
                        "pyLDAvis topic": range(1, len(topic_order) + 1),
                        "Topic name": [
                            topic_name_map.get(
                                original_topic - 1,
                                f"Model topic {original_topic - 1}",
                            )
                            for original_topic in topic_order
                        ],
                    }
                )
                st.markdown("**pyLDAvis topic key**")
                st.dataframe(pyldavis_key, width="stretch", hide_index=True)

                key_rows = "".join(
                    "<tr>"
                    f"<td style='padding:4px 12px'>{row['pyLDAvis topic']}</td>"
                    f"<td style='padding:4px 12px'>{row['Topic name']}</td>"
                    "</tr>"
                    for _, row in pyldavis_key.iterrows()
                )
                embedded_key = (
                    "<div style='font-family:Arial,sans-serif;margin:8px 20px 18px'>"
                    "<h3>Topic key</h3>"
                    "<table style='border-collapse:collapse'>"
                    "<tr><th style='text-align:left;padding:4px 12px'>"
                    "pyLDAvis topic</th>"
                    "<th style='text-align:left;padding:4px 12px'>"
                    "Descriptive name</th></tr>"
                    f"{key_rows}</table></div>"
                )
                visualisation_html = visualisation_html.replace(
                    "<body>",
                    f"<body>{embedded_key}",
                    1,
                )

            visualisation_html = apply_pyldavis_green_theme(visualisation_html)

            with st.expander("Open interactive topic map", expanded=True):
                encoded_html = base64.b64encode(
                    visualisation_html.encode("utf-8")
                ).decode("ascii")
                st.iframe(
                    f"data:text/html;base64,{encoded_html}",
                    height=1050,
                )
                st.caption(
                    "In the pyLDAvis map, the small chart at the bottom-left "
                    "is the marginal topic distribution: bubble area represents "
                    "how prevalent each topic is in the LDA corpus."
                )
            st.download_button(
                "Download interactive LDA visualisation",
                data=visualisation_html.encode("utf-8"),
                file_name="lda_visualisation.html",
                mime="text/html",
            )


with predictor_tab:
    st.info(
        "Live prediction is a demonstration of the trained models. Treat the "
        "output as model evidence, not as a definitive judgment of public opinion. "
        "Sarcasm, code-switching, very short comments and missing parent-post "
        "context can affect predictions."
    )
    predictors = available_predictors()
    if not predictors:
        st.warning("No model artifacts are available. Train the TF-IDF baseline first.")
    else:
        model_name = st.selectbox("Prediction model", list(predictors))
        comment = st.text_area(
            "Enter a comment about the legislation",
            height=140,
            placeholder="Example: This law will protect citizens from online abuse.",
            key="live_comment",
        )
        left, right = st.columns([1, 1])
        classify_clicked = left.button("Classify comment", type="primary")
        right.button("Reset", on_click=reset_live_prediction)

        if classify_clicked:
            if not comment.strip():
                st.warning("Enter a comment before classifying.")
            else:
                try:
                    prediction, probabilities = classify_text(
                        predictors[model_name],
                        comment,
                    )
                except Exception as error:
                    show_user_error(
                        "The selected model could not classify this comment.",
                        error,
                    )
                else:
                    st.session_state["live_prediction"] = prediction
                    st.session_state["live_probabilities"] = probabilities
                    st.session_state["live_model_name"] = model_name

        if "live_prediction" in st.session_state:
            st.subheader(
                f"Predicted sentiment: {st.session_state['live_prediction']}"
            )
            st.caption(f"Model: {st.session_state.get('live_model_name', model_name)}")
            probabilities = st.session_state.get("live_probabilities")
            if probabilities:
                probability_frame = pd.DataFrame({
                    "sentiment": LABELS,
                    "probability": [
                        probabilities.get(label, 0.0)
                        for label in LABELS
                    ],
                })
                figure = px.bar(
                    probability_frame,
                    x="sentiment",
                    y="probability",
                    color="sentiment",
                    color_discrete_map=COLORS,
                    range_y=[0, 1],
                    text_auto=".3f",
                )
                figure.update_layout(showlegend=False)
                st.plotly_chart(figure, width="stretch")
            else:
                st.caption(
                    "This classifier does not provide calibrated probabilities."
                )

st.divider()
st.caption(
    "Research dashboard for aggregate governance analysis. "
    "The corpus is anonymised and must not be used to profile individuals."
)
