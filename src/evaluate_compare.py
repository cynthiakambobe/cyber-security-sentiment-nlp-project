"""
Compare trained sentiment candidates on the canonical held-out test set.

The script uses the same `_doc_id` values and final human labels for every model.
DistilBERT predictions are read from the Colab artifact package, so the
transformer does not need to be rerun locally.

Run from the repository root:
    python src/06_evaluate_compare.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_split.csv"
MODEL_ROOT = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "model_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
TEXT_COL = "text_clean"
LABEL_COL = "label"


def load_test_set():
    """Load the canonical held-out test records."""
    df = pd.read_csv(SPLIT_PATH)
    required = {"_doc_id", TEXT_COL, LABEL_COL, "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    test = df[df["split"].eq("test")].copy()
    test[TEXT_COL] = test[TEXT_COL].fillna("").astype(str).str.strip()
    if test["_doc_id"].duplicated().any():
        raise ValueError("Canonical test set contains duplicated _doc_id values.")
    return test


def predict_tfidf(test):
    path = MODEL_ROOT / "svm" / "tfidf_svm.joblib"
    if not path.exists():
        return None
    model = joblib.load(path)
    return model.predict(test[TEXT_COL].to_numpy(dtype=str))


def predict_word2vec(test):
    model_path = MODEL_ROOT / "svm" / "w2v_svm.joblib"
    vectors_path = MODEL_ROOT / "svm" / "word2vec_sg300.joblib"
    if not model_path.exists() or not vectors_path.exists():
        return None

    model = joblib.load(model_path)
    word2vec = joblib.load(vectors_path)

    def document_vector(sentence):
        vectors = [
            word2vec.wv[word]
            for word in sentence.split()
            if word in word2vec.wv
        ]
        return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec.vector_size)

    matrix = np.vstack([document_vector(text) for text in test[TEXT_COL]])
    return model.predict(matrix)


def predict_cnn_lstm_svm(test):
    model_path = MODEL_ROOT / "cnn_lstm_svm" / "cnn_lstm.keras"
    head_path = MODEL_ROOT / "cnn_lstm_svm" / "cnn_lstm_svm_head.joblib"
    if not model_path.exists() or not head_path.exists():
        return None

    try:
        from tensorflow.keras.models import Model, load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        print("Skipping CNN-LSTM + SVM: TensorFlow is not installed.")
        return None

    network = load_model(model_path)
    bundle = joblib.load(head_path)
    tokenizer = bundle["tokenizer"]
    svm = bundle["svm"]
    maxlen = bundle["maxlen"]
    id_to_label = bundle["id_to_label"]

    sequences = pad_sequences(
        tokenizer.texts_to_sequences(test[TEXT_COL].to_numpy(dtype=str)),
        maxlen=maxlen,
        padding="post",
        truncating="post",
    )
    extractor = Model(network.input, network.get_layer("features").output)
    features = extractor.predict(sequences, verbose=0)
    prediction_ids = svm.predict(features)
    return np.array([id_to_label[int(index)] for index in prediction_ids])


def load_distilbert_predictions(test):
    path = MODEL_ROOT / "distilbert" / "test_predictions.csv"
    if not path.exists():
        return None

    predictions = pd.read_csv(path)
    required = {"_doc_id", "predicted_label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"DistilBERT predictions missing columns: {sorted(missing)}")
    if predictions["_doc_id"].duplicated().any():
        raise ValueError("DistilBERT prediction file contains duplicated _doc_id values.")

    aligned = test[["_doc_id"]].merge(
        predictions[["_doc_id", "predicted_label"]],
        on="_doc_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["predicted_label"].isna().any():
        missing_count = int(aligned["predicted_label"].isna().sum())
        raise ValueError(f"DistilBERT predictions are missing {missing_count} test records.")
    return aligned["predicted_label"].to_numpy(dtype=str)


def metrics_row(model_name, y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_negative": report["NEGATIVE"]["f1-score"],
        "f1_neutral": report["NEUTRAL"]["f1-score"],
        "f1_positive": report["POSITIVE"]["f1-score"],
    }


def save_outputs(test, predictions):
    """Save comparison tables, aligned predictions, and confusion matrices."""
    y_true = test[LABEL_COL].to_numpy(dtype=str)
    rows = []
    aligned = test[["_doc_id", LABEL_COL]].rename(columns={LABEL_COL: "true_label"})

    for model_name, y_pred in predictions.items():
        rows.append(metrics_row(model_name, y_true, y_pred))
        aligned[model_name] = y_pred

    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False)
    aligned.to_csv(RESULTS_DIR / "aligned_test_predictions.csv", index=False)

    count = len(predictions)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), squeeze=False)
    for axis, (model_name, y_pred) in zip(axes.ravel(), predictions.items()):
        matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
        sns.heatmap(
            pd.DataFrame(matrix, index=LABELS, columns=LABELS),
            annot=True,
            fmt="d",
            cmap="Greens",
            cbar=False,
            linewidths=0.5,
            ax=axis,
        )
        score = f1_score(y_true, y_pred, average="macro")
        axis.set_title(f"{model_name}\nMacro-F1 = {score:.3f}")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(metrics.round(3).to_string(index=False))
    print(f"\nOutputs saved to {RESULTS_DIR}")
    return metrics


def main():
    test = load_test_set()
    candidates = [
        ("TF-IDF + SVM", predict_tfidf(test)),
        ("Word2Vec + SVM", predict_word2vec(test)),
        ("CNN-LSTM + SVM", predict_cnn_lstm_svm(test)),
        ("DistilBERT", load_distilbert_predictions(test)),
    ]
    predictions = {
        name: prediction
        for name, prediction in candidates
        if prediction is not None
    }

    if not predictions:
        raise RuntimeError("No trained model artifacts or prediction files were found.")

    for name, prediction in predictions.items():
        if len(prediction) != len(test):
            raise ValueError(
                f"{name} produced {len(prediction)} predictions for {len(test)} test records."
            )
        print(f"Loaded {name}: {len(prediction)} predictions")

    save_outputs(test, predictions)


if __name__ == "__main__":
    main()
