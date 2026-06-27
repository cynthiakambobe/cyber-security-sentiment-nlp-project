"""
Candidate 2: Word2Vec -> CNN -> LSTM -> Linear SVM.

The neural network learns a 64-dimensional feature representation from the
canonical training split. A Linear SVM is then trained on those learned
features. The validation split is used only for early stopping; the test split
remains untouched until final evaluation.

Run from the repository root:
    python src/04_cnn_lstm_svm.py

TensorFlow is required. A GPU is recommended but not mandatory.
"""

import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        Conv1D,
        Dense,
        Dropout,
        Embedding,
        Input,
        LSTM,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required. Install it with: "
        "python -m pip install tensorflow"
    ) from exc


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_split.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "cnn_lstm_svm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text_clean"
LABEL_COL = "label"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

MAXLEN = 60
VOCAB = 20_000
EMBEDDING_DIM = 300
CNN_FILTERS = 128
KERNEL_SIZE = 5
LSTM_UNITS = 128
FEATURE_DIM = 64
DROPOUT = 0.4
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 3
RANDOM = 42


def set_reproducible_seeds(seed=RANDOM):
    """Set Python, NumPy, and TensorFlow seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_split():
    """Load and validate the canonical split."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} does not exist. Run: python src/02_prepare_split.py"
        )

    df = pd.read_csv(SPLIT_PATH)
    required = {TEXT_COL, LABEL_COL, "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required split columns: {sorted(missing)}")

    df = df[df[LABEL_COL].isin(LABELS)].copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()

    empty_count = int(df[TEXT_COL].eq("").sum())
    if empty_count:
        print(f"Dropping {empty_count} records with empty {TEXT_COL}.")
        df = df[df[TEXT_COL].ne("")].copy()

    df["label_id"] = df[LABEL_COL].map(LABEL_TO_ID).astype("int64")

    print(f"Corpus used: {len(df):,}")
    print(df.groupby(["split", LABEL_COL]).size().unstack(fill_value=0), "\n")
    return df


def prepare_sequences(df):
    """Fit the tokenizer on training text only and pad all partitions."""
    train = df[df["split"].eq("train")].copy()
    validation = df[df["split"].eq("val")].copy()
    test = df[df["split"].eq("test")].copy()

    tokenizer = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train[TEXT_COL].to_numpy(dtype=str))

    def encode(frame):
        texts = frame[TEXT_COL].to_numpy(dtype=str)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=MAXLEN,
            padding="post",
            truncating="post",
        )
        labels = frame["label_id"].to_numpy(dtype=np.int64)
        return padded, labels

    X_train, y_train = encode(train)
    X_validation, y_validation = encode(validation)
    X_test, y_test = encode(test)

    return (
        tokenizer,
        train,
        validation,
        test,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


def build_embedding_matrix(tokenizer, training_text):
    """Train skip-gram Word2Vec on training text and build the Keras matrix."""
    tokenised_training = [text.split() for text in training_text]
    word2vec = Word2Vec(
        sentences=tokenised_training,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=2,
        sg=1,
        epochs=30,
        workers=1,
        seed=RANDOM,
    )

    vocabulary_size = min(VOCAB, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros(
        (vocabulary_size, EMBEDDING_DIM),
        dtype=np.float32,
    )

    covered = 0
    for word, index in tokenizer.word_index.items():
        if index >= vocabulary_size:
            continue
        if word in word2vec.wv:
            embedding_matrix[index] = word2vec.wv[word]
            covered += 1

    eligible_words = min(len(tokenizer.word_index), vocabulary_size - 1)
    coverage = covered / eligible_words if eligible_words else 0.0
    print(
        f"Embedding coverage: {covered:,}/{eligible_words:,} "
        f"({coverage:.1%})"
    )

    return word2vec, embedding_matrix, vocabulary_size


def build_model(embedding_matrix, vocabulary_size):
    """Build and compile the CNN-LSTM feature extractor."""
    inputs = Input(shape=(MAXLEN,), name="tokens")
    x = Embedding(
        input_dim=vocabulary_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=True,
        mask_zero=False,
        name="word2vec_embedding",
    )(inputs)
    x = Conv1D(
        filters=CNN_FILTERS,
        kernel_size=KERNEL_SIZE,
        activation="relu",
        name="cnn",
    )(x)
    x = MaxPooling1D(pool_size=2, name="max_pool")(x)
    x = LSTM(LSTM_UNITS, name="lstm")(x)
    x = Dropout(DROPOUT, name="dropout")(x)
    features = Dense(
        FEATURE_DIM,
        activation="relu",
        name="features",
    )(x)
    outputs = Dense(len(LABELS), activation="softmax", name="softmax")(features)

    model = Model(inputs=inputs, outputs=outputs, name="cnn_lstm")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_predictions(y_true, y_pred, model_name):
    """Print and save classification metrics and confusion matrix."""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(LABELS)),
    )

    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(LABELS)),
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, cols=pred):", LABELS)
    print(matrix)

    slug = model_name.lower().replace(" ", "_").replace("+", "plus")
    pd.DataFrame(report).T.to_csv(MODEL_DIR / f"{slug}_report.csv")
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        MODEL_DIR / f"{slug}_confusion_matrix.csv"
    )

    return {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def main():
    set_reproducible_seeds()
    df = load_split()

    (
        tokenizer,
        train,
        _validation,
        _test,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_sequences(df)

    word2vec, embedding_matrix, vocabulary_size = build_embedding_matrix(
        tokenizer,
        train[TEXT_COL].to_numpy(dtype=str),
    )

    model = build_model(embedding_matrix, vocabulary_size)
    model.summary()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABELS)),
        y=y_train,
    )
    class_weight = {
        class_id: float(weight)
        for class_id, weight in enumerate(weights)
    }
    print("Class weights:", class_weight)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer("features").output,
        name="cnn_lstm_feature_extractor",
    )
    train_features = feature_extractor.predict(X_train, verbose=0)
    test_features = feature_extractor.predict(X_test, verbose=0)

    svm = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM,
    )
    svm.fit(train_features, y_train)

    softmax_predictions = model.predict(X_test, verbose=0).argmax(axis=1)
    svm_predictions = svm.predict(test_features)

    results = [
        evaluate_predictions(
            y_test,
            softmax_predictions,
            "CNN-LSTM softmax",
        ),
        evaluate_predictions(
            y_test,
            svm_predictions,
            "CNN-LSTM + SVM",
        ),
    ]

    model.save(MODEL_DIR / "cnn_lstm.keras")
    word2vec.save(str(MODEL_DIR / "word2vec.model"))
    joblib.dump(
        {
            "svm": svm,
            "tokenizer": tokenizer,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "maxlen": MAXLEN,
        },
        MODEL_DIR / "cnn_lstm_svm_head.joblib",
    )
    pd.DataFrame(history.history).to_csv(
        MODEL_DIR / "training_history.csv",
        index=False,
    )
    with open(MODEL_DIR / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nArtifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
