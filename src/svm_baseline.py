"""
svm_baseline.py
Candidate 1 (SVM) for the Cyber Security sentiment thesis.

Implements TWO feature representations so you can report a comparison:
  A) TF-IDF  + Linear SVM   (conventional strong baseline)
  B) Word2Vec (mean-pooled) + SVM   (the representation specified in your thesis)

Protocol: the shared stratified 70/15/15 train/validation/test split created by
prepare_split.py, stratified 10-fold CV over the combined development set
(train + validation), class-weight balancing, macro-F1 model selection, and one
final evaluation on the untouched test set.

Run from the repository root:
    python src/03_svm_baseline.py --model tfidf
    python src/03_svm_baseline.py --model w2v
    python src/03_svm_baseline.py --model both
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_split.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "svm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text_clean"
LABEL_COL = "label"
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
CV_FOLDS = 10
RANDOM = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SVM sentiment baselines.")
    parser.add_argument(
        "--model",
        choices=("tfidf", "w2v", "both"),
        default="both",
        help="Feature representation to train (default: both).",
    )
    return parser.parse_args()


def load_split():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH} does not exist. Run: python src/02_prepare_split.py"
        )

    df = pd.read_csv(SPLIT_PATH)
    assert "split" in df.columns, "Run prepare_split.py first to create corpus_split.csv"
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

    print(f"Corpus used: {len(df):,}")
    print(df.groupby(["split", LABEL_COL]).size().unstack(fill_value=0), "\n")
    return df


def main():
    args = parse_args()
    df = load_split()

    # shared canonical split: train+val used for CV/training, test held out
    tr = df[df["split"].isin(["train", "val"])]
    te = df[df["split"] == "test"]
    # Explicit NumPy conversion avoids scikit-learn indexing failures with
    # pandas Arrow-backed string arrays during cross-validation.
    X_tr = tr[TEXT_COL].to_numpy(dtype=str)
    y_tr = tr[LABEL_COL].to_numpy(dtype=str)
    X_te = te[TEXT_COL].to_numpy(dtype=str)
    y_te = te[LABEL_COL].to_numpy(dtype=str)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM)

    if args.model in {"tfidf", "both"}:
        train_tfidf_svm(X_tr, y_tr, X_te, y_te, cv)

    if args.model in {"w2v", "both"}:
        train_word2vec_svm(X_tr, y_tr, X_te, y_te, cv)


def train_tfidf_svm(X_tr, y_tr, X_te, y_te, cv):
    print("=" * 60, "\nA) TF-IDF + Linear SVM\n", "=" * 60)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=2)),
        ("svm", LinearSVC(class_weight="balanced", random_state=RANDOM)),
    ])
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [10000, 20000, None],
        "svm__C": [0.1, 1, 10],
    }
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_tr, y_tr)
    print("Best params:", search.best_params_)
    print("Best CV macro-F1: %.3f" % search.best_score_)
    evaluate(search.best_estimator_, X_te, y_te, "tfidf_svm")


def train_word2vec_svm(X_tr, y_tr, X_te, y_te, cv):
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise ImportError(
            "Word2Vec requires gensim. Install it with: pip install gensim"
        ) from exc

    print("\n" + "=" * 60, "\nB) Word2Vec (averaged) + SVM\n", "=" * 60)
    tokenised_train = [sentence.split() for sentence in X_tr]
    word2vec = Word2Vec(
        sentences=tokenised_train,
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,
        epochs=30,
        workers=1,
        seed=RANDOM,
    )

    def document_vector(sentence):
        vectors = [
            word2vec.wv[word]
            for word in sentence.split()
            if word in word2vec.wv
        ]
        return (
            np.mean(vectors, axis=0)
            if vectors
            else np.zeros(word2vec.vector_size)
        )

    Xtr_vec = np.vstack([document_vector(sentence) for sentence in X_tr])
    Xte_vec = np.vstack([document_vector(sentence) for sentence in X_te])

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(class_weight="balanced", random_state=RANDOM)),
    ])
    grid = [
        {"svm__kernel": ["linear"], "svm__C": [1, 10, 100]},
        {
            "svm__kernel": ["rbf"],
            "svm__C": [1, 10, 100],
            "svm__gamma": ["scale", 0.01],
        },
    ]
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(Xtr_vec, y_tr)
    print("Best params:", search.best_params_)
    print("Best CV macro-F1: %.3f" % search.best_score_)
    evaluate(search.best_estimator_, Xte_vec, y_te, "w2v_svm")
    joblib.dump(word2vec, MODEL_DIR / "word2vec_sg300.joblib")


def evaluate(model, X_te, y_te, tag):
    y_pred = model.predict(X_te)
    print("\nHeld-out test set — %s" % tag)
    print("Macro-F1: %.3f" % f1_score(y_te, y_pred, average="macro"))
    print(classification_report(y_te, y_pred, labels=LABELS, digits=3, zero_division=0))
    matrix = confusion_matrix(y_te, y_pred, labels=LABELS)
    print("Confusion matrix (rows=true, cols=pred):", LABELS)
    print(matrix)

    joblib.dump(model, MODEL_DIR / f"{tag}.joblib")
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        MODEL_DIR / f"{tag}_confusion_matrix.csv"
    )


if __name__ == "__main__":
    main()
