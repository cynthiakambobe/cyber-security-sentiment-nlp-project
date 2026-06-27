"""
distilbert_finetune.py
Candidate 3 (Transformer): fine-tuned DistilBERT
for the Cyber Security sentiment thesis.

Design (matches Chapter 3): the pre-trained DistilBERT model is fine-tuned on the
domain-specific labelled corpus under the pre-train / fine-tune paradigm. A weighted
cross-entropy loss handles class imbalance; macro-F1 is the primary metric.

Minimal preprocessing only: transformers expect natural text, so we DO NOT remove
stop words or lemmatise (unlike the SVM/CNN candidates) — we only strip URLs/mentions.

NOTE ON LANGUAGE: 'distilbert-base-uncased' is English (matches your English-dominant
scope). Because the corpus contains residual Bemba/Nyanja code-switching, you may also
try 'distilbert-base-multilingual-cased' as a robustness check — same code, just change
MODEL_NAME. The architecture/design is unchanged.

GPU required for reasonable speed (Colab: Runtime > Change runtime type > GPU).
    pip install "transformers>=4.40" "datasets>=2.19" torch scikit-learn pandas numpy
"""

import re, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

# ----------------------------------------------------------------------
SPLIT_PATH = "corpus_split.csv"   # canonical split from prepare_split.py
TEXT_COL   = "text"
LABEL_COL  = "human_label"
MODEL_NAME = "distilbert-base-uncased"     # or "distilbert-base-multilingual-cased"
MAXLEN     = 128
EPOCHS     = 4
LR         = 2e-5
BATCH      = 16
RANDOM     = 42
OUTDIR     = "distilbert_cyber"

def light_clean(t):
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)   # urls
    t = re.sub(r"@\w+", " ", t)               # mentions
    return re.sub(r"\s+", " ", t).strip()

def main():
    df = pd.read_csv(SPLIT_PATH)
    assert "split" in df.columns, "Run prepare_split.py first to create corpus_split.csv"
    df = df[df[LABEL_COL].isin(["POSITIVE","NEGATIVE","NEUTRAL"])].copy()
    df["text_clean"] = df[TEXT_COL].apply(light_clean)
    df = df[df["text_clean"].str.len() > 0]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df[LABEL_COL])
    names = list(le.classes_)
    print("Corpus:", len(df)); print(df[LABEL_COL].value_counts(), "\n")

    # shared canonical split
    tr  = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    te  = df[df["split"] == "test"]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    def to_ds(d):
        ds = Dataset.from_pandas(d[["text_clean","label"]].rename(columns={"text_clean":"text"}),
                                 preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAXLEN), batched=True)
    ds_tr, ds_val, ds_te = to_ds(tr), to_ds(val), to_ds(te)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
        id2label={i:n for i,n in enumerate(names)},
        label2id={n:i for i,n in enumerate(names)})

    # class-weighted loss for imbalance
    cw = compute_class_weight("balanced", classes=np.unique(df["label"]), y=df["label"])
    weights = torch.tensor(cw, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def metrics(p):
        preds = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, preds, average="macro"),
                "accuracy": accuracy_score(p.label_ids, preds)}

    args = TrainingArguments(
        output_dir=OUTDIR, num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        weight_decay=0.01, logging_steps=50, seed=RANDOM, report_to="none")

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_val,
        tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics)

    trainer.train()

    # ---- held-out test evaluation ----
    pred = trainer.predict(ds_te)
    y_pred, y_true = pred.predictions.argmax(-1), pred.label_ids
    print("\n--- DistilBERT held-out test (Candidate 3 result) ---")
    print("Macro-F1: %.3f" % f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=names, digits=3))
    print("Confusion (rows=true, cols=pred)", names)
    print(confusion_matrix(y_true, y_pred))

    trainer.save_model(OUTDIR); tok.save_pretrained(OUTDIR)
    print("Saved to", OUTDIR)

if __name__ == "__main__":
    main()
