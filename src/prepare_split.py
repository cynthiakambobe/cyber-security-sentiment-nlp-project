"""
02_prepare_split.py / prepare_split.py
Create ONE canonical train/val/test split shared by all three candidate models,
so the Chapter 4 comparison is fair (every model is tested on the exact same
unseen rows, with no leakage).

Run this ONCE, before training. It writes corpus_split.csv with a 'split' column.

Then make each training script use it (2-line change) instead of its own
train_test_split, e.g.:

    df = pd.read_csv("corpus_split.csv")
    tr  = df[df.split=="train"]; val = df[df.split=="val"]; te = df[df.split=="test"]
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_preprocessed_3472.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_split.csv"
LABEL_COL = "label"
RANDOM = 42

df = pd.read_csv(CSV_PATH)
df = df[df[LABEL_COL].isin(["POSITIVE","NEGATIVE","NEUTRAL"])].reset_index(drop=True)

idx = df.index.values
tr, tmp = train_test_split(idx, test_size=0.30, stratify=df[LABEL_COL], random_state=RANDOM)
val, te = train_test_split(tmp, test_size=0.50, stratify=df.loc[tmp, LABEL_COL], random_state=RANDOM)

df["split"] = "train"
df.loc[val, "split"] = "val"
df.loc[te,  "split"] = "test"

df.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {OUTPUT_PATH}")
print(df.groupby(["split", LABEL_COL]).size().unstack(fill_value=0))
