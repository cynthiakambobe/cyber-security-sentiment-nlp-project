# Streamlit Dashboard

Run from the repository root:

```powershell
conda activate .\venv
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard provides sentiment trends, theme analysis, model comparison,
topic-model results, anonymised comment exploration, and live prediction.

Required data:

```text
data/processed/corpus_preprocessed_3472.csv
```

Optional outputs are loaded from:

```text
models/
results/model_comparison/
results/lda/
```
