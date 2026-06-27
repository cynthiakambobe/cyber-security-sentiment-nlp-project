# Zambia Cyber-Legislation Sentiment Dashboard

This repository contains the Streamlit deployment package for the thesis project:

> Development of a Natural Language Processing Framework for Analysing Public Sentiment on Zambia's 2025 Cyber Security Legislation

The app presents aggregate analysis of anonymised public Facebook comments about Zambia's 2025 Cyber Security and Cyber Crimes legislation. It includes corpus summaries, sentiment distributions, model-comparison results, LDA topic outputs, representative comment exploration, and a live prediction demo using the lightweight TF-IDF + SVM model.

## Streamlit Cloud

Use this app entry point:

```text
dashboard/app.py
```

Streamlit Cloud will install the packages in:

```text
requirements.txt
```

## Run Locally

```powershell
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Included Deployment Artifacts

```text
dashboard/app.py
src/preprocessing.py
data/processed/corpus_preprocessed_3472.csv
models/svm/tfidf_svm.joblib
results/model_comparison/
results/lda/
```

Large training artifacts, notebooks, thesis documents, raw data, intermediate data, and local environment files are intentionally excluded from the deployment repository.

For full local experimentation and model training, use `requirements-dev.txt` and the scripts in `src/`.
