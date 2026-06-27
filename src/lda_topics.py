"""
Reproducible LDA thematic analysis of the 3,472-record corpus.

Run from the repository root:
    python src/07_lda_topics.py

Outputs are written to results/lda/.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

try:
    from src.preprocessing import tokens as shared_tokens
except ImportError:
    from preprocessing import tokens as shared_tokens


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_preprocessed_3472.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "lda"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text_original"
LABEL_COL = "label"
RANDOM = 42
EXTRA_STOPWORDS = {
    "law", "cyber", "act", "zambia", "zambian", "people", "government",
    "get", "go", "one", "said", "say", "like", "even", "also", "us",
    "will", "new", "bill",
    # Topic modelling needs thematic nouns rather than common stance/filler words.
    # Negators remain in supervised sentiment preprocessing but are removed here.
    "not", "good", "think", "need", "much", "time", "thing", "something",
    "would", "want", "wanted", "see", "let", "well", "please", "come",
    "know", "understand", "read", "back", "work", "start", "comment",
    "medium", "country",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LDA topic modelling.")
    parser.add_argument("--min-topics", type=int, default=3)
    parser.add_argument("--max-topics", type=int, default=10)
    parser.add_argument("--selection-passes", type=int, default=8)
    parser.add_argument("--final-passes", type=int, default=15)
    return parser.parse_args()


def tokenize(text):
    tokenised = shared_tokens(
        text,
        extra_stop=EXTRA_STOPWORDS,
        min_chars=3,
    )
    # shared_tokens checks stopwords before lemmatisation, so plural forms such
    # as "laws" can become the stopped lemma "law". Filter once more afterward.
    return [token for token in tokenised if token not in EXTRA_STOPWORDS]


def topic_overlap_table(model, topic_count, num_words=12):
    """Return pairwise top-word overlap as a topic-distinctiveness diagnostic."""
    topic_words = {
        topic_id: {
            word
            for word, _weight in model.show_topic(topic_id, topn=num_words)
        }
        for topic_id in range(topic_count)
    }
    rows = []
    for first_topic in range(topic_count):
        for second_topic in range(first_topic + 1, topic_count):
            overlap = sorted(topic_words[first_topic] & topic_words[second_topic])
            rows.append({
                "topic_a": first_topic,
                "topic_b": second_topic,
                "shared_word_count": len(overlap),
                "shared_words": ", ".join(overlap),
            })
    return pd.DataFrame(rows)


def load_corpus():
    df = pd.read_csv(CORPUS_PATH)
    required = {"_doc_id", TEXT_COL, LABEL_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing corpus columns: {sorted(missing)}")
    df["tokens"] = df[TEXT_COL].fillna("").apply(tokenize)
    excluded = int(df["tokens"].str.len().lt(3).sum())
    df = df[df["tokens"].str.len().ge(3)].copy().reset_index(drop=True)
    print(f"Documents used for LDA: {len(df):,}")
    print(f"Documents excluded (<3 tokens): {excluded:,}")
    return df


def prepare_bow(df):
    dictionary = corpora.Dictionary(df["tokens"])
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bow = [dictionary.doc2bow(document) for document in df["tokens"]]
    if len(dictionary) == 0:
        raise RuntimeError("The LDA dictionary is empty after frequency filtering.")
    print(f"Dictionary size: {len(dictionary):,}")
    return dictionary, bow


def select_topic_count(df, dictionary, bow, topic_range, passes):
    rows = []
    models = {}
    for topic_count in topic_range:
        model = LdaModel(
            corpus=bow,
            id2word=dictionary,
            num_topics=topic_count,
            passes=passes,
            iterations=100,
            random_state=RANDOM,
            alpha="auto",
            eta="auto",
        )
        coherence = CoherenceModel(
            model=model,
            texts=df["tokens"],
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence()
        rows.append({"topics": topic_count, "coherence_cv": coherence})
        models[topic_count] = model
        print(f"K={topic_count:2d} coherence(c_v)={coherence:.4f}")

    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["coherence_cv"].idxmax(), "topics"])
    scores.to_csv(RESULTS_DIR / "coherence_scores.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.plot(scores["topics"], scores["coherence_cv"], marker="o", color="#2171B5")
    plt.axvline(best_k, color="#08306B", linestyle="--", label=f"Selected K={best_k}")
    plt.xlabel("Number of topics")
    plt.ylabel("c_v coherence")
    plt.title("LDA Topic-Count Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "coherence_curve.png", dpi=200, bbox_inches="tight")
    plt.close()
    return best_k, scores


def train_final_model(df, dictionary, bow, topic_count, passes):
    model = LdaModel(
        corpus=bow,
        id2word=dictionary,
        num_topics=topic_count,
        passes=passes,
        iterations=150,
        random_state=RANDOM,
        alpha="auto",
        eta="auto",
    )

    topic_rows = []
    for topic_id, terms in model.show_topics(
        num_topics=topic_count,
        num_words=12,
        formatted=False,
    ):
        topic_rows.append({
            "topic": topic_id,
            "top_words": ", ".join(word for word, _weight in terms),
        })
    topics = pd.DataFrame(topic_rows).sort_values("topic")
    topics.to_csv(RESULTS_DIR / "topics.csv", index=False)
    print("\n", topics.to_string(index=False))

    overlap = topic_overlap_table(model, topic_count)
    overlap.to_csv(RESULTS_DIR / "topic_word_overlap.csv", index=False)
    repeated_pairs = overlap[overlap["shared_word_count"].gt(0)]
    if not repeated_pairs.empty:
        print("\nTop-word overlap between topics:")
        print(repeated_pairs.to_string(index=False))

    dominant = []
    for document_bow in bow:
        distribution = model.get_document_topics(
            document_bow,
            minimum_probability=0.0,
        )
        dominant.append(max(distribution, key=lambda pair: pair[1]))

    assignments = df[["_doc_id", TEXT_COL, LABEL_COL]].copy()
    assignments["dominant_topic"] = [topic for topic, _probability in dominant]
    assignments["topic_probability"] = [probability for _topic, probability in dominant]
    assignments.to_csv(RESULTS_DIR / "document_topics.csv", index=False)

    topic_sentiment = pd.crosstab(
        assignments["dominant_topic"],
        assignments[LABEL_COL],
    ).reindex(columns=["NEGATIVE", "NEUTRAL", "POSITIVE"], fill_value=0)
    topic_sentiment.to_csv(RESULTS_DIR / "topic_sentiment_counts.csv")

    topic_share = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100
    topic_share.to_csv(RESULTS_DIR / "topic_sentiment_percentages.csv")

    model.save(str(RESULTS_DIR / "lda_model.gensim"))
    dictionary.save(str(RESULTS_DIR / "lda_dictionary.gensim"))
    return model, assignments, topic_sentiment


def save_optional_visualisation(model, bow, dictionary):
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensim_vis
    except ImportError:
        print("Optional pyLDAvis output skipped. Install with: pip install pyLDAvis")
        return
    output_path = RESULTS_DIR / "lda_visualisation.html"
    try:
        # pyLDAvis defaults to PCoA, whose eigendecomposition can produce tiny
        # complex values with recent NumPy/SciPy releases. Metric MDS remains
        # real-valued and avoids JSON serialisation failures.
        visualisation = gensim_vis.prepare(
            model,
            bow,
            dictionary,
            mds="mmds",
            n_jobs=1,
            # Keep pyLDAvis bubble 1 aligned with LDA topic 0, bubble 2 with
            # topic 1, and so on. The dashboard supplies the descriptive key.
            sort_topics=False,
        )
        pyLDAvis.save_html(visualisation, str(output_path))
        print(f"Wrote interactive visualisation to {output_path}")
    except (TypeError, ValueError) as error:
        print(
            "Optional pyLDAvis output skipped because the installed numerical "
            f"libraries are incompatible: {error}"
        )


def main():
    args = parse_args()
    if args.min_topics < 2 or args.max_topics < args.min_topics:
        raise ValueError("Topic range must satisfy 2 <= min_topics <= max_topics.")

    df = load_corpus()
    dictionary, bow = prepare_bow(df)
    best_k, _scores = select_topic_count(
        df,
        dictionary,
        bow,
        range(args.min_topics, args.max_topics + 1),
        args.selection_passes,
    )
    print(f"\nSelected topic count: {best_k}")
    model, _assignments, _topic_sentiment = train_final_model(
        df,
        dictionary,
        bow,
        best_k,
        args.final_passes,
    )
    save_optional_visualisation(model, bow, dictionary)
    print(f"\nOutputs saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
