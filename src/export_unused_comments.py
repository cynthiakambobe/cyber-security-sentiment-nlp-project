"""Export raw Facebook comments absent from the canonical model corpus.

Run from the repository root:
    python src/export_unused_comments.py

The output has one row per unique ``_doc_id``. Repeated raw records are collapsed
and reported through ``raw_occurrences``.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "corpus_raw.json"
FILTERED_PATH = PROJECT_ROOT / "data" / "interim" / "corpus_filtered.json"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "corpus_split.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "comments_not_used_in_model.csv"
)
ELIGIBLE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "keyword_eligible_comments_not_used_in_model.csv"
)


def read_json(path):
    """Read either a JSON array or JSON Lines file."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)


def main():
    raw = read_json(RAW_PATH)
    filtered = read_json(FILTERED_PATH)
    split = pd.read_csv(SPLIT_PATH)

    required = {"_doc_id", "text"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Raw corpus is missing columns: {sorted(missing)}")

    model_ids = set(split["_doc_id"].dropna())
    eligible_ids = set(filtered["_doc_id"].dropna())

    unused = raw[~raw["_doc_id"].isin(model_ids)].copy()
    unused_raw_record_count = len(unused)
    occurrence_counts = unused["_doc_id"].value_counts()

    preferred_columns = [
        "_doc_id",
        "date",
        "text",
        "postTitle",
        "facebookUrl",
        "commentUrl",
        "_dataset_name",
        "keyword_match_source",
        "raw_occurrences",
    ]
    available_columns = [
        column for column in preferred_columns if column in unused.columns
    ]

    # Duplicate document IDs arise from repeated raw extraction records. Their
    # content is equivalent for this audit, so retain one representative row.
    unused = unused[available_columns].drop_duplicates(
        subset="_doc_id",
        keep="first",
    )
    unused["raw_occurrences"] = (
        unused["_doc_id"].map(occurrence_counts).astype("int64")
    )

    has_text = unused["text"].fillna("").astype(str).str.strip().ne("")
    unused["exclusion_reason"] = "keyword_ineligible"
    unused.loc[
        unused["_doc_id"].isin(eligible_ids),
        "exclusion_reason",
    ] = "keyword_eligible_not_in_final_corpus"
    unused.loc[~has_text, "exclusion_reason"] = "no_usable_comment_text"

    output_columns = [
        "_doc_id",
        "exclusion_reason",
        "date",
        "text",
        "postTitle",
        "facebookUrl",
        "commentUrl",
        "_dataset_name",
        "keyword_match_source",
        "raw_occurrences",
    ]
    output_columns = [
        column for column in output_columns if column in unused.columns
    ]
    unused = unused[output_columns].sort_values(
        ["exclusion_reason", "date", "_doc_id"],
        na_position="last",
    )
    unused.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    eligible_unused = unused[
        unused["exclusion_reason"].eq(
            "keyword_eligible_not_in_final_corpus"
        )
    ].copy()
    eligible_unused.to_csv(
        ELIGIBLE_OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Model corpus: {len(model_ids):,} unique comments")
    print(f"Unused raw records: {unused_raw_record_count:,}")
    print(f"Unused unique document IDs: {len(unused):,}")
    print("\nExclusion reasons:")
    print(unused["exclusion_reason"].value_counts().to_string())
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Wrote {ELIGIBLE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
