"""
Reproducibly select comments for annotation/model-corpus construction.

This script documents the selection stage from the raw Facebook comment corpus:

1. load data/raw/corpus_raw.json;
2. remove scraper error rows and empty comments;
3. apply the legislation keyword eligibility filter;
4. randomly select a fixed-size sample using a fixed random seed.

The output is a deterministic candidate set for human annotation. The final
model target label must still come from human annotation, not from the keyword
filter.

Run from the repository root:

    python src/01_select_model_corpus.py --sample-size 3472 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "corpus_raw.json"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_SAMPLE_SIZE = 3_472
DEFAULT_SEED = 42


KEYWORDS = [
    # Core legislation names
    "cyber security act", "cyber crimes act", "cyber crime bill",
    "cyber security act no. 3", "cybercrime", "cybercrimes",
    "cyber security", "cyber crime", "cyber law", "cyber bill",
    "cyber legislation", "cyber amendment",

    # Real section numbers — offences breakdown
    "section 3", "section 4", "section 5", "section 6", "section 10",
    "section 19", "section 21", "section 22", "section 24",

    # Real section numbers — surveillance/interception
    "section 29", "section 31", "section 34", "section 36", "section 37",
    "section 39", "section 40",

    # Offence descriptions used in public discourse
    "critical information", "misleading information", "false information",
    "domestic terrorism", "ethnic division", "ethnic divisions",
    "indecent", "vulgar", "humiliate",

    # Institutions and actors
    "law association of zambia", "laz", "zctu",
    "zambia congress of trade unions", "high court", "us embassy",

    # Legal/rights framing
    "constitutional", "unconstitutional", "civil liberties",
    "freedom of expression", "free speech", "human rights",
    "surveillance", "surveillance state", "interception",
    "data protection", "digital rights", "chilling effect",
    "criminalise", "criminalize", "criminalised", "criminalized",

    # Authoritarian framing
    "dictator", "dictatorship", "authoritarian", "tyranny", "tyrant",
    "police state", "oppression", "oppress", "gag", "totalitarian", "regime",

    # Penalties
    "imprisonment", "years imprisonment", "life imprisonment", "jail",
    "arrested",

    # Process/procedural terms
    "assent", "enacted", "act of parliament", "gazette",

    # General framing terms
    "draconian", "repressive", "vague", "overreach",

    # Privacy/monitoring
    "privacy", "monitor", "monitored", "spy", "spying", "silence",
    "silenced",
]


def load_raw_corpus(path: Path) -> pd.DataFrame:
    """Load either a JSON array or JSON-lines raw corpus."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return pd.DataFrame(json.loads(text))
    return pd.read_json(path, lines=True)


def stable_record_key(row: pd.Series) -> str:
    """Create a deterministic key from stable raw fields."""
    parts = [
        str(row.get("commentId", "")),
        str(row.get("commentUrl", "")),
        str(row.get("facebookUrl", "")),
        str(row.get("date", "")),
        str(row.get("text", "")),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def contains_keyword(value: object, keywords: list[str]) -> bool:
    if not isinstance(value, str):
        return False
    text = value.lower()
    return any(keyword.lower() in text for keyword in keywords)


def is_substantive_comment(value: object, min_tokens: int = 5) -> bool:
    if not isinstance(value, str):
        return False
    tokens = [word for word in value.split() if re.match(r"\w", word)]
    return len(tokens) >= min_tokens


def add_keyword_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same eligibility logic used by the annotation workflow."""
    frame = frame.copy()
    frame["comment_keyword_match"] = frame["text"].apply(
        lambda value: contains_keyword(value, KEYWORDS)
    )
    frame["post_title_keyword_match"] = frame.get("postTitle", "").apply(
        lambda value: contains_keyword(value, KEYWORDS)
    )
    frame["substantive_comment"] = frame["text"].apply(is_substantive_comment)
    frame["keyword_eligible"] = frame["comment_keyword_match"] | (
        frame["post_title_keyword_match"] & frame["substantive_comment"]
    )
    frame["keyword_match_source"] = ""
    frame.loc[frame["comment_keyword_match"], "keyword_match_source"] = "comment"
    frame.loc[
        ~frame["comment_keyword_match"]
        & frame["post_title_keyword_match"]
        & frame["substantive_comment"],
        "keyword_match_source",
    ] = "post_title_context"
    return frame


def prepare_raw(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove scraper errors and create stable IDs for traceability."""
    frame = frame.copy()
    if "text" not in frame.columns:
        raise ValueError("Raw corpus must contain a 'text' column.")

    error_mask = frame.get(
        "error", pd.Series([None] * len(frame), index=frame.index)
    ).notna()
    valid_raw_records = int((~error_mask).sum())
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["valid_comment"] = frame["text"].str.strip().ne("") & ~error_mask
    frame = frame[frame["valid_comment"]].reset_index(drop=True)

    if "_record_key" not in frame.columns:
        frame["_record_key"] = frame.apply(stable_record_key, axis=1)
    else:
        missing_key = frame["_record_key"].isna() | frame["_record_key"].astype(str).eq("")
        frame.loc[missing_key, "_record_key"] = frame[missing_key].apply(
            stable_record_key, axis=1
        )

    if "_doc_id" not in frame.columns:
        frame["_doc_id"] = "DOC_" + frame["_record_key"].str[:16].str.upper()
    else:
        missing_doc_id = frame["_doc_id"].isna() | frame["_doc_id"].astype(str).eq("")
        frame.loc[missing_doc_id, "_doc_id"] = (
            "DOC_" + frame.loc[missing_doc_id, "_record_key"].str[:16].str.upper()
        )

    return frame, valid_raw_records


def select_sample(eligible: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size > len(eligible):
        raise ValueError(
            f"Requested {sample_size:,} comments, but only "
            f"{len(eligible):,} are keyword-eligible."
        )
    return (
        eligible.sample(n=sample_size, random_state=seed)
        .sort_values("_doc_id")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_loaded = load_raw_corpus(args.raw_path)
    raw, valid_raw_records = prepare_raw(raw_loaded)
    flagged = add_keyword_flags(raw)

    eligible = flagged[flagged["keyword_eligible"]].copy()
    ineligible = flagged[~flagged["keyword_eligible"]].copy()
    selected = select_sample(eligible, args.sample_size, args.seed)

    eligible_path = INTERIM_DIR / "keyword_eligible_comments.csv"
    ineligible_path = INTERIM_DIR / "keyword_ineligible_comments.csv"
    selected_path = (
        PROCESSED_DIR
        / f"selected_comments_for_annotation_{args.sample_size}_seed{args.seed}.csv"
    )
    manifest_path = (
        PROCESSED_DIR
        / f"selection_manifest_{args.sample_size}_seed{args.seed}.json"
    )

    export_columns = [
        column for column in [
            "_doc_id", "_record_key", "date", "text", "postTitle",
            "facebookUrl", "commentUrl", "_dataset_name",
            "keyword_match_source", "comment_keyword_match",
            "post_title_keyword_match",
        ] if column in flagged.columns
    ]

    eligible[export_columns].to_csv(eligible_path, index=False, encoding="utf-8")
    ineligible[export_columns].to_csv(ineligible_path, index=False, encoding="utf-8")
    selected[export_columns].to_csv(selected_path, index=False, encoding="utf-8")

    manifest = {
        "raw_path": str(args.raw_path.relative_to(PROJECT_ROOT)),
        "sample_size": args.sample_size,
        "random_seed": args.seed,
        "keyword_count": len(KEYWORDS),
        "raw_records": int(len(raw_loaded)),
        "valid_raw_records_excluding_scraper_errors": valid_raw_records,
        "non_empty_valid_comments": int(len(raw)),
        "keyword_eligible_comments": int(len(eligible)),
        "keyword_ineligible_comments": int(len(ineligible)),
        "selected_comments": int(len(selected)),
        "selection_rule": (
            "Simple random sample without replacement from keyword-eligible "
            "comments using pandas.DataFrame.sample(random_state=seed)."
        ),
        "outputs": {
            "eligible": str(eligible_path.relative_to(PROJECT_ROOT)),
            "ineligible": str(ineligible_path.relative_to(PROJECT_ROOT)),
            "selected": str(selected_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
