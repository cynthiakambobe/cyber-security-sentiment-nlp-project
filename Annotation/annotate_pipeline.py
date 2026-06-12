"""
annotate_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Thesis: Development of a Natural Language Processing Framework for Analyzing Public Sentiment on Zambia’s 2025 Cyber Security Legislation 
Author: Cynthia Lubasi Muyunda (202502245) | ZCASU

Pipeline stages:
  1. Pull all datasets from Apify account via API
  2. Merge into single JSON — comments + post context
  3. Anonymise (strip profile identifiers)
  4. Keyword filter at comment level (eligibility check)
  5. Annotate via Claude API (codebook-grounded system prompt)
  6. Save outputs:
       corpus_raw.json          — merged, anonymised, unfiltered
       corpus_filtered.json     — keyword-eligible comments only
       corpus_annotated.csv     — full annotation output
       corpus_review_queue.csv  — UNCERTAIN + LOW confidence → human review
       pipeline_summary.txt     — counts at each stage for Section 3.4

SETUP INSTRUCTIONS:
  1. Install Ollama: https://ollama.ai
  2. Pull Mistral model: ollama pull mistral
  3. Start Ollama server: ollama serve (runs on http://localhost:11434)
  4. In another terminal, run this script:
     pip install apify-client pandas requests
     export APIFY_API_TOKEN="your_token_here"
     python annotate_pipeline.py

Model: mistral (7B, free, CPU-friendly, ~4GB RAM)
Cost: $0 (fully local inference)
Speed: ~2-5 sec per comment on CPU (depends on text length)
"""

import os
import json
import re
import time
import uuid
import requests
from datetime import datetime
from pathlib import Path

import pandas as pd
from apify_client import ApifyClient

# ── Configuration ─────────────────────────────────────────────────────────────

APIFY_TOKEN    = os.environ.get("APIFY_API_TOKEN", "")
OLLAMA_URL     = "http://localhost:11434"  # Ollama server address
OLLAMA_MODEL   = "mistral"  # 7B model, ~4GB quantized
OUTPUT_DIR     = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Keywords for comment-level eligibility filter
# A comment must contain at least one of these (case-insensitive) to be included
KEYWORDS = [
    "cyber security act", "cyber crimes act", "cybercrime", "cybercrimes",
    "cyber security", "cyber crime", "cyber law", "digital rights",
    "section 22", "section 42", "section 11", "section 8", "section 15",
    "section 3", "part iii", "part ii", "surveillance", "interception",
    "false information", "offensive communication", "chilling effect",
    "zambia cyber", "cyber bill", "cyber amendment", "cyber legislation",
    "data protection", "freedom of expression", "free speech",
    "unconstitutional", "vague", "draconian", "repressive",
    "zctu", "zambia congress", "trade union cyber",
]

# ── Codebook-grounded system prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert annotation assistant for an academic NLP research project at the Zambia Centre for Accountancy Studies (ZCASU). Your task is to classify English-language Facebook comments about Zambia's 2025 Cyber Security Act and Cyber Crimes Act into one of three sentiment categories.

You must follow these rules exactly. They are derived from a locked annotation codebook.

─────────────────────────────────────────────────────────
TASK
─────────────────────────────────────────────────────────
Classify the sentiment expressed in the comment toward Zambia's 2025 Cyber Security Act and/or Cyber Crimes Act. You are labelling the author's evaluative stance toward the LEGISLATION — not toward cybercrime in general, not toward the commenter's political party, not toward Zambia generally.

You will also receive the parent post title for context. Use it only to resolve ambiguous pronoun references (e.g. "this law", "it"). Do NOT let the parent post's sentiment influence your label for the comment — the comment is the unit of analysis.

─────────────────────────────────────────────────────────
CLASS DEFINITIONS
─────────────────────────────────────────────────────────

POSITIVE
The comment expresses approval, support, relief, or optimism about the legislation, its provisions, or its enforcement. The author endorses the legislative intent or expects a beneficial outcome.
Typical signals: praise for cybersecurity protections; relief at legal recourse for victims; support for data protection provisions; positive language about government intent.
Example signal words: welcome, protect, long overdue, good move, finally, much needed, progress, commend, support, applaud, safe, secure, accountability, justice, good job.

NEGATIVE
The comment expresses opposition, fear, anger, disappointment, or distrust toward the legislation, its provisions, or its enforcement. The author criticises the legislative intent, raises rights concerns, or anticipates harmful outcomes.
Typical signals: concerns about threats to freedom of expression or press freedom; fear of surveillance or government overreach; criticism of vague provisions; opposition to criminalisation of online speech; distrust of enforcement intentions.
Example signal words: dangerous, repressive, silence, surveillance, criminalise, unconstitutional, threat, chilling effect, vague, overreach, concerned, worried, wrong, reject, oppose, draconian, political tool, abuse, blind, poorly drafted.

NEUTRAL
The comment reports, describes, quotes, questions, or shares information about the legislation without a discernible evaluative stance. The author neither endorses nor opposes; the comment is informational, procedural, or genuinely ambiguous.
Typical signals: factual questions about provisions; defining legal terms; sharing information without evaluation; balanced commentary.
Example signal words: signed into law, provides that, Section X states, what does the Act say, an amendment is, defines, according to.

UNCERTAIN
Use ONLY when the sentiment genuinely cannot be determined after applying all rules. Do not use as a default for difficult cases. This triggers human review.

─────────────────────────────────────────────────────────
DECISION RULES (apply in order)
─────────────────────────────────────────────────────────

1. SARCASM/IRONY: Label by literal reading unless sarcasm is unambiguous (explicit "NOT", laughing emoji after absurd claim, universally recognised ironic phrasing). If uncertain whether sarcastic → NEUTRAL.

2. MIXED SENTIMENT: If one sentiment clearly dominates in intensity → apply that label. If both sentiments are approximately equal → NEUTRAL.

3. NEGATION: "not safe" = NEGATIVE. "not dangerous" = POSITIVE. Read the full sentence — do not match keywords alone.

4. RHETORICAL QUESTIONS: Questions implying a clear evaluative stance → label by implied sentiment. Genuine factual questions → NEUTRAL.

5. EXPRESSED FEAR OR CONCERN: Worry or apprehension about consequences of the legislation (e.g. "I am worried this will affect journalists") = NEGATIVE.

6. DOMAIN TERMS WITH FIXED ORIENTATION:
   - "chilling effect" → NEGATIVE
   - "surveillance state" or "surveillance tool" → NEGATIVE
   - "criminalise online speech" → NEGATIVE
   - "long overdue" or "much needed" → POSITIVE
   - "unconstitutional" or "violates rights" → NEGATIVE
   - "political tool" or "weaponise" → NEGATIVE
   - "vague", "poorly drafted", "lacks clarity" → NEGATIVE (in context of the Act)
   - "good move", "good job", "well done" → POSITIVE (in context of the Act)
   - Section number references alone → NEUTRAL unless accompanied by evaluation

7. COMMENTS NOT ABOUT THE LEGISLATION: If the comment is clearly about something else entirely (another law, a person, unrelated topic) → UNCERTAIN (will be excluded).

8. EMOJI-ONLY OR VERY SHORT COMMENTS: If the comment contains only emoji or fewer than 5 meaningful words with no clear sentiment signal → UNCERTAIN.

─────────────────────────────────────────────────────────
OUTPUT FORMAT
─────────────────────────────────────────────────────────
Respond with ONLY a valid JSON object. No preamble, no explanation outside the JSON, no markdown.

{
  "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "UNCERTAIN",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "rationale": "One sentence max 20 words citing the specific signal.",
  "rule_applied": "Which rule or class definition governed the decision."
}

Confidence:
- HIGH: Clear and unambiguous.
- MEDIUM: Likely correct but minor ambiguity.
- LOW: Judgement call; human review recommended even if not UNCERTAIN."""


# ── Stage 1: Pull all datasets from Apify ────────────────────────────────────

def pull_apify_datasets(token):
    """
    Pull all dataset items from all runs in the Apify account.
    Returns list of (dataset_name, items_list) tuples.
    """
    client = ApifyClient(token)
    datasets = []

    print("\n[Stage 1] Fetching datasets from Apify account...")
    dataset_list = list(client.datasets().list().items)
    print(f"  Found {len(dataset_list)} datasets")

    for ds in dataset_list:
        ds_id   = ds.get("id")
        ds_name = ds.get("name") or ds_id
        try:
            items = list(client.dataset(ds_id).iterate_items())
            if items:
                datasets.append((ds_name, items))
                print(f"  ✓ {ds_name}: {len(items)} items")
            else:
                print(f"  – {ds_name}: empty, skipped")
        except Exception as e:
            print(f"  ✗ {ds_name}: error — {e}")

    return datasets


# ── Stage 2: Merge into single flat list ─────────────────────────────────────

def merge_datasets(datasets):
    """
    Flatten all dataset items into a single list.
    Add dataset_name and a unique doc_id to each record.
    """
    merged = []
    for ds_name, items in datasets:
        for item in items:
            item["_dataset_name"] = ds_name
            item["_doc_id"] = str(uuid.uuid4())
            merged.append(item)

    print(f"\n[Stage 2] Merged {len(merged)} total records from {len(datasets)} datasets")
    return merged


# ── Stage 3: Anonymise ───────────────────────────────────────────────────────

FIELDS_TO_STRIP = [
    "profileUrl", "profilePicture", "profileId",
    "profileName", "id", "feedbackId",
]

def anonymise(records):
    """
    Remove all personal identifiers per codebook Section 2 and thesis §3.7.1.
    Replace profileName with an anonymous ID for audit traceability.
    """
    anonymised = []
    for rec in records:
        clean = {k: v for k, v in rec.items() if k not in FIELDS_TO_STRIP}
        # Replace profileName with anonymous token — preserves ability to
        # detect duplicate commenters without storing identity
        clean["anon_commenter_id"] = "ANON_" + rec.get("_doc_id", "")[:8].upper()
        anonymised.append(clean)

    print(f"[Stage 3] Anonymised {len(anonymised)} records — personal identifiers stripped")
    return anonymised


# ── Stage 4: Keyword filter at comment level ─────────────────────────────────

def keyword_filter(records, keywords):
    """
    Retain only comments whose text contains at least one keyword.
    Also retain comments where the postTitle contains a keyword AND the
    comment text is substantive (>=5 meaningful tokens) — borderline
    self-contained cases per codebook Section 2.1.
    """
    def matches(text):
        if not isinstance(text, str):
            return False
        t = text.lower()
        return any(kw.lower() in t for kw in keywords)

    def is_substantive(text):
        if not isinstance(text, str):
            return False
        tokens = [w for w in text.split() if re.match(r'\w', w)]
        return len(tokens) >= 5

    eligible   = []
    ineligible = []

    for rec in records:
        comment_text = rec.get("text", "")
        post_title   = rec.get("postTitle", "")
        comment_matches = matches(comment_text)
        post_matches    = matches(post_title)
        substantive     = is_substantive(comment_text)

        if comment_matches or (post_matches and substantive):
            rec["keyword_match_source"] = "comment" if comment_matches else "post_title_context"
            eligible.append(rec)
        else:
            ineligible.append(rec)

    print(f"[Stage 4] Keyword filter:")
    print(f"  Eligible   : {len(eligible)}")
    print(f"  Ineligible : {len(ineligible)}")
    print(f"  Retention  : {len(eligible)/len(records)*100:.1f}%")
    return eligible, ineligible


# ── Stage 5: Annotate via Ollama (local Mistral 7B) ────────────────────────────

def annotate_record(rec):
    """Annotate a single record via Ollama local model."""
    comment_text = rec.get("text", "")
    post_title   = rec.get("postTitle", "")[:300]  # truncate long post titles

    user_message = f"""Classify the following Facebook comment.

PARENT POST TITLE (context only — do not label this):
{post_title}

COMMENT TEXT:
{comment_text}"""

    try:
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"{SYSTEM_PROMPT}\n\n{user_message}",
                "stream": False,
                "temperature": 0.1,  # Low temperature for consistency
            },
            timeout=60
        )
        response.raise_for_status()
        
        raw_output = response.json().get("response", "").strip()
        # Try to extract JSON from response (model may include explanation)
        # Look for JSON object in the output
        json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', raw_output, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            raw = raw_output
        
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        return {
            "label": "UNCERTAIN",
            "confidence": "LOW",
            "rationale": "JSON parse error — model output malformed.",
            "rule_applied": "Parse failure"
        }
    except requests.exceptions.ConnectionError:
        return {
            "label": "ERROR",
            "confidence": "LOW",
            "rationale": "Ollama server not running at " + OLLAMA_URL,
            "rule_applied": "Connection error"
        }
    except Exception as e:
        return {
            "label": "ERROR",
            "confidence": "LOW",
            "rationale": str(e)[:100],
            "rule_applied": "API error"
        }


def annotate_corpus(records, checkpoint_every=100):
    """
    Annotate all eligible records. Save checkpoint CSV every N records.
    Returns annotated list.
    """
    annotated = []
    total     = len(records)

    print(f"\n[Stage 5] Annotating {total} records via Ollama (Mistral 7B, local)...")
    print(f"  Checkpoint every {checkpoint_every} records")
    print(f"  (Model server: {OLLAMA_URL})")

    for i, rec in enumerate(records):
        result = annotate_record(rec)

        annotated_rec = {
            **rec,
            "ai_label"       : result.get("label"),
            "ai_confidence"  : result.get("confidence"),
            "ai_rationale"   : result.get("rationale"),
            "ai_rule_applied": result.get("rule_applied"),
        }
        annotated.append(annotated_rec)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total} annotated...")

        # Checkpoint save
        if (i + 1) % checkpoint_every == 0:
            _save_checkpoint(annotated, i + 1)

        # Rate limiting
        time.sleep(0.5)

    print(f"  Done — {len(annotated)} records annotated")
    return annotated


def _save_checkpoint(records, n):
    path = OUTPUT_DIR / f"checkpoint_{n}.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"  [Checkpoint] Saved {n} records → {path}")


# ── Stage 6: Save outputs ────────────────────────────────────────────────────

EXPORT_COLUMNS = [
    "_doc_id", "anon_commenter_id",
    "commentId", "commentUrl", "facebookUrl", "inputUrl",
    "date", "text", "postTitle",
    "likesCount", "commentsCount", "threadingDepth",
    "facebookId", "_dataset_name", "keyword_match_source",
    "ai_label", "ai_confidence", "ai_rationale", "ai_rule_applied",
]

def save_outputs(raw, filtered, annotated):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw merged JSON
    raw_path = OUTPUT_DIR / "corpus_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    # Filtered JSON (pre-annotation)
    filtered_path = OUTPUT_DIR / "corpus_filtered.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    # Full annotated CSV
    df = pd.DataFrame(annotated)
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    df_out = df[cols]
    annotated_path = OUTPUT_DIR / f"corpus_annotated_{ts}.csv"
    df_out.to_csv(annotated_path, index=False)

    # Human review queue — UNCERTAIN + LOW confidence + ERROR
    review_mask = (
        df_out["ai_label"].isin(["UNCERTAIN", "ERROR"]) |
        (df_out["ai_confidence"] == "LOW")
    )
    review_df = df_out[review_mask].copy()
    review_df["human_label"]  = ""   # blank column for you to fill in
    review_df["human_notes"]  = ""
    review_path = OUTPUT_DIR / f"corpus_review_queue_{ts}.csv"
    review_df.to_csv(review_path, index=False)

    # Confirmed labels — HIGH + MEDIUM confidence, not UNCERTAIN/ERROR
    confirmed_mask = (
        ~df_out["ai_label"].isin(["UNCERTAIN", "ERROR"]) &
        (df_out["ai_confidence"].isin(["HIGH", "MEDIUM"]))
    )
    confirmed_df = df_out[confirmed_mask]
    confirmed_path = OUTPUT_DIR / f"corpus_confirmed_{ts}.csv"
    confirmed_df.to_csv(confirmed_path, index=False)

    print(f"\n[Stage 6] Outputs saved:")
    print(f"  Raw JSON          : {raw_path} ({len(raw)} records)")
    print(f"  Filtered JSON     : {filtered_path} ({len(filtered)} records)")
    print(f"  Annotated CSV     : {annotated_path} ({len(df_out)} records)")
    print(f"  Review queue      : {review_path} ({len(review_df)} records)")
    print(f"  Confirmed labels  : {confirmed_path} ({len(confirmed_df)} records)")

    return df_out, review_df, confirmed_df


def save_summary(raw, filtered, ineligible, annotated_df, review_df, confirmed_df):
    """Write pipeline summary for Section 3.4 data log."""

    label_counts = annotated_df["ai_label"].value_counts().to_dict()
    conf_counts  = annotated_df["ai_confidence"].value_counts().to_dict()

    lines = [
        "Annotation Pipeline Summary",
        f"Run date : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 55,
        "",
        "STAGE 2 — MERGE",
        f"  Total records merged          : {len(raw):,}",
        "",
        "STAGE 3 — ANONYMISATION",
        "  Profile identifiers stripped  : profileUrl, profilePicture,",
        "                                  profileId, profileName, id, feedbackId",
        "",
        "STAGE 4 — KEYWORD FILTER",
        f"  Records before filter         : {len(raw):,}",
        f"  Eligible (keyword match)      : {len(filtered):,}",
        f"  Ineligible (excluded)         : {len(ineligible):,}",
        f"  Retention rate                : {len(filtered)/len(raw)*100:.1f}%",
        "",
        "STAGE 5 — AI ANNOTATION",
        f"  Records annotated             : {len(annotated_df):,}",
        f"  POSITIVE                      : {label_counts.get('POSITIVE', 0):,}",
        f"  NEGATIVE                      : {label_counts.get('NEGATIVE', 0):,}",
        f"  NEUTRAL                       : {label_counts.get('NEUTRAL', 0):,}",
        f"  UNCERTAIN                     : {label_counts.get('UNCERTAIN', 0):,}",
        f"  ERROR                         : {label_counts.get('ERROR', 0):,}",
        "",
        f"  HIGH confidence               : {conf_counts.get('HIGH', 0):,}",
        f"  MEDIUM confidence             : {conf_counts.get('MEDIUM', 0):,}",
        f"  LOW confidence                : {conf_counts.get('LOW', 0):,}",
        "",
        "STAGE 6 — OUTPUTS",
        f"  Routed to human review        : {len(review_df):,}",
        f"  Confirmed (no review needed)  : {len(confirmed_df):,}",
        "",
        "=" * 55,
        "NEXT STEPS",
        "  1. Open corpus_review_queue.csv",
        "  2. Fill in human_label column for each row",
        "  3. Apply codebook rules — your label overrides the AI label",
        "  4. Compute agreement rate on spot-check subset",
        "  5. Merge confirmed + adjudicated into final labelled corpus",
    ]

    summary_path = OUTPUT_DIR / "pipeline_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Summary           : {summary_path}")
    print("\n" + "\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not APIFY_TOKEN:
        raise ValueError("APIFY_API_TOKEN environment variable not set.")
    
    # Check if Ollama is running
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
    except requests.exceptions.ConnectionError:
        raise ValueError(
            f"Ollama server not running at {OLLAMA_URL}\n"
            "Start it with: ollama serve\n"
            "And pull the model with: ollama pull mistral"
        )

    # Stage 1 — Pull from Apify
    datasets = pull_apify_datasets(APIFY_TOKEN)
    if not datasets:
        print("No datasets found. Check your APIFY_API_TOKEN.")
        return

    # Stage 2 — Merge
    raw = merge_datasets(datasets)

    # Stage 3 — Anonymise
    raw = anonymise(raw)

    # Save raw JSON immediately after anonymisation
    raw_path = OUTPUT_DIR / "corpus_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"  Raw corpus saved  : {raw_path}")

    # Stage 4 — Keyword filter
    filtered, ineligible = keyword_filter(raw, KEYWORDS)

    # Stage 5 — Annotate
    annotated = annotate_corpus(filtered)

    # Stage 6 — Save outputs
    annotated_df, review_df, confirmed_df = save_outputs(raw, filtered, annotated)

    # Summary
    save_summary(raw, filtered, ineligible, annotated_df, review_df, confirmed_df)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
   