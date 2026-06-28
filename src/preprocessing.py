"""
preprocessing.py
SINGLE SOURCE OF TRUTH for text cleaning, imported by every script and the dashboard.

  preprocess(text)  -> cleaned space-joined string  (SVM, CNN-LSTM candidates)
  light_clean(text) -> minimal clean, natural text   (DistilBERT)
  tokens(text, ...) -> list of tokens                (LDA / custom)

Why one file: if cleaning is defined separately in each script, the dashboard's
predictions can silently stop matching the training pipeline. Import from here
everywhere so they can never drift.

(In the recommended repo layout this file lives at src/preprocessing.py; it is kept
beside the scripts here so `from preprocessing import ...` works in the flat layout.)
"""
import re

try:
    import nltk
    for _pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(_pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None

# Negation words are RETAINED (they flip sentiment: "not safe" != "safe").
NEGATORS = {"no", "not", "nor", "none", "never", "cannot", "cant", "wont", "dont",
            "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "shouldnt",
            "wouldnt", "couldnt", "against", "without", "nothing", "neither"}
_BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "they", "this",
    "to", "was", "we", "were", "with", "you", "your",
}

try:
    STOPWORDS = set(stopwords.words("english")) - NEGATORS
except Exception:
    STOPWORDS = _BASIC_STOPWORDS - NEGATORS

_LEM = WordNetLemmatizer() if WordNetLemmatizer is not None else None


def _lemmatize(word):
    if _LEM is None:
        return word
    try:
        return _LEM.lemmatize(word)
    except Exception:
        return word


def light_clean(text):
    """Minimal cleaning for transformer models: keep natural text and word order."""
    t = re.sub(r"http\S+|www\.\S+", " ", str(text))   # urls
    t = re.sub(r"@\w+", " ", t)                       # mentions
    return re.sub(r"\s+", " ", t).strip()


def tokens(text, extra_stop=frozenset(), min_chars=2):
    """Clean -> tokenise -> stop-word removal (negators kept) -> lemmatise -> list.

    Strips emojis and non-Latin script (so Latin-script Bemba/Nyanja words survive
    as ordinary tokens). To keep emoji sentiment instead, `pip install emoji` and add
    `t = emoji.demojize(t, delimiters=(' ', ' '))` before the [^a-z] substitution.
    """
    t = str(text).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)          # keep the hashtag word, drop '#'
    t = re.sub(r"[^a-z\s]", " ", t)
    stop = STOPWORDS | set(extra_stop)
    return [_lemmatize(w) for w in t.split() if w not in stop and len(w) >= min_chars]


def preprocess(text):
    """Cleaned, space-joined string for TF-IDF / Word2Vec / CNN-LSTM candidates."""
    return " ".join(tokens(text, min_chars=2))
