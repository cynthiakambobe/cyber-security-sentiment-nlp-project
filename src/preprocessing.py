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
import nltk
for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Negation words are RETAINED (they flip sentiment: "not safe" != "safe").
NEGATORS = {"no", "not", "nor", "none", "never", "cannot", "cant", "wont", "dont",
            "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "shouldnt",
            "wouldnt", "couldnt", "against", "without", "nothing", "neither"}
STOPWORDS = set(stopwords.words("english")) - NEGATORS
_LEM = WordNetLemmatizer()


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
    return [_LEM.lemmatize(w) for w in t.split() if w not in stop and len(w) >= min_chars]


def preprocess(text):
    """Cleaned, space-joined string for TF-IDF / Word2Vec / CNN-LSTM candidates."""
    return " ".join(tokens(text, min_chars=2))
