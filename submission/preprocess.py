"""preprocess.py — return headlines."""
import re, unicodedata, pandas as pd

_HEADLINE_COLS = ["headline", "scraped_headline", "alternative_headline", "title", "text"]
_URL_COLS = ["url", "URL", "link", "Link"]
_LABEL_COLS = ["label", "source", "outlet", "publisher", "y"]
_SUFFIX_RE = re.compile(r"\s*[\|\-\u2013\u2014]\s*(?:fox news|nbc news|nbcnews\.com|foxnews\.com|fox business)\s*$", re.IGNORECASE)


def _norm(t):
    if not isinstance(t, str): return ""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _SUFFIX_RE.sub("", t).strip()
    return t


def _label_from_row(row):
    for c in _LABEL_COLS:
        if c in row.index:
            v = row[c]
            if pd.isna(v): continue
            sv = str(v).strip().lower()
            if sv in ("foxnews","fox","1","true"): return 1
            if sv in ("nbcnews","nbc","0","false"): return 0
            try: return int(sv)
            except: pass
    for c in _URL_COLS:
        if c in row.index and isinstance(row[c], str):
            u = row[c].lower()
            if "foxnews.com" in u: return 1
            if "nbcnews.com" in u: return 0
    return 0


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    h_col = next((c for c in _HEADLINE_COLS if c in cols), None)
    headlines, labels = [], []
    for _, row in df.iterrows():
        h = ""
        if h_col is not None and not pd.isna(row[h_col]):
            h = _norm(str(row[h_col]))
        if not h: h = "(empty)"
        headlines.append(h)
        labels.append(_label_from_row(row))
    return headlines, labels
