"""Package a 5-model ensemble: 4 transformers + FeatPlusStylo soft-vote.

Uses inline tokenizer JSON for each transformer."""
from __future__ import annotations

import argparse
import base64
import os
import pickle
import sys
import string
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification


PREPROCESS_PY = '''"""preprocess.py — return headlines."""
import re, unicodedata, pandas as pd

_HEADLINE_COLS = ["headline", "scraped_headline", "alternative_headline", "title", "text"]
_URL_COLS = ["url", "URL", "link", "Link"]
_LABEL_COLS = ["label", "source", "outlet", "publisher", "y"]
_SUFFIX_RE = re.compile(r"\\s*[\\|\\-\\u2013\\u2014]\\s*(?:fox news|nbc news|nbcnews\\.com|foxnews\\.com|fox business)\\s*$", re.IGNORECASE)


def _norm(t):
    if not isinstance(t, str): return ""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\\s+", " ", t).strip()
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
'''


MODEL_PY_TEMPLATE = '''"""model.py — 4-transformer + stylo ensemble."""
from __future__ import annotations

import base64, inspect, os, pickle, string, sys, tempfile
from typing import Any, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack as sp_hstack


W_M1 = {W_M1}
W_M2 = {W_M2}
W_M3 = {W_M3}
W_M4 = {W_M4}
W_STY = {W_STY}
MAX_LEN = {MAX_LEN}

# Each model has: config_json, tokenizer_json_b64, tokenizer_config, vocab/merges
M1_CFG = """{M1_CFG}"""
M1_TOKJ = "{M1_TOKJ}"
M1_TOKC = """{M1_TOKC}"""
M1_VOCAB = "{M1_VOCAB}"
M1_MERGES = "{M1_MERGES}"
M1_SP = "{M1_SP}"

M2_CFG = """{M2_CFG}"""
M2_TOKJ = "{M2_TOKJ}"
M2_TOKC = """{M2_TOKC}"""
M2_VOCAB = "{M2_VOCAB}"
M2_MERGES = "{M2_MERGES}"
M2_SP = "{M2_SP}"

M3_CFG = """{M3_CFG}"""
M3_TOKJ = "{M3_TOKJ}"
M3_TOKC = """{M3_TOKC}"""
M3_VOCAB = "{M3_VOCAB}"
M3_MERGES = "{M3_MERGES}"
M3_SP = "{M3_SP}"

M4_CFG = """{M4_CFG}"""
M4_TOKJ = "{M4_TOKJ}"
M4_TOKC = """{M4_TOKC}"""
M4_VOCAB = "{M4_VOCAB}"
M4_MERGES = "{M4_MERGES}"
M4_SP = "{M4_SP}"


STOPWORDS = set("a an the and or but if then is are was were be been being have has had do does did will would could should may might must can shall to of in on at by for with about against between into through during before after above below from up down out over under again further once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t just don now i me my we us our you your he him his she her it its they them their this that these those who whom which what whose as".split())


def stylo_features(texts):
    out = np.zeros((len(texts), 18), dtype=np.float32)
    for i, t in enumerate(texts):
        if not isinstance(t, str): t = str(t)
        n = max(len(t), 1); words = t.split(); nw = max(len(words), 1)
        n_alpha = sum(1 for c in t if c.isalpha())
        n_upper = sum(1 for c in t if c.isupper())
        n_digit = sum(1 for c in t if c.isdigit())
        n_punct = sum(1 for c in t if c in string.punctuation)
        n_q = t.count("?"); n_excl = t.count("!")
        n_quote = t.count(chr(39)) + t.count(chr(34)) + t.count("\\u2018") + t.count("\\u2019") + t.count("\\u201c") + t.count("\\u201d")
        n_colon = t.count(":"); n_dash = t.count("-") + t.count("\\u2014"); n_comma = t.count(",")
        n_caps_words = sum(1 for w in words if w.isupper() and len(w) >= 2)
        n_titlecase = sum(1 for w in words if w[:1].isupper() and not w.isupper())
        n_stop = sum(1 for w in words if w.lower() in STOPWORDS)
        avg_wl = float(np.mean([len(w) for w in words])) if words else 0.0
        n_space = sum(1 for c in t if c.isspace())
        out[i, 0] = n; out[i, 1] = nw
        out[i, 2] = n_upper / max(n_alpha, 1); out[i, 3] = n_digit / n; out[i, 4] = n_punct / n
        out[i, 5] = float(n_q > 0); out[i, 6] = float(n_excl > 0); out[i, 7] = n_q; out[i, 8] = n_excl
        out[i, 9] = n_quote; out[i, 10] = n_colon; out[i, 11] = n_dash; out[i, 12] = n_comma
        out[i, 13] = n_caps_words / nw; out[i, 14] = n_titlecase / nw; out[i, 15] = n_stop / nw
        out[i, 16] = avg_wl; out[i, 17] = n_space / n
    return out


class StyloTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler(with_mean=False)
    def fit(self, X, y=None):
        f = stylo_features(X); self.scaler.fit(f); return self
    def transform(self, X):
        f = stylo_features(X); f = self.scaler.transform(f)
        return csr_matrix(f.astype(np.float32))


class FeatPlusStylo:
    def __init__(self, feats, scaler, model):
        self.feats = feats; self.scaler = scaler; self.model = model
    def _build(self, X):
        F_ = self.feats.transform(X)
        S = self.scaler.transform(stylo_features(X)).astype(np.float32)
        return sp_hstack([F_, csr_matrix(S)]).tocsr()
    def predict(self, X):
        return self.model.predict(self._build(X))
    def predict_proba(self, X):
        return self.model.predict_proba(self._build(X))


class FeatThenModel:
    def __init__(self, feats, model):
        self.feats = feats; self.model = model
    def predict(self, X):
        return self.model.predict(self.feats.transform(X))
    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(self.feats.transform(X))


import __main__, types
_hp = types.ModuleType("headline_pipeline")
for cls in [StyloTransformer, FeatPlusStylo, FeatThenModel]:
    setattr(__main__, cls.__name__, cls)
    setattr(_hp, cls.__name__, cls)
_hp.stylo_features = stylo_features
_hp.STOPWORDS = STOPWORDS
sys.modules["headline_pipeline"] = _hp


_DEVICE = None
_MODELS = {}
_TOKS = {}
_STYLO = None


def _find_pt():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in [here, os.getcwd(), os.path.join(here, "..")]:
        for n in ["model.pt"]:
            p = os.path.join(d, n)
            if os.path.exists(p) and os.path.getsize(p) > 1024:
                return p
    return None


def _setup_tok_dir(cfg, tok_b64, tok_cfg, vocab_b64, merges_b64, sp_b64, prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    with open(os.path.join(d, "tokenizer.json"), "wb") as f:
        f.write(base64.b64decode(tok_b64))
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        f.write(tok_cfg)
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write(cfg)
    if vocab_b64:
        # Try both vocab.txt and vocab.json
        try:
            data = base64.b64decode(vocab_b64)
            # Heuristic: if first byte is { it's json, else txt
            fname = "vocab.json" if data[:1] == b"{" else "vocab.txt"
            with open(os.path.join(d, fname), "wb") as f:
                f.write(data)
        except Exception:
            pass
    if merges_b64:
        with open(os.path.join(d, "merges.txt"), "wb") as f:
            f.write(base64.b64decode(merges_b64))
    if sp_b64:
        with open(os.path.join(d, "special_tokens_map.json"), "wb") as f:
            f.write(base64.b64decode(sp_b64))
    return d


def _ensure_model(idx):
    global _DEVICE
    if idx in _MODELS: return
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = globals()[f"M{idx}_CFG"]
    tok_b64 = globals()[f"M{idx}_TOKJ"]
    tok_cfg = globals()[f"M{idx}_TOKC"]
    vocab = globals()[f"M{idx}_VOCAB"]
    merges = globals()[f"M{idx}_MERGES"]
    sp = globals()[f"M{idx}_SP"]
    d = _setup_tok_dir(cfg, tok_b64, tok_cfg, vocab, merges, sp, f"m{idx}_")
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    _TOKS[idx] = AutoTokenizer.from_pretrained(d)
    config = AutoConfig.from_pretrained(d)
    config.num_labels = 2
    m = AutoModelForSequenceClassification.from_config(config)
    pt = _find_pt()
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    sd = {k[len(f"m{idx}_"):]: v for k, v in ckpt.items() if k.startswith(f"m{idx}_")}
    m.load_state_dict(sd, strict=False)
    m.eval().to(_DEVICE)
    _MODELS[idx] = m


def _ensure_stylo():
    global _STYLO
    if _STYLO is not None: return
    pt = _find_pt()
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    pb = ckpt["_stylo_pickle"]
    if isinstance(pb, torch.Tensor): pb = bytes(pb.cpu().numpy().tobytes())
    _STYLO = pickle.loads(pb)


def _proba(idx, texts, batch_size=64):
    m = _MODELS[idx]; tok = _TOKS[idx]
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = texts[i:i+batch_size]
            enc = tok(b, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(_DEVICE)
            allowed = set(inspect.signature(m.forward).parameters.keys())
            enc = {k: v for k, v in enc.items() if k in allowed}
            out = m(**enc).logits
            probs.append(F.softmax(out.float(), dim=-1).cpu().numpy())
    return np.vstack(probs)


class Model(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        self._sentinel = nn.Parameter(torch.zeros(1))

    def predict(self, batch):
        for i in (1, 2, 3, 4):
            _ensure_model(i)
        _ensure_stylo()
        texts = [str(x) for x in batch]
        ps = []
        for i, w in zip((1, 2, 3, 4), (W_M1, W_M2, W_M3, W_M4)):
            ps.append(w * _proba(i, texts))
        ps.append(W_STY * _STYLO.predict_proba(texts))
        wsum = W_M1 + W_M2 + W_M3 + W_M4 + W_STY
        agg = np.sum(ps, axis=0) / wsum
        return [int(p) for p in agg.argmax(1)]


def get_model():
    return Model()
'''


def _b64_or_empty(p):
    if not p.exists(): return ""
    return base64.b64encode(p.read_bytes()).decode("ascii")


def collect_files(md):
    cfg = (md / "config.json").read_text()
    tok_json = (md / "tokenizer.json").read_bytes() if (md / "tokenizer.json").exists() else b""
    tok_cfg = (md / "tokenizer_config.json").read_text() if (md / "tokenizer_config.json").exists() else ""
    # vocab can be either vocab.txt or vocab.json
    vocab_path = md / "vocab.json"
    if not vocab_path.exists(): vocab_path = md / "vocab.txt"
    vocab = _b64_or_empty(vocab_path) if vocab_path.exists() else ""
    merges = _b64_or_empty(md / "merges.txt")
    sp = _b64_or_empty(md / "special_tokens_map.json")
    return {
        "cfg": cfg,
        "tok_b64": base64.b64encode(tok_json).decode("ascii"),
        "tok_cfg": tok_cfg,
        "vocab_b64": vocab,
        "merges_b64": merges,
        "sp_b64": sp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-dir", required=True)
    ap.add_argument("--m2-dir", required=True)
    ap.add_argument("--m3-dir", required=True)
    ap.add_argument("--m4-dir", required=True)
    ap.add_argument("--stylo-pkl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--w1", type=float, default=1.0)
    ap.add_argument("--w2", type=float, default=1.0)
    ap.add_argument("--w3", type=float, default=1.0)
    ap.add_argument("--w4", type=float, default=1.0)
    ap.add_argument("--w-sty", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ckpt = {"_sentinel": torch.zeros(1)}
    files = []
    for i, mdir in enumerate([args.m1_dir, args.m2_dir, args.m3_dir, args.m4_dir], start=1):
        m = AutoModelForSequenceClassification.from_pretrained(mdir, torch_dtype=torch.float16)
        sd = {f"m{i}_{k}": v for k, v in m.state_dict().items()}
        ckpt.update(sd)
        files.append(collect_files(Path(mdir)))

    with open(args.stylo_pkl, "rb") as f:
        sty_bytes = f.read()
    ckpt["_stylo_pickle"] = torch.frombuffer(sty_bytes, dtype=torch.uint8).clone()

    torch.save(ckpt, out / "model.pt")
    sz = (out / "model.pt").stat().st_size / 1e6
    print(f"  model.pt: {sz:.1f} MB")

    subs = {
        "{W_M1}": str(args.w1), "{W_M2}": str(args.w2),
        "{W_M3}": str(args.w3), "{W_M4}": str(args.w4),
        "{W_STY}": str(args.w_sty),
        "{MAX_LEN}": str(args.max_len),
    }
    for i in (1, 2, 3, 4):
        f = files[i-1]
        subs[f"{{M{i}_CFG}}"] = f["cfg"].replace('"""', '\\"\\"\\"')
        subs[f"{{M{i}_TOKJ}}"] = f["tok_b64"]
        subs[f"{{M{i}_TOKC}}"] = f["tok_cfg"].replace('"""', '\\"\\"\\"')
        subs[f"{{M{i}_VOCAB}}"] = f["vocab_b64"]
        subs[f"{{M{i}_MERGES}}"] = f["merges_b64"]
        subs[f"{{M{i}_SP}}"] = f["sp_b64"]
    model_py = MODEL_PY_TEMPLATE
    for k, v in subs.items():
        model_py = model_py.replace(k, v)
    (out / "model.py").write_text(model_py)
    (out / "preprocess.py").write_text(PREPROCESS_PY)
    sz_total = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"  total: {sz_total:.1f} MB")


if __name__ == "__main__":
    main()
