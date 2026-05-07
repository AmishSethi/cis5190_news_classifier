"""N-transformer + stylo ensemble (variable N)."""
import argparse, base64, os, pickle, sys
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification

sys.path.insert(0, "scripts")
from package_ens5 import collect_files, PREPROCESS_PY


HEADER = '''"""N-way ensemble (transformers + stylo)."""
import base64, inspect, os, pickle, string, sys, tempfile
from typing import Any, Iterable, List
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack as sp_hstack


N_MODELS = __NMODELS__
WEIGHTS = __WEIGHTS__
W_STY = __WSTY__
MAX_LEN = __MAXLEN__


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


_DEVICE = None; _MODELS = {}; _TOKS = {}; _STYLO = None


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
        try:
            data = base64.b64decode(vocab_b64)
            fname = "vocab.json" if data[:1] == b"{" else "vocab.txt"
            with open(os.path.join(d, fname), "wb") as f:
                f.write(data)
        except: pass
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
    cfg = MODEL_FILES[idx]["cfg"]
    tok_b64 = MODEL_FILES[idx]["tok_b64"]
    tok_cfg = MODEL_FILES[idx]["tok_cfg"]
    vocab = MODEL_FILES[idx]["vocab_b64"]
    merges = MODEL_FILES[idx]["merges_b64"]
    sp = MODEL_FILES[idx]["sp_b64"]
    d = _setup_tok_dir(cfg, tok_b64, tok_cfg, vocab, merges, sp, "m" + str(idx) + "_")
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    _TOKS[idx] = AutoTokenizer.from_pretrained(d)
    config = AutoConfig.from_pretrained(d)
    config.num_labels = 2
    m = AutoModelForSequenceClassification.from_config(config)
    pt = _find_pt()
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    pre = "m" + str(idx) + "_"
    sd = {k[len(pre):]: v for k, v in ckpt.items() if k.startswith(pre)}
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
        for i in range(1, N_MODELS + 1):
            _ensure_model(i)
        _ensure_stylo()
        texts = [str(x) for x in batch]
        ps = []
        for i in range(1, N_MODELS + 1):
            ps.append(WEIGHTS[i - 1] * _proba(i, texts))
        ps.append(W_STY * _STYLO.predict_proba(texts))
        wsum = sum(WEIGHTS) + W_STY
        agg = np.sum(ps, axis=0) / wsum
        return [int(p) for p in agg.argmax(1)]


def get_model():
    return Model()
'''


MODEL_FILES_BLOCK = '''MODEL_FILES = __BODY__
'''


def b64s(b):
    return base64.b64encode(b).decode("ascii") if b else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--stylo-pkl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--w-sty", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()
    assert len(args.model_dirs) == len(args.weights), "weights must match model_dirs"
    n = len(args.model_dirs)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ckpt = {"_sentinel": torch.zeros(1)}
    all_files = []
    for i, mdir in enumerate(args.model_dirs, start=1):
        m = AutoModelForSequenceClassification.from_pretrained(mdir, torch_dtype=torch.float16)
        sd = {f"m{i}_{k}": v for k, v in m.state_dict().items()}
        ckpt.update(sd)
        all_files.append(collect_files(Path(mdir)))

    with open(args.stylo_pkl, "rb") as f:
        sty_bytes = f.read()
    ckpt["_stylo_pickle"] = torch.frombuffer(sty_bytes, dtype=torch.uint8).clone()

    torch.save(ckpt, out / "model.pt")
    sz = (out / "model.pt").stat().st_size / 1e6
    print(f"  model.pt: {sz:.1f} MB")

    # Build MODEL_FILES dict source code (safe via repr)
    body_lines = ["{"]
    for i in range(1, n + 1):
        f = all_files[i - 1]
        body_lines.append(f"    {i}: {{")
        body_lines.append(f"        \"cfg\": {repr(f['cfg'])},")
        body_lines.append(f"        \"tok_b64\": {repr(f['tok_b64'])},")
        body_lines.append(f"        \"tok_cfg\": {repr(f['tok_cfg'])},")
        body_lines.append(f"        \"vocab_b64\": {repr(f['vocab_b64'])},")
        body_lines.append(f"        \"merges_b64\": {repr(f['merges_b64'])},")
        body_lines.append(f"        \"sp_b64\": {repr(f['sp_b64'])},")
        body_lines.append(f"    }},")
    body_lines.append("}")
    body = "\n".join(body_lines)

    model_py = HEADER
    model_py = model_py.replace("__NMODELS__", str(n))
    model_py = model_py.replace("__WEIGHTS__", repr(list(args.weights)))
    model_py = model_py.replace("__WSTY__", str(args.w_sty))
    model_py = model_py.replace("__MAXLEN__", str(args.max_len))
    model_py += "\n\n" + MODEL_FILES_BLOCK.replace("__BODY__", body)
    (out / "model.py").write_text(model_py)
    (out / "preprocess.py").write_text(PREPROCESS_PY)
    sz_total = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e6
    print(f"  total: {sz_total:.1f} MB ({n} transformers + stylo)")


if __name__ == "__main__":
    main()
