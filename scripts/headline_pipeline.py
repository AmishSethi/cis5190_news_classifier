"""Reusable classes for the headline classification pipeline.
Importable so pickled instances can be loaded.
"""
import string
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack as sp_hstack


STOPWORDS = set(
    """a an the and or but if then is are was were be been being have has had
do does did will would could should may might must can shall to of in on at by for with
about against between into through during before after above below from up down out over
under again further once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t just don now i me my we us
our you your he him his she her it its they them their this that these those who whom which
what whose as""".split()
)


def stylo_features(texts):
    out = np.zeros((len(texts), 18), dtype=np.float32)
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            t = str(t)
        n = max(len(t), 1)
        words = t.split()
        nw = max(len(words), 1)
        n_alpha = sum(1 for c in t if c.isalpha())
        n_upper = sum(1 for c in t if c.isupper())
        n_digit = sum(1 for c in t if c.isdigit())
        n_punct = sum(1 for c in t if c in string.punctuation)
        n_q = t.count("?")
        n_excl = t.count("!")
        n_quote = t.count("'") + t.count('"') + t.count("‘") + t.count("’") + t.count("“") + t.count("”")
        n_colon = t.count(":")
        n_dash = t.count("-") + t.count("—")
        n_comma = t.count(",")
        n_caps_words = sum(1 for w in words if w.isupper() and len(w) >= 2)
        n_titlecase = sum(1 for w in words if w[:1].isupper() and not w.isupper())
        n_stop = sum(1 for w in words if w.lower() in STOPWORDS)
        avg_wl = float(np.mean([len(w) for w in words])) if words else 0.0
        n_space = sum(1 for c in t if c.isspace())
        out[i, 0] = n
        out[i, 1] = nw
        out[i, 2] = n_upper / max(n_alpha, 1)
        out[i, 3] = n_digit / n
        out[i, 4] = n_punct / n
        out[i, 5] = float(n_q > 0)
        out[i, 6] = float(n_excl > 0)
        out[i, 7] = n_q
        out[i, 8] = n_excl
        out[i, 9] = n_quote
        out[i, 10] = n_colon
        out[i, 11] = n_dash
        out[i, 12] = n_comma
        out[i, 13] = n_caps_words / nw
        out[i, 14] = n_titlecase / nw
        out[i, 15] = n_stop / nw
        out[i, 16] = avg_wl
        out[i, 17] = n_space / n
    return out


class StyloTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, X, y=None):
        f = stylo_features(X)
        self.scaler.fit(f)
        return self

    def transform(self, X):
        f = stylo_features(X)
        f = self.scaler.transform(f)
        return csr_matrix(f.astype(np.float32))


class FeatThenModel:
    """Wraps fitted feats + model into a predict-on-text interface."""

    def __init__(self, feats, model):
        self.feats = feats
        self.model = model

    def predict(self, X):
        return self.model.predict(self.feats.transform(X))

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(self.feats.transform(X))
        return None


class FeatPlusStylo:
    """TF-IDF feats hstacked with scaled stylo features, fed to a fitted model."""

    def __init__(self, feats, scaler, model):
        self.feats = feats
        self.scaler = scaler
        self.model = model

    def _build(self, X):
        F = self.feats.transform(X)
        S = self.scaler.transform(stylo_features(X)).astype(np.float32)
        return sp_hstack([F, csr_matrix(S)]).tocsr()

    def predict(self, X):
        return self.model.predict(self._build(X))

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(self._build(X))
        return None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class StackingPipeline:
    def __init__(self, feats, base_models, meta, names):
        self.feats = feats
        self.base_models = base_models
        self.meta = meta
        self.names = names

    def _meta_X(self, X):
        F = self.feats.transform(X)
        out = np.zeros((F.shape[0], len(self.names)), dtype=np.float32)
        for j, k in enumerate(self.names):
            m = self.base_models[k]
            if hasattr(m, "predict_proba"):
                out[:, j] = m.predict_proba(F)[:, 1]
            else:
                out[:, j] = sigmoid(m.decision_function(F))
        return out

    def predict(self, X):
        return self.meta.predict(self._meta_X(X))

    def predict_proba(self, X):
        return self.meta.predict_proba(self._meta_X(X))


class StackStyloPipeline:
    def __init__(self, feats, base_models, meta, names, scaler):
        self.feats = feats
        self.base_models = base_models
        self.meta = meta
        self.names = names
        self.scaler = scaler

    def _meta_X(self, X):
        F = self.feats.transform(X)
        bx = np.zeros((F.shape[0], len(self.names)), dtype=np.float32)
        for j, k in enumerate(self.names):
            m = self.base_models[k]
            if hasattr(m, "predict_proba"):
                bx[:, j] = m.predict_proba(F)[:, 1]
            else:
                bx[:, j] = sigmoid(m.decision_function(F))
        sx = self.scaler.transform(stylo_features(X)).astype(np.float32)
        return np.hstack([bx, sx])

    def predict(self, X):
        return self.meta.predict(self._meta_X(X))

    def predict_proba(self, X):
        return self.meta.predict_proba(self._meta_X(X))
