"""
Strong headline-only classifier for Fox-vs-NBC.
Reuses classes from headline_pipeline.py so saved pipes can be loaded
in any process that imports that module.
"""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

import functools
print = functools.partial(print, flush=True)

ROOT = "/home/asethi04/cis5190-finalproject"
sys.path.insert(0, f"{ROOT}/scripts")

from headline_pipeline import (
    stylo_features,
    StyloTransformer,
    FeatThenModel,
    FeatPlusStylo,
    StackingPipeline,
    StackStyloPipeline,
    sigmoid,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix, hstack as sp_hstack
import lightgbm as lgb

OUTDIR = f"{ROOT}/experiments/best_headline"
os.makedirs(OUTDIR, exist_ok=True)

print("Loading data...")
train = pd.read_csv(f"{ROOT}/data/train.csv")
val = pd.read_csv(f"{ROOT}/data/val_frozen.csv")
tval = pd.read_csv(f"{ROOT}/data/temporal_val_frozen.csv")
X_train = train["headline"].astype(str).tolist()
y_train = train["label"].values.astype(int)
X_val = val["headline"].astype(str).tolist()
y_val = val["label"].values.astype(int)
X_tval = tval["headline"].astype(str).tolist()
y_tval = tval["label"].values.astype(int)

print(f"train={len(X_train)} val={len(X_val)} tval={len(X_tval)}")
results = {}


def evaluate(name, pipe):
    pv = pipe.predict(X_val)
    pt = pipe.predict(X_tval)
    av = accuracy_score(y_val, pv)
    at = accuracy_score(y_tval, pt)
    print(f"  [{name}] val={av:.4f}  tval={at:.4f}")
    results[name] = (av, at, pipe)
    return av, at


def save_best_so_far(label):
    if not results:
        return
    sr = sorted(results.items(), key=lambda kv: kv[1][1], reverse=True)
    name, (av, at, pipe) = sr[0]
    out_path = f"{OUTDIR}/pipe.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  [{label}] BEST so far: {name} tval={at:.4f}, saved to {out_path}")


# Build TF-IDF features once
t0 = time.time()
print("Fitting main TF-IDF feats (1-3 word + 2-6 char, 500k each)...")
feats = FeatureUnion(
    [
        ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=500000, min_df=2,
                                 sublinear_tf=True, lowercase=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), max_features=500000,
                                 min_df=2, sublinear_tf=True, lowercase=True, strip_accents="unicode")),
    ]
)
feats.fit(X_train, y_train)
Xtr = feats.transform(X_train)
Xv = feats.transform(X_val)
Xt = feats.transform(X_tval)
print(f"  dim: {Xtr.shape[1]} ({time.time()-t0:.1f}s)")


# -------------------------------------------------
# Approach 4: LR sweeps on TF-IDF
# -------------------------------------------------
print("\n=== Approach 4: LR sweeps ===")
for C in [2.0, 4.0, 8.0]:
    t1 = time.time()
    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
    clf.fit(Xtr, y_train)
    evaluate(f"lr_C{C}", FeatThenModel(feats, clf))
    print(f"    fit {time.time()-t1:.1f}s")
save_best_so_far("after-LR-only")


# -------------------------------------------------
# Approach 3: Stylometric (concat)
# -------------------------------------------------
print("\n=== Approach 3: Stylometric ===")
t1 = time.time()
sty_tr = stylo_features(X_train)
sty_v = stylo_features(X_val)
sty_t = stylo_features(X_tval)
sty_scaler = StandardScaler()
sty_tr_s = sty_scaler.fit_transform(sty_tr).astype(np.float32)
sty_v_s = sty_scaler.transform(sty_v).astype(np.float32)
sty_t_s = sty_scaler.transform(sty_t).astype(np.float32)
print(f"  stylo computed ({time.time()-t1:.1f}s)")

Xtr_st = sp_hstack([Xtr, csr_matrix(sty_tr_s)]).tocsr()
Xv_st = sp_hstack([Xv, csr_matrix(sty_v_s)]).tocsr()
Xt_st = sp_hstack([Xt, csr_matrix(sty_t_s)]).tocsr()

# C=2.0 gave 0.8056 (best). Also try class_weight balanced and C=1.0/3.0.
for cfg in [
    {"name": "stylo_lr_C1.0", "C": 1.0, "cw": None},
    {"name": "stylo_lr_C2.0", "C": 2.0, "cw": None},
    {"name": "stylo_lr_C3.0", "C": 3.0, "cw": None},
    {"name": "stylo_lr_C4.0", "C": 4.0, "cw": None},
    {"name": "stylo_lr_C2.0_bal", "C": 2.0, "cw": "balanced"},
    {"name": "stylo_lr_C4.0_bal", "C": 4.0, "cw": "balanced"},
]:
    t2 = time.time()
    clf = LogisticRegression(C=cfg["C"], max_iter=2000, solver="liblinear",
                             class_weight=cfg["cw"])
    clf.fit(Xtr_st, y_train)
    evaluate(cfg["name"], FeatPlusStylo(feats, sty_scaler, clf))
    print(f"    fit {time.time()-t2:.1f}s")
save_best_so_far("after-stylo")


# -------------------------------------------------
# Approach 2: Stacking (OOF)
# -------------------------------------------------
print("\n=== Approach 2: Stacking ===")


def get_decision(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return sigmoid(model.decision_function(X))
    raise RuntimeError


def make_base():
    return {
        "lr": LogisticRegression(C=4.0, max_iter=2000, solver="liblinear"),
        "svc": LinearSVC(C=1.0, max_iter=3000),
        "cnb": ComplementNB(alpha=0.5),
        "sgd": SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=30, random_state=0, tol=1e-4),
    }


names = list(make_base().keys())
oof_train = np.zeros((Xtr.shape[0], len(names)), dtype=np.float32)
val_meta = np.zeros((Xv.shape[0], len(names)), dtype=np.float32)
tval_meta = np.zeros((Xt.shape[0], len(names)), dtype=np.float32)
trained_on_all = {}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for j, key in enumerate(names):
    print(f"  base {key} oof...")
    t1 = time.time()
    for fold, (tr, vd) in enumerate(skf.split(Xtr, y_train)):
        m = make_base()[key]
        m.fit(Xtr[tr], y_train[tr])
        oof_train[vd, j] = get_decision(m, Xtr[vd])
    m_all = make_base()[key]
    m_all.fit(Xtr, y_train)
    trained_on_all[key] = m_all
    val_meta[:, j] = get_decision(m_all, Xv)
    tval_meta[:, j] = get_decision(m_all, Xt)
    print(f"    {key}: {time.time()-t1:.1f}s")

meta = LogisticRegression(C=2.0, max_iter=2000)
meta.fit(oof_train, y_train)
pv_s = meta.predict(val_meta)
pt_s = meta.predict(tval_meta)
av_s = accuracy_score(y_val, pv_s)
at_s = accuracy_score(y_tval, pt_s)
print(f"  [stack_meta_lr] val={av_s:.4f}  tval={at_s:.4f}")
stack_pipe = StackingPipeline(feats, trained_on_all, meta, names)
results["stack_meta_lr"] = (av_s, at_s, stack_pipe)
save_best_so_far("after-stack")


# -------------------------------------------------
# Bonus: Stack outputs + stylo into meta
# -------------------------------------------------
print("\n=== Bonus: stack+stylo meta ===")
oof_meta_X = np.hstack([oof_train, sty_tr_s])
val_meta_X = np.hstack([val_meta, sty_v_s])
tval_meta_X = np.hstack([tval_meta, sty_t_s])
for C in [0.5, 1.0, 2.0]:
    meta2 = LogisticRegression(C=C, max_iter=2000)
    meta2.fit(oof_meta_X, y_train)
    pv_b = meta2.predict(val_meta_X)
    pt_b = meta2.predict(tval_meta_X)
    av_b = accuracy_score(y_val, pv_b)
    at_b = accuracy_score(y_tval, pt_b)
    print(f"  [stack_stylo_C{C}] val={av_b:.4f}  tval={at_b:.4f}")
    pipe = StackStyloPipeline(feats, trained_on_all, meta2, names, sty_scaler)
    results[f"stack_stylo_C{C}"] = (av_b, at_b, pipe)
save_best_so_far("after-stack-stylo")


# -------------------------------------------------
# Approach 1: LightGBM (small, capped time)
# -------------------------------------------------
print("\n=== Approach 1: LightGBM ===")
t1 = time.time()
print("  fitting compact TF-IDF for LGB...")
feats_lgb = FeatureUnion(
    [
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=3,
                                 sublinear_tf=True, lowercase=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000,
                                 min_df=3, sublinear_tf=True, lowercase=True, strip_accents="unicode")),
    ]
)
feats_lgb.fit(X_train, y_train)
Xtr_l = feats_lgb.transform(X_train)
print(f"  LGB TF-IDF dim: {Xtr_l.shape[1]} ({time.time()-t1:.1f}s)")
for cfg in [
    {"name": "lgb_300_31", "n": 300, "leaves": 31, "lr": 0.08},
]:
    t2 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=cfg["n"], num_leaves=cfg["leaves"], learning_rate=cfg["lr"],
        n_jobs=8, objective="binary", verbose=-1,
        subsample=0.9, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=0.1, random_state=42,
    )
    clf.fit(Xtr_l, y_train)
    evaluate(cfg["name"], FeatThenModel(feats_lgb, clf))
    print(f"    LGB fit {time.time()-t2:.1f}s")
save_best_so_far("after-lgb")


# -------------------------------------------------
# Final summary
# -------------------------------------------------
print("\n=== Final summary (sorted by tval) ===")
sr = sorted(results.items(), key=lambda kv: kv[1][1], reverse=True)
for name, (av, at, _) in sr:
    print(f"  {name}: val={av:.4f}  tval={at:.4f}")

best_name, (best_av, best_at, best_pipe) = sr[0]
print(f"\nBEST: {best_name} -> tval={best_at:.4f}  val={best_av:.4f}")

out_path = f"{OUTDIR}/pipe.pkl"
with open(out_path, "wb") as f:
    pickle.dump(best_pipe, f)

# Verify
with open(out_path, "rb") as f:
    loaded = pickle.load(f)
preds = loaded.predict(X_tval)
acc = accuracy_score(y_tval, preds)
print(f"Saved {out_path}; reloaded tval acc = {acc:.4f}")
print(f"\nBEST tval accuracy: {acc:.4f}")
