"""Adversarial validation: how easy is it for a classifier to distinguish
training-set headlines from temporal-val headlines? If accuracy >> 50%, we
have strong feature drift and our model will likely degrade on the leaderboard
test set.

Reports the AUC and the most-drifted features (n-grams) for diagnosis.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=str(DATA / "train.csv"))
    ap.add_argument("--temporal-val-csv", default=str(DATA / "temporal_val.csv"))
    ap.add_argument("--top-k-features", type=int, default=30)
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    temp = pd.read_csv(args.temporal_val_csv)
    if temp.empty:
        print("Temporal val empty; skipping adversarial validation.")
        return

    n_per = min(len(train), len(temp), 5000)
    train_s = train.sample(n=n_per, random_state=42)
    temp_s = temp.sample(n=min(n_per, len(temp)), random_state=42)
    print(f"Adversarial validation samples: train={len(train_s)} vs temp={len(temp_s)}")

    X = pd.concat([train_s["headline"], temp_s["headline"]], ignore_index=True).astype(str).tolist()
    y = np.array([0] * len(train_s) + [1] * len(temp_s))  # 1 = temporal_val (the "future")

    vect = TfidfVectorizer(
        ngram_range=(1, 2), max_features=20000, min_df=2, sublinear_tf=True
    )
    Xv = vect.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    accs = []
    for fold, (tr, te) in enumerate(skf.split(Xv, y)):
        clf = LogisticRegression(C=1.0, max_iter=200, solver="liblinear")
        clf.fit(Xv[tr], y[tr])
        prob = clf.predict_proba(Xv[te])[:, 1]
        pred = (prob > 0.5).astype(int)
        auc = roc_auc_score(y[te], prob)
        acc = float((pred == y[te]).mean())
        aucs.append(auc)
        accs.append(acc)
        print(f"  fold {fold}: auc={auc:.3f} acc={acc:.3f}")

    print(f"\nMEAN AUC: {np.mean(aucs):.3f} (±{np.std(aucs):.3f})")
    print(f"MEAN ACC: {np.mean(accs):.3f} (±{np.std(accs):.3f})")
    print("Interpretation:")
    print("  AUC ~ 0.5: train and temporal-val look identical. Drift is not a concern.")
    print("  AUC > 0.7: noticeable drift. Models will likely degrade on the leaderboard.")
    print("  AUC > 0.85: severe drift. Need topic/entity-blind features or augmentation.")

    # Top drifted features
    clf = LogisticRegression(C=1.0, max_iter=300, solver="liblinear")
    clf.fit(Xv, y)
    coefs = clf.coef_[0]
    feat_names = np.array(vect.get_feature_names_out())
    top_temp = np.argsort(coefs)[-args.top_k_features:][::-1]
    top_train = np.argsort(coefs)[: args.top_k_features]
    print(f"\nFeatures most associated with temporal-val (the drifted side):")
    for i in top_temp:
        print(f"  {feat_names[i]:30s}  coef={coefs[i]:+.3f}")
    print(f"\nFeatures most associated with train (less prevalent in temp val):")
    for i in top_train:
        print(f"  {feat_names[i]:30s}  coef={coefs[i]:+.3f}")


if __name__ == "__main__":
    main()
