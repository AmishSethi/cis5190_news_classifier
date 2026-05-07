"""Build train/val splits with both random and temporal validation.

Inputs:
- project-resources/Newsheadlines/url_with_headlines.csv (the course-provided 3,805)
- data/scraped_fox.csv, data/scraped_nbc.csv (our additional scraped headlines)

Outputs:
- data/train.csv, data/val.csv (random stratified, 90/10)
- data/temporal_val.csv (most recent 10% of scraped data, by lastmod)
- data/all_clean.csv (all clean labeled headlines, for downstream MLM)
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ROOT / "data"
COURSE_CSV = ROOT / "project-resources" / "Newsheadlines" / "url_with_headlines.csv"

# Labels: 1 = FoxNews (matches the 1-for-FoxNews convention in the spec baseline)
FOX = 1
NBC = 0


def _norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    # strip stray suffixes
    text = re.sub(r"\s*[\|\-–—]\s*(?:fox news|nbc news|nbcnews\.com|foxnews\.com|fox business)\s*$", "", text, flags=re.IGNORECASE).strip()
    return text


def _source_from_url(url: str) -> str | None:
    if "foxnews.com" in url:
        return "fox"
    if "nbcnews.com" in url:
        return "nbc"
    return None


def load_course() -> pd.DataFrame:
    df = pd.read_csv(COURSE_CSV)
    df["headline"] = df["headline"].astype(str).map(_norm)
    df["source"] = df["url"].astype(str).map(_source_from_url)
    df = df.dropna(subset=["source"])
    df = df[df["headline"].str.len() >= 8]
    df["label"] = df["source"].map({"fox": FOX, "nbc": NBC})
    df["lastmod"] = ""
    df["origin"] = "course"
    return df[["url", "headline", "label", "source", "lastmod", "origin"]]


def load_scraped(source: str) -> pd.DataFrame:
    p = DATA / f"scraped_{source}.csv"
    if not p.exists():
        return pd.DataFrame(columns=["url", "headline", "label", "source", "lastmod", "origin"])
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["url", "headline", "label", "source", "lastmod", "origin"])
    df["headline"] = df["headline"].astype(str).map(_norm)
    df = df[df["headline"].str.len() >= 8]
    df["label"] = FOX if source == "fox" else NBC
    df["source"] = source
    df["origin"] = "scraped"
    if "lastmod" not in df.columns:
        df["lastmod"] = ""
    return df[["url", "headline", "label", "source", "lastmod", "origin"]]


def main():
    course = load_course()
    print(f"course: {len(course)} ({(course.label==FOX).sum()} fox / {(course.label==NBC).sum()} nbc)")
    fox_s = load_scraped("fox")
    nbc_s = load_scraped("nbc")
    print(f"scraped fox: {len(fox_s)}, scraped nbc: {len(nbc_s)}")

    all_df = pd.concat([course, fox_s, nbc_s], ignore_index=True)
    # dedupe on (headline, label) to drop any URL collisions / cross-corpus duplicates
    all_df = all_df.drop_duplicates(subset=["headline", "label"], keep="first").reset_index(drop=True)
    print(f"after dedupe: {len(all_df)} ({(all_df.label==FOX).sum()} fox / {(all_df.label==NBC).sum()} nbc)")

    all_df.to_csv(DATA / "all_clean.csv", index=False)

    # If frozen val/temporal_val exist, treat them as fixed and just rebuild train.
    val_frozen = DATA / "val_frozen.csv"
    temp_frozen = DATA / "temporal_val_frozen.csv"
    if val_frozen.exists():
        val = pd.read_csv(val_frozen)
        if temp_frozen.exists():
            temporal_val = pd.read_csv(temp_frozen)
        else:
            temporal_val = pd.DataFrame()
        used_keys = set(val["url"].tolist()) | set(temporal_val.get("url", pd.Series([])).tolist())
        train = all_df[~all_df["url"].isin(used_keys)].reset_index(drop=True)
        # also drop duplicates by headline that match val/temp_val
        used_h = set(val["headline"].tolist()) | set(temporal_val.get("headline", pd.Series([])).tolist())
        train = train[~train["headline"].isin(used_h)].reset_index(drop=True)
        print(f"frozen splits: train={len(train)} (held-out val={len(val)}, temporal_val={len(temporal_val)})")
        train.to_csv(DATA / "train.csv", index=False)
        val.to_csv(DATA / "val.csv", index=False)
        if not temporal_val.empty:
            temporal_val.to_csv(DATA / "temporal_val.csv", index=False)
        return

    # Temporal val: take most recent ~5% of scraped from EACH source (balanced)
    # This mimics the test conditions — leaderboard test set will be future-dated.
    scraped_only = all_df[all_df["origin"] == "scraped"].copy()
    if len(scraped_only) > 200:
        scraped_only["_ts"] = scraped_only["lastmod"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
        scraped_only["_ts"] = pd.to_numeric(scraped_only["_ts"], errors="coerce").fillna(0).astype(int)
        # if all timestamps are 0 (e.g., NBC has no lastmod), fall back to random
        parts = []
        for lab in [FOX, NBC]:
            sub = scraped_only[scraped_only.label == lab].copy()
            if sub.empty:
                continue
            if (sub["_ts"] > 0).any():
                sub = sub.sort_values("_ts", ascending=False)
            else:
                sub = sub.sample(frac=1.0, random_state=42)
            n = min(max(250, int(len(sub) * 0.10)), 1500)
            parts.append(sub.head(n))
        if parts:
            temporal_val = pd.concat(parts, ignore_index=True).drop(columns=["_ts"])
        else:
            temporal_val = scraped_only.head(0).drop(columns=["_ts"], errors="ignore")
    else:
        temporal_val = scraped_only.head(0)

    # Remaining: course + scraped not in temporal_val
    used_keys = set(temporal_val["url"].tolist())
    rest = all_df[~all_df["url"].isin(used_keys)].copy()

    # Stratified random val: 10% of rest
    rng = np.random.default_rng(42)
    rest = rest.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_n = int(len(rest) * 0.10)
    # stratified: take 10% of fox and 10% of nbc separately
    fox_idx = rest[rest.label == FOX].index.to_list()
    nbc_idx = rest[rest.label == NBC].index.to_list()
    rng.shuffle(fox_idx)
    rng.shuffle(nbc_idx)
    fox_val = fox_idx[: int(len(fox_idx) * 0.10)]
    nbc_val = nbc_idx[: int(len(nbc_idx) * 0.10)]
    val_idx = set(fox_val + nbc_val)
    val = rest.loc[list(val_idx)].reset_index(drop=True)
    train = rest.drop(list(val_idx)).reset_index(drop=True)

    print(f"splits: train={len(train)}, val={len(val)}, temporal_val={len(temporal_val)}")
    train.to_csv(DATA / "train.csv", index=False)
    val.to_csv(DATA / "val.csv", index=False)
    temporal_val.to_csv(DATA / "temporal_val.csv", index=False)


if __name__ == "__main__":
    main()
