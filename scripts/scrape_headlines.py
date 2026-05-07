"""Scrape headlines from a list of article URLs (Fox or NBC).

Reads data/urls_<source>.csv (url, lastmod), fetches each URL with polite parallelism,
extracts the headline via h1 / og:title / twitter:title, and writes
data/scraped_<source>.csv (url, headline, lastmod, source).

Caches raw HTML to data/cache_<source>/<sha1>.html.gz so reruns are cheap.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA_POOL = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ROOT / "data"


def _h(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(UA_POOL),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


_SPACE_RE = re.compile(r"\s+")
_SUFFIX_RE = re.compile(
    r"\s*[\|\-–—]\s*(?:fox news|nbc news|nbcnews\.com|foxnews\.com|fox business)\s*$",
    re.IGNORECASE,
)


def _clean_headline(text: str) -> str:
    text = _SPACE_RE.sub(" ", text).strip()
    text = _SUFFIX_RE.sub("", text).strip()
    return text


def _extract_headline(html: bytes) -> str | None:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return None

    # 1. og:title
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        h = _clean_headline(og["content"])
        if h:
            return h
    # 2. twitter:title
    tw = soup.find("meta", attrs={"name": "twitter:title"})
    if tw and tw.get("content"):
        h = _clean_headline(tw["content"])
        if h:
            return h
    # 3. h1.headline.speakable (Fox) or h1[role=heading] / h1.article-hero-headline / h1
    for sel in [
        "h1.headline.speakable",
        "h1.article-hero-headline",
        "h1.headline",
        "h1",
    ]:
        el = soup.select_one(sel)
        if el:
            h = _clean_headline(el.get_text(" ", strip=True))
            if h:
                return h
    # 4. <title>
    if soup.title and soup.title.string:
        return _clean_headline(soup.title.string)
    return None


def _fetch(url: str, cache_dir: Path, sess: requests.Session, timeout: int = 25) -> bytes | None:
    cp = cache_dir / f"{_h(url)}.html.gz"
    if cp.exists():
        try:
            with gzip.open(cp, "rb") as f:
                return f.read()
        except Exception:
            pass
    for attempt in range(3):
        try:
            r = sess.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and r.content:
                with gzip.open(cp, "wb") as f:
                    f.write(r.content)
                return r.content
            if r.status_code in (404, 410, 451):
                # don't retry permanent failures
                return None
        except Exception:
            pass
        time.sleep(1.0 + random.random())
    return None


def scrape(source: str, max_urls: int | None = None, workers: int = 16, in_path: str | None = None):
    in_csv = Path(in_path) if in_path else (DATA / f"urls_{source}.csv")
    out_csv = DATA / f"scraped_{source}.csv"
    cache_dir = DATA / f"cache_{source}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    urls: list[tuple[str, str]] = []
    with open(in_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            urls.append((row["url"], row.get("lastmod", "")))
    if max_urls:
        urls = urls[:max_urls]

    # If output already exists, skip URLs already scraped
    seen: set[str] = set()
    out_mode = "a"
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                seen.add(row["url"])
    else:
        out_mode = "w"
    todo = [(u, lm) for u, lm in urls if u not in seen]
    print(f"[{source}] total {len(urls)}, already done {len(seen)}, todo {len(todo)}", flush=True)

    sess = _make_session()
    n_ok = 0
    n_err = 0
    t0 = time.time()
    write_lock_check = 0

    with open(out_csv, out_mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if out_mode == "w":
            w.writerow(["url", "headline", "lastmod", "source"])

        def task(url_lm):
            url, lm = url_lm
            data = _fetch(url, cache_dir, sess)
            if not data:
                return None
            h = _extract_headline(data)
            if not h:
                return None
            return (url, h, lm, source)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(task, ul): ul for ul in todo}
            for fut in as_completed(futs):
                res = fut.result()
                if res is None:
                    n_err += 1
                else:
                    w.writerow(res)
                    n_ok += 1
                if (n_ok + n_err) % 200 == 0:
                    rate = (n_ok + n_err) / max(time.time() - t0, 1)
                    print(
                        f"[{source}] ok={n_ok} err={n_err} rate={rate:.1f}/s",
                        flush=True,
                    )
                    f.flush()
    print(f"[{source}] DONE ok={n_ok} err={n_err}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["fox", "nbc"])
    p.add_argument("--max-urls", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--in-path", default=None)
    args = p.parse_args()
    scrape(args.source, max_urls=args.max_urls, workers=args.workers, in_path=args.in_path)


if __name__ == "__main__":
    main()
