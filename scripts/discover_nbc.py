"""Discover NBC News article URLs via Wayback Machine CDX API.

Filters URLs to article-shaped slugs (-rcna\\d+ or -ncna\\d+ or -n\\d{6,}) and
collapses by urlkey to dedupe snapshots. Supports paginated fetches across
multiple year ranges.
"""
from __future__ import annotations

import csv
import os
import time
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

UA = "Mozilla/5.0 (compatible; AcademicResearch/1.0; +mailto:research@example.edu)"
SESS = requests.Session()
SESS.headers.update({"User-Agent": UA})

CDX = "http://web.archive.org/cdx/search/cdx"


def cdx_query(year_from: int, year_to: int, page: int, page_size: int = 150000) -> list[list[str]]:
    qs = (
        "url=nbcnews.com/*"
        "&output=json"
        f"&from={year_from}0101&to={year_to}1231"
        "&filter=statuscode:200"
        "&filter=mimetype:text/html"
        "&filter=original:.*(rcna|ncna|n[0-9]{7,}).*"
        "&collapse=urlkey"
        "&fl=original,timestamp"
        f"&pageSize={page_size}&page={page}"
    )
    url = f"{CDX}?{qs}"
    for attempt in range(5):
        try:
            r = SESS.get(url, timeout=180)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(5 + attempt * 5)
                continue
            return []
        except Exception:
            time.sleep(2 + attempt)
    return []


def cdx_pages(year_from: int, year_to: int, page_size: int = 150000) -> int:
    """Get total pages for a query."""
    qs = (
        "url=nbcnews.com/*"
        f"&from={year_from}0101&to={year_to}1231"
        "&filter=statuscode:200"
        "&filter=mimetype:text/html"
        "&filter=original:.*(rcna|ncna|n[0-9]{7,}).*"
        "&collapse=urlkey"
        f"&pageSize={page_size}&showNumPages=true"
    )
    url = f"{CDX}?{qs}"
    try:
        r = SESS.get(url, timeout=120)
        if r.status_code == 200:
            return int(r.text.strip())
    except Exception:
        pass
    return 1


def main():
    out = DATA / "urls_nbc.csv"
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []

    # Use a smaller page size to keep responses manageable & reduce 503s
    PAGE_SIZE = 50000

    # Walk recent years first (2024, 2023, 2022, 2021, 2020) — most relevant
    year_ranges = [(2024, 2025), (2023, 2023), (2022, 2022), (2021, 2021), (2020, 2020)]

    for yf, yt in year_ranges:
        n_pages = cdx_pages(yf, yt, page_size=PAGE_SIZE)
        print(f"[{yf}-{yt}] {n_pages} pages", flush=True)
        for p in range(n_pages):
            data = cdx_query(yf, yt, p, page_size=PAGE_SIZE)
            if not data or len(data) < 2:
                print(f"  page {p}: empty", flush=True)
                continue
            n_added = 0
            for row in data[1:]:  # skip header
                if len(row) < 2:
                    continue
                url, ts = row[0], row[1]
                # normalize
                url = url.split("#")[0].split("?")[0]
                # strip http:// vs https:// difference
                if url.startswith("http://"):
                    url = "https://" + url[7:]
                if url in seen:
                    continue
                seen.add(url)
                rows.append((url, ts[:8] if ts else ""))
                n_added += 1
            print(f"  page {p}: +{n_added}, total {len(rows)}", flush=True)
            time.sleep(1.0)  # be polite
            if len(rows) > 250000:
                break
        if len(rows) > 250000:
            break

    # Write
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "lastmod"])
        w.writerows(rows)
    print(f"wrote {out} with {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
