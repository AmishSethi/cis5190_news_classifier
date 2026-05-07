"""Discover article URLs from Fox News & NBC News for headline scraping.

Strategy:
- Fox: sitemap.xml is an index pointing to ~165 paginated article sitemaps. Walk all of them.
- NBC: sitemap blocked at root. Use multiple fallbacks: per-section sitemaps if reachable,
  RSS feeds, Wayback Machine CDX index for historical URLs, and section listing pages.

Output:
- data/urls_fox.csv (url, lastmod)
- data/urls_nbc.csv (url, lastmod)
"""
from __future__ import annotations

import csv
import gzip
import io
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

import requests

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SESS = requests.Session()
SESS.headers.update({"User-Agent": UA, "Accept": "*/*"})


def _get(url: str, timeout: int = 30, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            r = SESS.get(url, timeout=timeout)
            if r.status_code == 200:
                content = r.content
                if url.endswith(".gz") or r.headers.get("Content-Encoding") == "gzip":
                    try:
                        content = gzip.decompress(content)
                    except OSError:
                        pass
                return content
            if r.status_code in (403, 404):
                return None
        except Exception:
            pass
        time.sleep(1.5 ** attempt)
    return None


def _parse_sitemap(xml_bytes: bytes) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (sub_sitemap_urls, [(article_url, lastmod)])."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return [], []
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sub = []
    arts = []
    for s in root.findall("sm:sitemap", ns):
        loc = s.find("sm:loc", ns)
        if loc is not None and loc.text:
            sub.append(loc.text.strip())
    for u in root.findall("sm:url", ns):
        loc = u.find("sm:loc", ns)
        lastmod = u.find("sm:lastmod", ns)
        if loc is not None and loc.text:
            lm = lastmod.text.strip() if lastmod is not None and lastmod.text else ""
            arts.append((loc.text.strip(), lm))
    return sub, arts


def _crawl_sitemap(start_url: str, max_subs: int = 1000) -> list[tuple[str, str]]:
    """BFS through sitemap index until article URLs are found."""
    seen: set[str] = set()
    queue: list[str] = [start_url]
    out: list[tuple[str, str]] = []
    while queue and len(seen) < max_subs:
        batch = queue[:20]
        queue = queue[20:]
        with ThreadPoolExecutor(max_workers=10) as pool:
            futs = {pool.submit(_get, u): u for u in batch}
            for fut in as_completed(futs):
                u = futs[fut]
                if u in seen:
                    continue
                seen.add(u)
                data = fut.result()
                if not data:
                    continue
                subs, arts = _parse_sitemap(data)
                for su in subs:
                    if su not in seen:
                        queue.append(su)
                out.extend(arts)
        print(f"  ... crawled {len(seen)} sitemaps, {len(out)} article urls", flush=True)
    return out


def discover_fox() -> list[tuple[str, str]]:
    print("=== FOX: sitemap walk ===", flush=True)
    arts = _crawl_sitemap("https://www.foxnews.com/sitemap.xml")
    # dedupe
    seen = set()
    deduped = []
    for u, lm in arts:
        if u in seen:
            continue
        seen.add(u)
        deduped.append((u, lm))
    print(f"FOX total unique: {len(deduped)}", flush=True)
    return deduped


def _wayback_cdx(domain_pattern: str, limit: int = 50000, from_year: int = 2018, to_year: int = 2025) -> list[tuple[str, str]]:
    """Hit the Wayback CDX API for historical URLs from a domain."""
    api = "http://web.archive.org/cdx/search/cdx"
    params = {
        "url": domain_pattern,
        "output": "json",
        "limit": str(limit),
        "filter": "statuscode:200",
        "filter": "mimetype:text/html",
        "fl": "original,timestamp",
        "collapse": "urlkey",
        "from": f"{from_year}0101",
        "to": f"{to_year}1231",
    }
    # requests doesn't allow duplicate params via dict; build manually
    qs = (
        f"url={urllib.parse.quote(domain_pattern)}"
        f"&output=json&limit={limit}"
        f"&filter=statuscode:200&filter=mimetype:text/html"
        f"&fl=original,timestamp&collapse=urlkey"
        f"&from={from_year}0101&to={to_year}1231"
    )
    url = f"{api}?{qs}"
    try:
        r = SESS.get(url, timeout=120)
        if r.status_code != 200:
            return []
        data = r.json()
        if not data or len(data) < 2:
            return []
        # first row is header
        out = []
        for row in data[1:]:
            if len(row) >= 2:
                out.append((row[0], row[1]))
        return out
    except Exception as e:
        print(f"  wayback error: {e}", flush=True)
        return []


def discover_nbc() -> list[tuple[str, str]]:
    print("=== NBC: multi-source ===", flush=True)
    out: list[tuple[str, str]] = []

    # Strategy 1: try various sitemap-ish paths first (cheap)
    for cand in [
        "https://www.nbcnews.com/sitemap_news.xml",
        "https://www.nbcnews.com/google-news.xml",
        "https://www.nbcnews.com/sitemaps/news.xml",
        "https://www.nbcnews.com/sitemap.xml",
        "https://www.nbcnews.com/feeds/atom.xml",
    ]:
        data = _get(cand)
        if data:
            print(f"  found sitemap at {cand}", flush=True)
            arts = _crawl_sitemap(cand)
            out.extend(arts)

    # Strategy 2: Wayback CDX — historical NBC article URLs
    print("  hitting wayback CDX for nbcnews.com/* ...", flush=True)
    # Fetch by year to keep response sizes manageable
    for y in range(2018, 2025):
        chunk = _wayback_cdx("nbcnews.com/*", limit=20000, from_year=y, to_year=y)
        # filter to article-shaped URLs
        for u, ts in chunk:
            if re.search(r"nbcnews\.com/.+/.+", u) and "rcna" in u or re.search(r"-n\d{6,}$", u) or re.search(r"-ncna\d+$", u):
                out.append((u, ts[:8]))
        print(f"  wayback {y}: cumulative {len(out)}", flush=True)

    seen = set()
    deduped = []
    for u, lm in out:
        # normalize: drop fragment, query
        u2 = u.split("#")[0].split("?")[0]
        if u2 in seen:
            continue
        seen.add(u2)
        deduped.append((u2, lm))
    print(f"NBC total unique: {len(deduped)}", flush=True)
    return deduped


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    if target in ("fox", "both"):
        fox = discover_fox()
        with open(os.path.join(DATA_DIR, "urls_fox.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "lastmod"])
            w.writerows(fox)
        print(f"wrote data/urls_fox.csv with {len(fox)} rows")
    if target in ("nbc", "both"):
        nbc = discover_nbc()
        with open(os.path.join(DATA_DIR, "urls_nbc.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "lastmod"])
            w.writerows(nbc)
        print(f"wrote data/urls_nbc.csv with {len(nbc)} rows")


if __name__ == "__main__":
    main()
