"""LLM-based classification of headlines as Fox vs NBC.

Uses OpenRouter API. Few-shot prompting with calibrated examples.
Outputs P(Fox|headline) so we can build soft labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_PROMPT = """You are an expert at distinguishing news writing styles. Given a single news headline, you must classify it as coming from Fox News or NBC News.

Output exactly one of: FOX or NBC

Style cues:
- Fox News: tends toward populist/conservative framing, often dramatic, uses words like "blasts", "slams", emphasizes individual figures
- NBC News: tends toward establishment/centrist framing, more neutral phrasing, uses "exclusive", "experts", "according to", more institutional voice

Be decisive. If uncertain, default to your best guess based on style alone (not topic)."""


FEW_SHOT_FOX = [
    "AOC blasts 'morally reprehensible' Republicans over border policy",
    "Liberal media meltdown after Trump victory speech",
    "WATCH: Crowd erupts as MAGA supporters storm rally",
]

FEW_SHOT_NBC = [
    "Experts warn climate change could displace millions by 2050",
    "Federal Reserve cuts rates amid economic uncertainty, study finds",
    "Civil rights groups condemn proposed legislation, citing concerns",
]


def build_prompt(headlines, with_few_shot=True):
    """Build a single user message asking to classify a batch of headlines."""
    if with_few_shot:
        msg = "Classify each headline as FOX or NBC. Reply with only N labels separated by spaces, one per headline.\n\n"
        msg += "Examples:\n"
        for h in FEW_SHOT_FOX:
            msg += f"  FOX: {h}\n"
        for h in FEW_SHOT_NBC:
            msg += f"  NBC: {h}\n"
        msg += "\nNow classify these headlines. Output exactly N labels separated by spaces:\n"
    else:
        msg = "Classify each headline as FOX or NBC. Output exactly N labels separated by spaces:\n\n"
    for i, h in enumerate(headlines, 1):
        msg += f"{i}. {h}\n"
    msg += "\nLabels (space-separated):"
    return msg


def call_llm(model, messages, key, max_tokens=200, temperature=0.0, retries=3, logprobs=False):
    """Call OpenRouter chat completion. Returns text + cost + raw."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # GPT-5 and other reasoning models need minimal effort + higher max_tokens
    if "gpt-5" in model or "o1" in model or "o3" in model or "o4" in model:
        body["reasoning_effort"] = "minimal"
        body["max_tokens"] = max(max_tokens, 2000)
    if "deepseek-r1" in model or "reasoner" in model:
        body["max_tokens"] = max(max_tokens, 4000)
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "choices" not in d or not d["choices"]:
                last_err = f"no choices: {str(d)[:200]}"
                time.sleep(2 ** attempt)
                continue
            text = d["choices"][0]["message"].get("content")
            if text is None:
                # Try reasoning field as fallback for some reasoning models
                text = d["choices"][0]["message"].get("reasoning", "")
            cost = d.get("usage", {}).get("cost", 0.0)
            return text, cost, d
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:200]}"
            if e.code == 429:
                time.sleep(5 + 5 * attempt)
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_llm failed after {retries} retries: {last_err}")


def parse_batch_response(text, n_expected):
    """Parse LLM response. Expects N labels (FOX/NBC) space-separated."""
    text = text.strip()
    # Extract just FOX/NBC tokens
    tokens = [t.strip(".,;:- \n").upper() for t in text.replace("\n", " ").split()]
    labels = []
    for t in tokens:
        if t.startswith("FOX") or t == "F":
            labels.append(1)
        elif t.startswith("NBC") or t == "N":
            labels.append(0)
        if len(labels) >= n_expected:
            break
    return labels


def classify_batch(args):
    headlines, idx_offset, model, key, with_few_shot = args
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(headlines, with_few_shot=with_few_shot)},
    ]
    text, cost, _ = call_llm(model, messages, key, max_tokens=10 * len(headlines) + 50)
    labels = parse_batch_response(text, len(headlines))
    if len(labels) != len(headlines):
        # Truncate or pad with -1 (unknown)
        if len(labels) < len(headlines):
            labels = labels + [-1] * (len(headlines) - len(labels))
        else:
            labels = labels[:len(headlines)]
    return idx_offset, labels, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-few-shot", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")

    df = pd.read_csv(args.csv)
    headlines = df["headline"].astype(str).tolist()
    if args.limit > 0:
        headlines = headlines[: args.limit]
    print(f"classifying {len(headlines)} headlines with {args.model}")

    # Build batches
    batches = []
    for i in range(0, len(headlines), args.batch_size):
        batch = headlines[i : i + args.batch_size]
        batches.append((batch, i, args.model, key, not args.no_few_shot))

    results = [None] * len(headlines)
    total_cost = 0.0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify_batch, b): b[1] for b in batches}
        done = 0
        for fut in as_completed(futures):
            try:
                idx_offset, labels, cost = fut.result()
                for j, l in enumerate(labels):
                    if idx_offset + j < len(results):
                        results[idx_offset + j] = l
                total_cost += cost
                done += 1
                if done % 10 == 0 or done == len(batches):
                    el = time.time() - t0
                    print(f"  {done}/{len(batches)} batches ({el:.0f}s, ${total_cost:.3f})", flush=True)
            except Exception as e:
                idx_offset = futures[fut]
                print(f"  batch at {idx_offset} failed: {e}", flush=True)

    # Save
    valid = [r for r in results if r is not None and r >= 0]
    print(f"\ngot {len(valid)}/{len(results)} predictions, total cost ${total_cost:.3f}")

    out_df = df.iloc[: len(headlines)].copy()
    out_df["llm_label"] = [r if r is not None else -1 for r in results]
    out_df.to_csv(args.out, index=False)
    print(f"saved {args.out}")

    # If labels in original df, report agreement
    if "label" in out_df.columns:
        mask = (out_df["llm_label"] >= 0)
        agree = (out_df.loc[mask, "llm_label"] == out_df.loc[mask, "label"]).mean()
        print(f"agreement with ground truth: {agree:.4f} ({mask.sum()} valid)")


if __name__ == "__main__":
    main()
