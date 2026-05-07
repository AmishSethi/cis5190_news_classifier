"""Domain-adaptive masked-language-model pretraining on news headlines.

Takes all_clean.csv (every labeled headline we have, but we ignore labels here)
plus any extra unlabeled headlines and continues MLM on a base encoder for ~1
epoch. Saves the resulting model directory which can then be fine-tuned for
classification.

Why this helps: our classification training sees ~10k examples, but MLM can
see all ~150k headlines. The encoder learns domain-specific token
distributions, sentence-completion patterns, and entity priors that the
downstream classifier benefits from.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ROOT / "data"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=96):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, max_length=self.max_len, padding=False,
        )
        return {k: v for k, v in enc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-csv", default=str(DATA / "all_clean.csv"))
    p.add_argument("--extra-csvs", nargs="*", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    # Collect headlines from all sources
    texts: list[str] = []
    df = pd.read_csv(args.data_csv)
    if "headline" in df.columns:
        texts.extend(df["headline"].dropna().astype(str).tolist())
    for extra in args.extra_csvs:
        if not Path(extra).exists():
            continue
        df2 = pd.read_csv(extra)
        if "headline" in df2.columns:
            texts.extend(df2["headline"].dropna().astype(str).tolist())
    # dedupe + filter short
    seen = set()
    deduped = []
    for t in texts:
        t = t.strip()
        if len(t) < 10 or t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    print(f"MLM corpus: {len(deduped)} unique headlines", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.mask_token is None:
        # Some causal LMs don't have mask tokens; fall back to a sentinel
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(device)

    ds = TextDataset(deduped, tokenizer, args.max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(grouped, lr=args.lr)
    total_steps = math.ceil(len(loader) * args.epochs)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp

    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**batch)
            else:
                out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            running += loss.item()
            if step % 50 == 0:
                print(f"ep{epoch} step{step}/{total_steps} loss={running/50:.4f} elapsed={time.time()-t0:.1f}s", flush=True)
                running = 0.0
        # save end of each epoch
        ck = out_dir / f"epoch{epoch}"
        ck.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(ck)
        model.save_pretrained(ck)
        print(f"saved epoch {epoch} -> {ck}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
