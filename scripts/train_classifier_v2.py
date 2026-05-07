"""Same as train_classifier.py but with several extra tricks:

- R-Drop: dual forward pass with dropout; add symmetric KL between the two
  output distributions. Robust regularizer for short-text fine-tuning.
- Optional Stochastic Weight Averaging (SWA): average model weights across the
  last N steps for better generalization.
- Optional layer-wise LR decay (deeper layers get smaller LRs).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = ROOT / "data"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class HeadlineDataset(Dataset):
    def __init__(self, df, tok, max_len=96):
        self.h = df["headline"].astype(str).tolist()
        self.y = df["label"].astype(int).tolist()
        self.tok = tok
        self.max_len = max_len
    def __len__(self):
        return len(self.h)
    def __getitem__(self, i):
        return self.h[i], self.y[i]


def make_collate(tok, max_len):
    def collate(batch):
        return (
            tok([b[0] for b in batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt"),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
        )
    return collate


def evaluate(model, loader, device, use_amp=True):
    model.eval()
    correct = total = 0
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for enc, labels in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), labels)
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1), float(np.mean(losses))


def make_layerwise_param_groups(model, base_lr: float, decay: float, weight_decay: float):
    """Layer-wise LR decay: layer i gets lr * decay**(N - i)."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    # Find encoder layers via name pattern
    layer_pat = re.compile(r"\\.layer\\.(\\d+)\\.|encoder\\.layers?\\.(\\d+)\\.")
    layer_indices = []
    for n, _ in model.named_parameters():
        m = layer_pat.search(n)
        if m:
            for g in m.groups():
                if g is not None:
                    layer_indices.append(int(g))
    n_layers = max(layer_indices) + 1 if layer_indices else 0

    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = layer_pat.search(n)
        if m:
            layer = max(int(g) for g in m.groups() if g is not None)
            scale = decay ** (n_layers - 1 - layer)
        elif "embeddings" in n or "embed_tokens" in n:
            scale = decay ** n_layers
        elif "classifier" in n or "score" in n or "pooler" in n or "head" in n:
            scale = 1.0
        else:
            scale = 1.0
        wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
        key = (round(scale, 4), wd)
        groups.setdefault(key, []).append(p)
    return [
        {"params": ps, "lr": base_lr * scale, "weight_decay": wd}
        for (scale, wd), ps in groups.items()
    ]


def kl_loss(p, q):
    """Symmetric KL divergence between two softmax outputs."""
    p_logsoft = F.log_softmax(p, dim=-1)
    q_logsoft = F.log_softmax(q, dim=-1)
    p_soft = p_logsoft.exp()
    q_soft = q_logsoft.exp()
    return 0.5 * (
        F.kl_div(p_logsoft, q_soft, reduction="batchmean")
        + F.kl_div(q_logsoft, p_soft, reduction="batchmean")
    )


def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, model: {args.model}", flush=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    temp_df = None
    if args.temporal_val_csv and Path(args.temporal_val_csv).exists():
        td = pd.read_csv(args.temporal_val_csv)
        if not td.empty:
            temp_df = td
    print(f"train={len(train_df)}, val={len(val_df)}, temp={len(temp_df) if temp_df is not None else 0}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if hasattr(model, "config") and model.config.pad_token_id is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.to(device)

    train_ds = HeadlineDataset(train_df, tok, args.max_len)
    val_ds = HeadlineDataset(val_df, tok, args.max_len)
    temp_ds = HeadlineDataset(temp_df, tok, args.max_len) if temp_df is not None else None

    labels = np.array(train_df["label"].tolist())
    cc = np.bincount(labels, minlength=2)
    cw = 1.0 / np.maximum(cc, 1)
    sw = cw[labels]
    sampler = WeightedRandomSampler(sw.tolist(), num_samples=len(sw), replacement=True)

    collate = make_collate(tok, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)
    temp_loader = DataLoader(temp_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True) if temp_ds is not None else None

    if args.layerwise_decay < 1.0:
        groups = make_layerwise_param_groups(model, args.lr, args.layerwise_decay, args.weight_decay)
    else:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        groups = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "lr": args.lr, "weight_decay": 0.0},
        ]
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    total_steps = math.ceil(len(train_loader) * args.epochs / args.grad_accum)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    use_amp = not args.no_amp
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    history = []
    step = 0
    t0 = time.time()

    swa_state = None
    swa_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for i, (enc, labels) in enumerate(train_loader):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out1 = model(**enc).logits
                    if args.rdrop > 0:
                        out2 = model(**enc).logits
            else:
                out1 = model(**enc).logits
                if args.rdrop > 0:
                    out2 = model(**enc).logits
            ce1 = loss_fn(out1.float(), labels)
            if args.rdrop > 0:
                ce2 = loss_fn(out2.float(), labels)
                kl = kl_loss(out1.float(), out2.float())
                loss = (ce1 + ce2) / 2 + args.rdrop * kl
            else:
                loss = ce1
            loss = loss / args.grad_accum
            loss.backward()
            running += loss.item()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                # SWA accumulator (last 30% of training)
                if args.swa and step >= int(0.7 * total_steps):
                    if swa_state is None:
                        swa_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
                    else:
                        for k, v in model.state_dict().items():
                            swa_state[k] = swa_state[k] + v.detach().float()
                    swa_count += 1
                if step % args.log_every == 0:
                    print(f"ep{epoch} step{step}/{total_steps} loss={running/args.log_every:.4f} elapsed={time.time()-t0:.1f}s", flush=True)
                    running = 0.0

        val_acc, val_loss = evaluate(model, val_loader, device, use_amp)
        line = {"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss, "step": step, "elapsed_s": time.time() - t0}
        if temp_loader is not None:
            t_acc, t_loss = evaluate(model, temp_loader, device, use_amp)
            line["temp_acc"] = t_acc
            line["temp_loss"] = t_loss
        history.append(line)
        print(f"=== EPOCH {epoch} === {json.dumps(line)}", flush=True)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        score = val_acc + 0.001 * line.get("temp_acc", 0.0)
        if score > best:
            best = score
            print(f"  saving best (val_acc={val_acc:.4f})", flush=True)
            tok.save_pretrained(out_dir / "best")
            model.save_pretrained(out_dir / "best")
            with open(out_dir / "best_metrics.json", "w") as f:
                json.dump(line, f, indent=2)

    # End of training: optionally save SWA model
    if args.swa and swa_state is not None and swa_count > 1:
        swa_state = {k: v / swa_count for k, v in swa_state.items()}
        # Cast back to original dtypes
        orig = model.state_dict()
        for k in swa_state:
            swa_state[k] = swa_state[k].to(orig[k].dtype)
        model.load_state_dict(swa_state)
        # Update batchnorm stats if any (skip for transformer encoders)
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp)
        print(f"=== SWA === val_acc={val_acc:.4f}", flush=True)
        if temp_loader is not None:
            t_acc, _ = evaluate(model, temp_loader, device, use_amp)
            print(f"  swa temp_acc={t_acc:.4f}", flush=True)
        if val_acc > best:
            print(f"  saving SWA best (val_acc={val_acc:.4f})", flush=True)
            tok.save_pretrained(out_dir / "best")
            model.save_pretrained(out_dir / "best")
            with open(out_dir / "best_metrics.json", "w") as f:
                json.dump({"swa": True, "val_acc": val_acc}, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="answerdotai/ModernBERT-large")
    p.add_argument("--train-csv", default=str(DATA / "train.csv"))
    p.add_argument("--val-csv", default=str(DATA / "val.csv"))
    p.add_argument("--temporal-val-csv", default=str(DATA / "temporal_val.csv"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--rdrop", type=float, default=0.0, help="R-Drop alpha (0 disables)")
    p.add_argument("--swa", action="store_true", help="Stochastic Weight Averaging over last 30%% of steps")
    p.add_argument("--layerwise-decay", type=float, default=1.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
