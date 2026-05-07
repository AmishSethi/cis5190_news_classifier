"""Supervised contrastive pretraining on all_clean.csv (52k labeled headlines).

Uses SimCSE-style contrastive: each headline is a query; positives are other
headlines of the SAME label, negatives are different label. Trains a DistilBERT
encoder so that within-class headlines cluster in embedding space.

Then fine-tunes a classifier head on co_val_temp.csv.

Architecture:
1. Pretrain: contrastive on all_clean (~52k, no holdouts to worry about since
   val_frozen / temp_val are subsets that we're allowed to see)
2. Fine-tune: cls head with cross-entropy on co_val_temp (~7k high-quality)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], int(self.labels[i])


def supcon_loss(features, labels, temperature=0.07):
    """Supervised contrastive loss (Khosla et al. 2020).

    features: [B, D] L2-normalized
    labels: [B]
    """
    device = features.device
    B = features.shape[0]
    sim = torch.matmul(features, features.T) / temperature  # [B, B]
    # Mask out self-similarity
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, -1e9)
    # Positive mask: same label, not self
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = labels_eq & ~self_mask
    # SupCon: -log( sum(exp(sim)*pos) / sum(exp(sim)) )
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # Mean log_prob over positives (skip rows with no positives)
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1)[valid] / n_pos[valid].float()
    return -mean_log_prob_pos.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-csv", default="data/all_clean.csv")
    ap.add_argument("--finetune-csv", default="data/co_val_temp.csv")
    ap.add_argument("--val-csv", default="data/val_frozen.csv")
    ap.add_argument("--temp-csv", default="data/temporal_val_frozen.csv")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--pretrain-batch", type=int, default=128)
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--pretrain-lr", type=float, default=3e-5)
    ap.add_argument("--ft-batch", type=int, default=64)
    ap.add_argument("--ft-epochs", type=int, default=4)
    ap.add_argument("--ft-lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ============== Stage 1: Contrastive pretraining ==============
    print(f"=== Pretraining on {args.pretrain_csv} ===")
    pre_df = pd.read_csv(args.pretrain_csv)
    Xp = pre_df["headline"].astype(str).tolist()
    yp = pre_df["label"].astype(int).to_numpy()
    print(f"pretrain examples: {len(Xp)} ({(yp == 1).sum()} Fox, {(yp == 0).sum()} NBC)")

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModel.from_pretrained(args.model).to(device)
    proj = nn.Linear(base.config.hidden_size, args.proj_dim).to(device)

    def encode_batch(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
        out = base(**enc).last_hidden_state[:, 0, :]  # [CLS]
        return out

    pre_ds = TextDataset(Xp, yp)
    pre_dl = DataLoader(pre_ds, batch_size=args.pretrain_batch, shuffle=True,
                        collate_fn=lambda b: (
                            [x[0] for x in b], torch.tensor([x[1] for x in b], dtype=torch.long)
                        ))

    opt = torch.optim.AdamW(list(base.parameters()) + list(proj.parameters()),
                             lr=args.pretrain_lr, weight_decay=1e-4)
    n_steps = args.pretrain_epochs * len(pre_dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    base.train()
    proj.train()
    t0 = time.time()
    log_every = max(50, len(pre_dl) // 5)
    for ep in range(1, args.pretrain_epochs + 1):
        ep_loss = 0; n_batches = 0
        for step, (texts, labels) in enumerate(pre_dl):
            labels = labels.to(device)
            cls = encode_batch(texts)
            z = F.normalize(proj(cls), dim=1)
            loss = supcon_loss(z, labels, temperature=args.temperature)
            if loss.item() == 0:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(base.parameters()) + list(proj.parameters()), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item(); n_batches += 1
            if step % log_every == 0:
                el = time.time() - t0
                print(f"  pretrain ep{ep} step{step}/{len(pre_dl)} loss={loss.item():.4f} ({el:.0f}s)", flush=True)
        print(f"  ep{ep} avg loss = {ep_loss/max(n_batches,1):.4f}", flush=True)

    # Save the pretrained encoder
    pre_out = out / "pretrained"
    pre_out.mkdir(exist_ok=True)
    base.save_pretrained(str(pre_out))
    tok.save_pretrained(str(pre_out))

    # ============== Stage 2: Supervised fine-tune ==============
    print(f"\n=== Fine-tuning on {args.finetune_csv} ===")
    ft_df = pd.read_csv(args.finetune_csv)
    val_df = pd.read_csv(args.val_csv)
    temp_df = pd.read_csv(args.temp_csv)
    Xft = ft_df["headline"].astype(str).tolist()
    yft = ft_df["label"].astype(int).to_numpy()
    Xv = val_df["headline"].astype(str).tolist(); yv = val_df["label"].astype(int).to_numpy()
    Xt = temp_df["headline"].astype(str).tolist(); yt = temp_df["label"].astype(int).to_numpy()

    cfg = AutoConfig.from_pretrained(str(pre_out))
    cfg.num_labels = 2
    cls_model = AutoModelForSequenceClassification.from_pretrained(str(pre_out), config=cfg).to(device)

    def eval_split(model, X, y):
        model.eval()
        preds = []
        bsz = 128
        with torch.no_grad():
            for i in range(0, len(X), bsz):
                enc = tok(X[i:i+bsz], padding=True, truncation=True,
                          max_length=args.max_len, return_tensors="pt").to(device)
                p = model(**enc).logits.argmax(dim=1).cpu().numpy()
                preds.append(p)
        return np.concatenate(preds)

    ft_ds = TextDataset(Xft, yft)
    ft_dl = DataLoader(ft_ds, batch_size=args.ft_batch, shuffle=True,
                       collate_fn=lambda b: (
                           [x[0] for x in b], torch.tensor([x[1] for x in b], dtype=torch.long)
                       ))
    opt2 = torch.optim.AdamW(cls_model.parameters(), lr=args.ft_lr, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.ft_epochs * len(ft_dl))

    best_val = 0; best_state = None; history = []
    for ep in range(1, args.ft_epochs + 1):
        cls_model.train()
        t0 = time.time()
        for texts, labels in ft_dl:
            enc = tok(texts, padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            labels = labels.to(device)
            logits = cls_model(**enc).logits
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cls_model.parameters(), 1.0)
            opt2.step()
            sched2.step()
        pv = eval_split(cls_model, Xv, yv)
        pt = eval_split(cls_model, Xt, yt)
        av = (pv == yv).mean(); at = (pt == yt).mean()
        elapsed = time.time() - t0
        print(f"  ft ep{ep} val={av:.4f} temp={at:.4f} ({elapsed:.1f}s)", flush=True)
        history.append({"epoch": ep, "val_acc": float(av), "temp_acc": float(at)})
        if av > best_val:
            best_val = av
            best_state = {k: v.cpu().clone() for k, v in cls_model.state_dict().items()}

    cls_model.load_state_dict(best_state)
    best_dir = out / "best"
    best_dir.mkdir(exist_ok=True)
    cls_model.save_pretrained(str(best_dir))
    tok.save_pretrained(str(best_dir))
    with open(out / "best_metrics.json", "w") as f:
        json.dump({"val_acc": best_val, "history": history}, f, indent=2)
    print(f"\nbest val_acc: {best_val:.4f}")
    print(f"saved {best_dir}")


if __name__ == "__main__":
    main()
