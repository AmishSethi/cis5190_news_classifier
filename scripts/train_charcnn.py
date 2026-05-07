"""Character-level CNN for headline classification.

Tokenizes at byte level (uint8), embeds each char as 32-dim vector, applies
multi-scale 1D convolutions (kernel sizes 2,3,4,5), max-pools, and projects
to logits. Captures char-level style cues (capitalization, punctuation
patterns, n-gram morphology) that token-level transformers may miss.
"""
from __future__ import annotations

import argparse
import json
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


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


VOCAB_SIZE = 256  # bytes


class CharDataset(Dataset):
    def __init__(self, texts, labels, max_len=160):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        b = self.texts[i].encode("utf-8", errors="replace")[: self.max_len]
        x = np.zeros(self.max_len, dtype=np.int64)
        for j, byte in enumerate(b):
            x[j] = byte
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.long)


class CharCNN(nn.Module):
    def __init__(self, embed_dim=64, num_filters=128, kernel_sizes=(2, 3, 4, 5, 6, 7), num_classes=2, dropout=0.4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # 2 stylo features added: length, fraction of uppercase
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes) + 2, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, L]
        e = self.embed(x).transpose(1, 2)  # [B, D, L]
        feats = []
        for conv in self.convs:
            f = F.relu(conv(e))
            f = F.max_pool1d(f, f.size(2)).squeeze(2)  # [B, F]
            feats.append(f)
        h = torch.cat(feats, dim=1)
        # stylo: length + uppercase frac
        length = (x != 0).sum(dim=1, keepdim=True).float() / 160.0
        upper = ((x >= 65) & (x <= 90)).sum(dim=1, keepdim=True).float() / (length * 160 + 1e-6)
        h = torch.cat([h, length, upper], dim=1)
        h = self.dropout(F.relu(self.fc1(h)))
        return self.fc2(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/co_val_temp.csv")
    ap.add_argument("--val-csv", default="data/val_frozen.csv")
    ap.add_argument("--temp-csv", default="data/temporal_val_frozen.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--num-filters", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    temp_df = pd.read_csv(args.temp_csv)
    Xtr = train_df["headline"].astype(str).tolist()
    ytr = train_df["label"].astype(int).to_numpy()
    Xv = val_df["headline"].astype(str).tolist(); yv = val_df["label"].astype(int).to_numpy()
    Xt = temp_df["headline"].astype(str).tolist(); yt = temp_df["label"].astype(int).to_numpy()
    print(f"train={len(Xtr)} val={len(Xv)} temp={len(Xt)}")

    train_ds = CharDataset(Xtr, ytr, args.max_len)
    val_ds = CharDataset(Xv, yv, args.max_len)
    temp_ds = CharDataset(Xt, yt, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=128, num_workers=2)
    temp_dl = DataLoader(temp_ds, batch_size=128, num_workers=2)

    model = CharCNN(
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        dropout=args.dropout,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))

    best_val = 0; best_state = None
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        # Eval
        model.eval()
        with torch.no_grad():
            preds_v, preds_t = [], []
            for x, _ in val_dl:
                preds_v.append(model(x.to(device)).argmax(1).cpu().numpy())
            for x, _ in temp_dl:
                preds_t.append(model(x.to(device)).argmax(1).cpu().numpy())
            pv = np.concatenate(preds_v)
            pt = np.concatenate(preds_t)
            av = (pv == yv).mean(); at = (pt == yt).mean()
        elapsed = time.time() - t0
        print(f"ep{ep:2d} val={av:.4f} temp={at:.4f} ({elapsed:.1f}s)", flush=True)
        history.append({"epoch": ep, "val_acc": float(av), "temp_acc": float(at)})
        if av > best_val:
            best_val = av
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"best val_acc: {best_val:.4f}")
    torch.save({
        "state_dict": best_state,
        "config": {
            "embed_dim": args.embed_dim,
            "num_filters": args.num_filters,
            "dropout": args.dropout,
            "max_len": args.max_len,
        },
    }, out / "model.pt")
    with open(out / "history.json", "w") as f:
        json.dump({"history": history, "best_val_acc": best_val, "config": vars(args)}, f, indent=2)
    print(f"saved {out}/model.pt")


if __name__ == "__main__":
    main()
