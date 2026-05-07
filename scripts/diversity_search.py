"""Find the most error-diverse model subset for ensembling.

Predicts every trained model on val + temp, computes pairwise disagreement,
and runs forward greedy selection by ensemble accuracy + minimum-overlap
heuristic to find the best 3-6 model ensemble.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


sys.path.insert(0, "scripts")


def get_proba(model_dir, texts, batch_size=64, max_len=64, device="cuda:0"):
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    probs = []
    import inspect
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            b = texts[i:i+batch_size]
            enc = tok(b, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            allowed = set(inspect.signature(m.forward).parameters.keys())
            enc = {k: v for k, v in enc.items() if k in allowed}
            out = m(**enc).logits
            probs.append(F.softmax(out, dim=-1).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.vstack(probs)


def main():
    val = pd.read_csv("data/val_frozen.csv")
    temp = pd.read_csv("data/temporal_val_frozen.csv")
    yv = val["label"].to_numpy(); yt = temp["label"].to_numpy()
    Xv = val["headline"].astype(str).tolist()
    Xt = temp["headline"].astype(str).tolist()

    # All models trained on (course+val+temp) — val accuracy is inflated, but
    # disagreement patterns are still meaningful.
    transformers_to_score = {
        "distilbert_s7": "experiments/distilbert_covt_seed7/best",
        "distilbert_s99": "experiments/distilbert_covt_seed99/best",
        "distilbert_s11": "experiments/distilbert_covt_seed11/best",
        "distilbert_layerwise": "experiments/distilbert_covt_layerwise/best",
        "distilbert_megareg": "experiments/distilbert_covt_megareg/best",
        "distilbert_smooth": "experiments/distilbert_covt_smooth/best",
        "distilbert_long": "experiments/distilbert_covt_long/best",
        "distilbert_random10k": "experiments/distilbert_random10k/best",
        "albert": "experiments/albert_covt/best",
        "albert_random10k": "experiments/albert_random10k/best",
        "mpnet": "experiments/mpnet_covt/best",
        "mpnet_random10k": "experiments/mpnet_random10k/best",
        "convbert": "experiments/convbert_covt/best",
        "convbert_s99": "experiments/convbert_covt_seed99/best",
        "convbert_random10k": "experiments/convbert_random10k/best",
        "modernbert": "experiments/modernbert_covt/best",
        "bert": "experiments/bert_co_val_temp/best",
        "bert_large": "experiments/bert_large_covt/best",
        "electra": "experiments/electra_covt/best",
        "distilroberta": "experiments/distilroberta_covt/best",
        "xlmr": "experiments/xlmr_covt/best",
        "funnel": "experiments/funnel_covt/best",
        "contrastive_db": "experiments/contrastive_db_covt/best",
    }

    out_dir = Path("experiments/diversity_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache all probs
    pv_dict = {}
    pt_dict = {}
    for name, path in transformers_to_score.items():
        if not os.path.exists(path):
            print(f"  skip {name}: missing")
            continue
        pv_path = out_dir / f"pv_{name}.npy"
        pt_path = out_dir / f"pt_{name}.npy"
        if pv_path.exists() and pt_path.exists():
            pv_dict[name] = np.load(pv_path)
            pt_dict[name] = np.load(pt_path)
            print(f"  cached {name}: val={(pv_dict[name].argmax(1) == yv).mean():.4f}", flush=True)
            continue
        print(f"  scoring {name}...", flush=True)
        try:
            pv = get_proba(path, Xv)
            pt = get_proba(path, Xt)
            pv_dict[name] = pv; pt_dict[name] = pt
            np.save(pv_path, pv); np.save(pt_path, pt)
            print(f"    {name}: val={(pv.argmax(1) == yv).mean():.4f}  temp={(pt.argmax(1) == yt).mean():.4f}", flush=True)
        except Exception as e:
            print(f"    {name} FAILED: {e}", flush=True)

    # Add stylo (sklearn pipeline)
    import pickle
    from headline_pipeline import FeatPlusStylo, StyloTransformer, FeatThenModel
    sys.modules["__main__"].FeatPlusStylo = FeatPlusStylo
    sys.modules["__main__"].StyloTransformer = StyloTransformer
    sys.modules["__main__"].FeatThenModel = FeatThenModel
    with open("experiments/best_headline/pipe.pkl", "rb") as f:
        sty = pickle.load(f)
    pv_dict["stylo"] = sty.predict_proba(Xv)
    pt_dict["stylo"] = sty.predict_proba(Xt)
    print(f"  stylo: val={(pv_dict['stylo'].argmax(1) == yv).mean():.4f}  temp={(pt_dict['stylo'].argmax(1) == yt).mean():.4f}")

    # Add CharCNN (custom torch model)
    try:
        sys.path.insert(0, "submission_FINAL_85.75/..")
        import importlib.util
        cnn_path = "submission_178_charcnn"
        spec = importlib.util.spec_from_file_location("charcnn_module", f"{cnn_path}/model.py")
        cm = importlib.util.module_from_spec(spec)
        # The CharCNN model.py auto-loads from cwd. Set cwd to the submission dir.
        old_cwd = os.getcwd()
        os.chdir(cnn_path)
        spec.loader.exec_module(cm)
        os.chdir(old_cwd)
        m = cm.get_model()
        # Get probs not just preds — modify ad-hoc
        # CharCNN model in submission has predict() returning hard labels. We'll re-derive probs.
        # Easier: just use 0.95/0.05 hard probs from preds for now.
        pv_cnn_pred = np.array(m.predict(Xv))
        pt_cnn_pred = np.array(m.predict(Xt))
        pv_dict["charcnn"] = np.zeros((len(yv), 2))
        pt_dict["charcnn"] = np.zeros((len(yt), 2))
        for i, p in enumerate(pv_cnn_pred):
            pv_dict["charcnn"][i] = [0.05, 0.95] if p == 1 else [0.95, 0.05]
        for i, p in enumerate(pt_cnn_pred):
            pt_dict["charcnn"][i] = [0.05, 0.95] if p == 1 else [0.95, 0.05]
        print(f"  charcnn: val={(pv_dict['charcnn'].argmax(1) == yv).mean():.4f}")
    except Exception as e:
        print(f"  charcnn skipped: {e}")

    print(f"\nTotal models: {len(pv_dict)}")

    # Pairwise disagreement matrix on temp (since temp is the test-distribution proxy)
    names = list(pv_dict.keys())
    print("\n=== Pairwise disagreement on temp (% of examples where they disagree) ===")
    # disagreement[i,j] = fraction of examples where argmax differs
    n = len(names)
    disagree = np.zeros((n, n))
    pred_v = {n_: pv_dict[n_].argmax(1) for n_ in names}
    pred_t = {n_: pt_dict[n_].argmax(1) for n_ in names}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            disagree[i, j] = (pred_t[ni] != pred_t[nj]).mean()
    # Print as matrix
    print(f"{'':30s} " + " ".join(f"{n[:6]:>7s}" for n in names))
    for i, ni in enumerate(names):
        print(f"{ni[:30]:30s} " + " ".join(f"{disagree[i,j]:7.3f}" for j in range(n)))

    # Forward greedy selection: maximize ensemble TEMP accuracy
    def ensemble_acc(idxs, X="temp"):
        if X == "val":
            probs = np.mean([pv_dict[names[i]] for i in idxs], axis=0)
            return (probs.argmax(1) == yv).mean()
        else:
            probs = np.mean([pt_dict[names[i]] for i in idxs], axis=0)
            return (probs.argmax(1) == yt).mean()

    print("\n=== Greedy forward selection (maximize TEMP ensemble accuracy) ===")
    selected = []
    remaining = list(range(n))
    while remaining and len(selected) < 8:
        best_i = None; best_acc = -1
        for i in remaining:
            acc = ensemble_acc(selected + [i])
            if acc > best_acc:
                best_acc = acc; best_i = i
        selected.append(best_i)
        remaining.remove(best_i)
        val_acc = ensemble_acc(selected, "val")
        print(f"  add {names[best_i]:30s}  ensemble: val={val_acc:.4f} temp={best_acc:.4f}  size={len(selected)}")

    # Also try: pick top-K by individual temp accuracy that span max architectural categories
    print("\n=== Architectural-diversity heuristic: best per arch family ===")
    families = {
        "distilbert": [n for n in names if n.startswith("distilbert")],
        "albert": [n for n in names if n.startswith("albert")],
        "mpnet": [n for n in names if n.startswith("mpnet")],
        "convbert": [n for n in names if n.startswith("convbert")],
        "modernbert": ["modernbert"] if "modernbert" in names else [],
        "bert": [n for n in names if n.startswith("bert") and n != "bert_large"] + (["bert_large"] if "bert_large" in names else []),
        "electra": ["electra"] if "electra" in names else [],
        "distilroberta": ["distilroberta"] if "distilroberta" in names else [],
        "xlmr": ["xlmr"] if "xlmr" in names else [],
        "funnel": ["funnel"] if "funnel" in names else [],
        "contrastive_db": ["contrastive_db"] if "contrastive_db" in names else [],
        "charcnn": ["charcnn"] if "charcnn" in names else [],
        "stylo": ["stylo"] if "stylo" in names else [],
    }
    # For each family, pick the best (highest individual temp accuracy)
    best_in_family = {}
    for fam, members in families.items():
        if not members: continue
        accs = [(m, (pred_t[m] == yt).mean()) for m in members]
        accs.sort(key=lambda x: -x[1])
        best_in_family[fam] = accs[0][0]
        print(f"  {fam:15s}: {accs[0][0]:30s}  temp={accs[0][1]:.4f}")

    # Now greedy select among family bests
    print("\n=== Greedy forward over family bests ===")
    fam_names = list(best_in_family.values())
    fam_idxs = [names.index(n) for n in fam_names]
    selected_fam = []
    remaining_fam = fam_idxs.copy()
    while remaining_fam and len(selected_fam) < 8:
        best_i = None; best_acc = -1
        for i in remaining_fam:
            acc = ensemble_acc(selected_fam + [i])
            if acc > best_acc:
                best_acc = acc; best_i = i
        selected_fam.append(best_i)
        remaining_fam.remove(best_i)
        val_acc = ensemble_acc(selected_fam, "val")
        print(f"  add {names[best_i]:30s}  ensemble: val={val_acc:.4f} temp={best_acc:.4f}  size={len(selected_fam)}")


if __name__ == "__main__":
    main()
