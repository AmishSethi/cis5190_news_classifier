# CIS 4190/5190 — Track B: Fox vs NBC News Headline Classification

**Final hidden-leaderboard accuracy: 85.75%** (baseline 66.49%, +19.26 pp).

Binary classification of news headlines as Fox News (label 1) vs NBC News (label 0). The hidden test set is an after-deadline scrape from foxnews.com and nbcnews.com, so models are evaluated under temporal drift: any classifier that latches onto specific named entities or current-events vocabulary degrades. Our final system is a 5-member soft-vote ensemble of four transformers with deliberately different inductive biases (DistilBERT, ALBERT, MPNet, ConvBERT) plus a stylometric TF-IDF + 18 hand-features pipeline.

## Headline results

| Approach | Hidden-test accuracy |
|---|---:|
| TF-IDF + Logistic Regression baseline | 66.49% |
| DistilBERT, full 48k scrape | 76.83% |
| DistilBERT on (course + temporal-val) — distribution-matched | 80.58% |
| + best_headline FeatPlusStylo ensemble | 82.25% |
| 2-seed DistilBERT + stylo | 83.25% |
| + ALBERT + MPNet (architectural diversity) | 84.75% |
| + ModernBERT (5 archs) | 85.42% |
| **swap ModernBERT → ConvBERT (final)** | **85.75%** |


- **Dataset (Hugging Face Hub):** [`ASethi04/cis5190-fox-nbc-headlines`](https://huggingface.co/datasets/ASethi04/cis5190-fox-nbc-headlines) — 52,530-headline corpus with the five canonical splits used here.
- **Google Colab Notebook:** https://drive.google.com/file/d/1m6gyeNh1HmYh07ibmigOsUfQr26Rqe1H/view?usp=sharing (Initial models, before adding web scraping & more complexity)
- Video Demo: https://drive.google.com/file/d/1HGUDYYdqQ2j1PSODtFS-up8mzZF8IPl_/view?usp=drive_link

## Quickstart

```bash
# 1. Set up the environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the demo notebook
jupyter lab demo.ipynb
```

## Reproducing the final model

The final submission is a 4-transformer + sklearn-stylo soft-vote ensemble. Reproducing it from scratch takes ~30 minutes on a single GPU.

### Step 1: Build the data splits (skip if you have `data/` populated)

If you want to rebuild from scratch:

```bash
# Discover URLs from outlet sitemaps (one-time, ~5 min)
python scripts/discover_urls.py
python scripts/discover_nbc.py

# Scrape headlines (caches HTML; ~30 min depending on network)
python scripts/scrape_headlines.py --outlet fox --max 80000
python scripts/scrape_headlines.py --outlet nbc --max 50000

# Build the canonical splits
python scripts/build_splits.py
```

The committed `data/` directory already has all five splits ready to use.

### Step 2: Train the four transformers (~15 min on one H100)

Each member is trained on `data/co_val_temp.csv` with the same recipe:

```bash
for ARCH in distilbert-base-uncased albert-base-v2 microsoft/mpnet-base YituTech/conv-bert-base; do
    python scripts/train_classifier_v2.py \
        --model $ARCH \
        --train data/co_val_temp.csv \
        --val data/val_frozen.csv \
        --temp data/temporal_val_frozen.csv \
        --out experiments/${ARCH//\//_} \
        --epochs 6 --batch 32 --lr 2e-5 --max-len 64 \
        --r-drop 1.0 --label-smoothing 0.05 --swa-last 2 --layerwise-lr-decay 0.95
done
```

### Step 3: Train the stylometric pipeline (~30 sec on CPU)

```bash
python scripts/train_best_headline.py \
    --train data/train.csv \
    --out experiments/best_headline/pipe.pkl
```

### Step 4: Package the ensemble

```bash
python scripts/package_ensN.py \
    --model-dirs experiments/distilbert-base-uncased/best \
                 experiments/albert-base-v2/best \
                 experiments/microsoft_mpnet-base/best \
                 experiments/YituTech_conv-bert-base/best \
    --weights 1 1 1 1 \
    --stylo-pkl experiments/best_headline/pipe.pkl \
    --w-sty 1 --max-len 64 \
    --out-dir submission
```

This produces `submission/model.pt`, `submission/model.py`, `submission/preprocess.py`. The course evaluator can then be run with:

```bash
python project-resources/Newsheadlines/eval_project_b.py \
    --model submission/model.py \
    --preprocess submission/preprocess.py \
    --weights submission/model.pt \
    --csv data/temporal_val_frozen.csv
```

## Submission

The leaderboard sandbox provides only `numpy, pandas, torch==2.9.1, torchvision, scikit-learn, opencv-python` — **no `transformers`, `tokenizers`, `datasets`, or `xgboost`**. Tokenizer JSONs are base64-embedded directly into `submission/model.py`, written to a tempdir at first `predict()` call, and loaded via `AutoTokenizer.from_pretrained(tempdir)`. Architectures are reconstructed via `AutoConfig` + `AutoModelForSequenceClassification.from_config` so no `from_pretrained` HTTP call is made. The eval sentinel `weights_path="__no_weights__.pth"` is honored.

The three-file submission contract is:
- `preprocess.py` with `prepare_data(csv_path) -> (X, y)`.
- `model.py` with a `Model` class implementing `predict(batch) -> list` (or callable returning logits).
- `model.pt` loaded via `torch.load(..., map_location="cpu")` then strict-key-matching `load_state_dict`.

## Key findings

1. **Distribution-matched data > more data.** Training on 7.5k carefully selected examples (course + val + temp) outperformed training on the full 48k scraped corpus by ~4 pp because the smaller set better matched the leaderboard test distribution. Adversarial validation between train and temporal-val gave AUC 0.694 — discriminative tokens were named entities (`harris`, `israel`, `biden` vs `kimmel`, `king charles`, `derby`), confirming temporal drift.
2. **Architectural diversity > seed diversity.** A 4-architecture ensemble (DistilBERT + ALBERT + MPNet + ConvBERT) beat any 4-7 multi-seed DistilBERT ensemble. Held-out pairwise correctness correlation: transformers 0.47–0.60; stylo vs. transformers 0.18–0.28.
3. **Ensemble accuracy saturates.** Adding a 6th transformer (DeBERTa-v1, CANINE, XLNet, GPT-2, etc.) consistently regressed: 85.75% → 84.58% (6 models) → 83.42% (7) → 83.08% (8). The optimal recipe is the most *complementary* one, not the largest one.
4. **Style beats topic.** The hand-engineered stylometric pipeline was the single strongest *individual* held-out member (0.819) — it captures topic-independent style features (punctuation density, capitalization patterns, mean word length) that survive temporal drift. Zero-shot Claude Opus 4.7 / GPT-5 with extended reasoning capped at 80% on this task: a frontier ~1T-parameter LLM is beaten by a ~100M-parameter encoder fine-tuned on 7k matched-distribution examples.

## Files of interest

- **`scripts/train_classifier_v2.py`** — fine-tunes any HuggingFace classifier with R-Drop, SWA, label smoothing, and layerwise LR decay. The recipe used for all four transformer members.
- **`scripts/headline_pipeline.py`** — `FeatPlusStylo`: TF-IDF char + word n-grams concatenated with 18 hand-engineered style features.
- **`scripts/package_ensN.py`** — packages an N-transformer + stylo soft-vote ensemble into the three-file submission format with all tokenizer JSONs base64-embedded inline.
- **`scripts/adversarial_val.py`** — quantifies temporal drift between training and temporal-val by training a TF-IDF + LR discriminator (AUC and most-drifted features).
- **`scripts/diversity_search.py`** — pairwise disagreement matrix across all trained members + greedy forward selection for ensemble composition.
- **`scripts/llm_classify.py`** — zero-shot benchmark of Claude Haiku/Sonnet/Opus, GPT-5, and DeepSeek-R1 via OpenRouter.
- **`scripts/contrastive_pretrain.py`** — supervised contrastive pretraining (Khosla et al. 2020) on the 52k-headline corpus.
- **`scripts/train_charcnn.py`** — character-level CNN exploratory model (1D convs over byte embeddings, kernels 2–7).

