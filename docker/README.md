# Docker: train IDF + FastText from Wikipedia and run toy example

Build and run from the **repository root** (parent of `docker/`). **Training runs when you start the container**, not at build time, so you can train with different caps without rebuilding.

## Commands

```bash
# 1. Build the image once (only installs deps + code; no training)
docker build -f docker/Dockerfile -t wisse-train .

# 2. Run: train with default cap (2000 articles / 5M tokens), then register + toy demo
docker run --rm wisse-train
```

## Optional: keep trained models on a volume

```bash
# Create a volume and run; models will persist in the volume
docker run --rm -v wisse-models:/workspace/models wisse-train

# Later: run again using the same volume (skip training, only register + demo)
# (To skip training you’d need a different image or entrypoint; this image always trains then runs demo.)
```

## Train another model with a different scale (no rebuild)

Use the **same image**; pass env vars. No need to download or install again.

**Quick larger run:**
```bash
docker run --rm -e CAP_ARTICLES=50000 -e CAP_TOKENS=20000000 wisse-train
```

**Final training at published embedding-era scale:**
```bash
# 6B tokens (GloVe 2014: Wikipedia + Gigaword)
docker run --rm -e CAP_TOKENS=6000000000 wisse-train

# 16B tokens (FastText Wikipedia+news)
docker run --rm -e CAP_TOKENS=16000000000 wisse-train
```

**Persist models on a volume:**
```bash
docker run --rm -v wisse-models:/workspace/models -e CAP_TOKENS=6000000000 wisse-train
```

**Large training on a host directory (e.g. `/mnt/wisse-training` on blue-demon):** mount it and set `WISSE_TRAINING_ROOT` so the sentence stream + IDF + embeddings use the big disk (not container layer).

```bash
# From repo root (after: docker build -f docker/Dockerfile -t wisse-train .)
docker run --rm \
  -v /mnt/wisse-training:/data \
  -e WISSE_TRAINING_ROOT=/data \
  -e CAP_TOKENS=6000000000 \
  -e WORKERS=8 \
  -e EPOCHS=5 \
  wisse-train
```

Outputs on the host:

- `/mnt/wisse-training/corpus/wiki-en-sentences.txt` — streaming sentence file  
- `/mnt/wisse-training/models/idf-en.pkl`  
- `/mnt/wisse-training/models/fasttext-300-indexed/`  

Optional: `-e SENTENCE_CORPUS=/data/corpus/custom.txt` — `-e IDF_OUT=...` — `-e EMBEDDINGS_OUT=...` — `-e WIKIPEDIA_LANG=es`.

**16B tokens:** `-e CAP_TOKENS=16000000000` (ensure hundreds of GB free on `/mnt`).

## Similarity sampling (no Python on the host)

The image includes `scripts/sample_sentence_similarities.py`. Mount your data and **override the entrypoint** so training does not run:

```bash
# Rebuild once if your image predates the scripts/ COPY in the Dockerfile
docker build -f docker/Dockerfile -t wisse-train .

docker run --rm \
  -v /mnt/wisse-training:/data \
  --entrypoint python \
  wisse-train \
  /workspace/scripts/sample_sentence_similarities.py \
  --sentence-corpus /data/corpus/wiki-en-sentences.txt \
  --model /data/models/fasttext-300-indexed \
  --idf /data/models/idf-en.pkl \
  --output /data/models/sentence_pair_similarities.txt
```

Writes the report to the host at `/mnt/wisse-training/models/sentence_pair_similarities.txt`. Adjust paths if your mount differs.

## What the container does

1. **At build:** Installs `wisse-sentence` with the `train` extra (adds `datasets`). No training.
2. **At run:** Runs `wisse-train --wikipedia en` with `CAP_ARTICLES` and `CAP_TOKENS` (default 2000 / 5M).
3. Writes IDF to `/workspace/output/idf-en.pkl` and embeddings to `/workspace/output/fasttext-300-indexed/`.
4. Runs `train_and_demo.py`: registers assets under `WISSE_HOME` and runs the toy example.

**Large caps (e.g. 6B+ tokens):** Training uses the **streaming pipeline** automatically (`--cap-tokens` > 15M): one pass writes sentences to a temp file — ensure the container has **enough free disk** (tens–100+ GB for multi-billion tokens). Logs show `Streaming pass` / `Pass done` instead of loading the full corpus in RAM.

**How to confirm caps:** `Training from Wikipedia ... cap_articles=... cap_tokens=...`. Then `Pass done: ... docs, ... sentences` (streaming) or `Corpus: X documents...` (small in-memory run). For 16B tokens expect a large vocab and a very long run.

## One-liner (build + run)

```bash
docker build -f docker/Dockerfile -t wisse-train . && docker run --rm wisse-train
```
