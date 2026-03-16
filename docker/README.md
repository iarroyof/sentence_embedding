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

## What the container does

1. **At build:** Installs `wisse-sentence` with the `train` extra (adds `datasets`). No training.
2. **At run:** Runs `wisse-train --wikipedia en` with `CAP_ARTICLES` and `CAP_TOKENS` (default 2000 / 5M).
3. Writes IDF to `/workspace/output/idf-en.pkl` and embeddings to `/workspace/output/fasttext-300-indexed/`.
4. Runs `train_and_demo.py`: registers assets under `WISSE_HOME` and runs the toy example.

## One-liner (build + run)

```bash
docker build -f docker/Dockerfile -t wisse-train . && docker run --rm wisse-train
```
