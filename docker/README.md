# Docker: train IDF + FastText from Wikipedia and run toy example

Build and run from the **repository root** (parent of `docker/`).

## Commands

```bash
# 1. From repo root: build the image (installs wisse + train deps, runs Wikipedia training with small cap)
docker build -f docker/Dockerfile -t wisse-train .

# 2. Run the container: registers trained assets under WISSE_HOME and runs the toy example
docker run --rm wisse-train
```

## Optional: keep trained models on a volume

```bash
# Create a volume and run; models will persist in the volume
docker run --rm -v wisse-models:/workspace/models wisse-train

# Later: run again using the same volume (skip training, only register + demo)
# (To skip training you’d need a different image or entrypoint; this image always trains then runs demo.)
```

## Optional: larger training (more articles/tokens)

```bash
docker run --rm \
  -e CAP_ARTICLES=50000 \
  -e CAP_TOKENS=20000000 \
  wisse-train
```

## What the container does

1. Installs `wisse-sentence` with the `train` extra (adds `datasets` for Wikipedia).
2. Runs `wisse-train --wikipedia en` with a small cap (2000 articles / 5M tokens by default) so the image builds in reasonable time.
3. Writes IDF to `/workspace/output/idf-en.pkl` and embeddings to `/workspace/output/fasttext-300-indexed/`.
4. Runs `train_and_demo.py`, which:
   - Copies those outputs into `$WISSE_HOME` in the layout expected by the default registry (`wisse-fasttext-300`, `wisse-idf-en`).
   - Calls `SentenceEmbedding()` with no arguments (so it uses the registered assets) and encodes three toy sentences, then prints pairwise similarity.

## One-liner (build + run)

```bash
docker build -f docker/Dockerfile -t wisse-train . && docker run --rm wisse-train
```
