#!/bin/sh
# Run training at container start (not at build). Use env vars to control scale.
#
# Default quick run:
#   docker run --rm wisse-train
#
# Large training on host disk (mount /mnt/wisse-training -> /data):
#   docker run --rm -v /mnt/wisse-training:/data \
#     -e WISSE_TRAINING_ROOT=/data -e CAP_TOKENS=6000000000 -e WORKERS=8 wisse-train

set -e
CAP_ARTICLES="${CAP_ARTICLES:-2000}"
CAP_TOKENS="${CAP_TOKENS:-5000000}"
WIKIPEDIA_LANG="${WIKIPEDIA_LANG:-en}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-3}"

if [ "${CAP_TOKENS}" -gt 1000000000 ] 2>/dev/null; then
  CAP_ARTICLES=20000000
fi

if [ -n "${WISSE_TRAINING_ROOT}" ]; then
  mkdir -p "${WISSE_TRAINING_ROOT}/corpus" "${WISSE_TRAINING_ROOT}/models"
  IDF_OUT="${IDF_OUT:-${WISSE_TRAINING_ROOT}/models/idf-${WIKIPEDIA_LANG}.pkl}"
  EMB_OUT="${EMBEDDINGS_OUT:-${WISSE_TRAINING_ROOT}/models/fasttext-300-indexed}"
  if [ -n "${SENTENCE_CORPUS}" ]; then
    CORPUS_ARG="--sentence-corpus ${SENTENCE_CORPUS}"
  else
    CORPUS_ARG="--sentence-corpus ${WISSE_TRAINING_ROOT}/corpus/wiki-${WIKIPEDIA_LANG}-sentences.txt"
  fi
  export TRAIN_IDF="${IDF_OUT}"
  export TRAIN_EMBEDDINGS="${EMB_OUT}"
else
  IDF_OUT="${IDF_OUT:-/workspace/output/idf-en.pkl}"
  EMB_OUT="${EMBEDDINGS_OUT:-/workspace/output/fasttext-300-indexed}"
  CORPUS_ARG=""
  export TRAIN_IDF="${IDF_OUT}"
  export TRAIN_EMBEDDINGS="${EMB_OUT}"
fi

echo "Training from Wikipedia (${WIKIPEDIA_LANG}): cap_articles=${CAP_ARTICLES} cap_tokens=${CAP_TOKENS} workers=${WORKERS} epochs=${EPOCHS}"
if [ -n "${WISSE_TRAINING_ROOT}" ]; then
  echo "WISSE_TRAINING_ROOT=${WISSE_TRAINING_ROOT} (streaming sentence file + models on mounted volume)"
fi
if [ "${CAP_TOKENS}" -gt 15000000 ] 2>/dev/null; then
  echo "Note: streaming uses a large on-disk sentence corpus — ensure enough free space on the volume."
fi

if [ -n "${CORPUS_ARG}" ]; then
  # shellcheck disable=SC2086
  wisse-train --wikipedia "${WIKIPEDIA_LANG}" \
      --document-unit article \
      --idf-out "${IDF_OUT}" \
      --embeddings-out "${EMB_OUT}" \
      ${CORPUS_ARG} \
      --cap-articles "${CAP_ARTICLES}" \
      --cap-tokens "${CAP_TOKENS}" \
      --dim 300 --window 5 --min-count 5 --epochs "${EPOCHS}" \
      --workers "${WORKERS}"
else
  wisse-train --wikipedia "${WIKIPEDIA_LANG}" \
      --document-unit article \
      --idf-out "${IDF_OUT}" \
      --embeddings-out "${EMB_OUT}" \
      --cap-articles "${CAP_ARTICLES}" \
      --cap-tokens "${CAP_TOKENS}" \
      --dim 300 --window 5 --min-count 5 --epochs "${EPOCHS}" \
      --workers "${WORKERS}"
fi

echo "Registering assets and running toy example..."
exec python /workspace/train_and_demo.py
