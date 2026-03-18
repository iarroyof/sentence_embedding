#!/bin/sh
# Run training at container start (not at build). Use env vars to control scale.
# Example: docker run --rm -e CAP_TOKENS=6000000000 wisse-train

set -e
CAP_ARTICLES="${CAP_ARTICLES:-2000}"
CAP_TOKENS="${CAP_TOKENS:-5000000}"
WIKIPEDIA_LANG="${WIKIPEDIA_LANG:-en}"

# If token cap is large (e.g. 6B/16B), use high article cap so token limit is the one that stops
if [ "${CAP_TOKENS}" -gt 1000000000 ] 2>/dev/null; then
  CAP_ARTICLES=20000000
fi

echo "Training from Wikipedia (${WIKIPEDIA_LANG}): cap_articles=${CAP_ARTICLES} cap_tokens=${CAP_TOKENS}"
if [ "${CAP_TOKENS}" -gt 15000000 ] 2>/dev/null; then
  echo "Note: large cap_tokens uses streaming (temp sentence file on disk; ensure enough free space)."
fi
wisse-train --wikipedia "${WIKIPEDIA_LANG}" \
    --document-unit article \
    --idf-out /workspace/output/idf-en.pkl \
    --embeddings-out /workspace/output/fasttext-300-indexed \
    --cap-articles "${CAP_ARTICLES}" \
    --cap-tokens "${CAP_TOKENS}" \
    --dim 300 --window 5 --min-count 5 --epochs 3

echo "Registering assets and running toy example..."
exec python /workspace/train_and_demo.py
