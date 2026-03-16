#!/usr/bin/env python3
"""
After training: register trained assets under WISSE_HOME so default registry
keys (wisse-fasttext-300, wisse-idf-en) use them, then run a toy example.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

WISSE_HOME = Path(os.environ.get("WISSE_HOME", "/workspace/models"))
TRAIN_IDF = Path("/workspace/output/idf-en.pkl")
TRAIN_EMBEDDINGS = Path("/workspace/output/fasttext-300-indexed")


def register_assets() -> None:
    """Copy trained outputs into WISSE cache layout so SentenceEmbedding() uses them."""
    WISSE_HOME.mkdir(parents=True, exist_ok=True)

    # IDF: .../idf/wisse-idf-en.pkl
    idf_dest = WISSE_HOME / "idf" / "wisse-idf-en.pkl"
    idf_dest.parent.mkdir(parents=True, exist_ok=True)
    if TRAIN_IDF.exists():
        shutil.copy2(TRAIN_IDF, idf_dest)
        print(f"Registered IDF: {idf_dest}")
    else:
        print(f"Warning: {TRAIN_IDF} not found (run training first)")

    # Embeddings: .../embeddings/wisse-fasttext-300/extracted/*.npy
    emb_extracted = WISSE_HOME / "embeddings" / "wisse-fasttext-300" / "extracted"
    emb_extracted.mkdir(parents=True, exist_ok=True)
    if TRAIN_EMBEDDINGS.is_dir():
        for f in TRAIN_EMBEDDINGS.glob("*.npy"):
            shutil.copy2(f, emb_extracted / f.name)
        print(f"Registered embeddings: {emb_extracted} ({len(list(emb_extracted.glob('*.npy')))} .npy files)")
    else:
        print(f"Warning: {TRAIN_EMBEDDINGS} not found (run training first)")


def run_toy_example() -> None:
    """Use default SentenceEmbedding() (registry keys) and encode toy sentences."""
    from wisse import SentenceEmbedding
    import numpy as np

    print("\n--- Toy example (using registered assets) ---")
    model = SentenceEmbedding()
    sentences = [
        "The weather is lovely today.",
        "It is so sunny outside.",
        "He drove to the stadium.",
    ]
    embeddings = model.encode(sentences)
    print(f"Encoded {len(sentences)} sentences -> shape {embeddings.shape}")
    sim = model.similarity(embeddings, embeddings)
    print("Pairwise cosine similarity (diagonal = 1.0):")
    print(sim)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)
    print("OK: toy example passed.\n")


if __name__ == "__main__":
    register_assets()
    run_toy_example()
