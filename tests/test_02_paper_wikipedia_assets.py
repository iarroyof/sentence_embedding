# -*- coding: utf-8 -*-
"""
Tests using the paper's Wikipedia-trained assets when available:
- FastText 300d (English Wikipedia), indexed WISSE format
- TF-IDF weights (IDF) trained on English Wikipedia, stop words ignored

Sources (manual download from MEGA if needed):
- FastText indexed: https://mega.nz/#!zKBUzL7J!V2BN6hsb2_I61WbM3C8OIrSnJotFyxaqfBmapddns4Y
- IDF (pretrained_idf): https://mega.nz/#!WPx1iYwA!okha3WRVIksZJuq7cJKeKzplxuDYqOa0aq31hyMHvAo

Alternatively, if hosted on Hugging Face (wisse-models), registry keys
wisse-fasttext-300 and wisse-idf-en are used automatically.
"""
import os
from pathlib import Path

import numpy as np
import pytest

# Sentences that should be in vocabulary of Wikipedia-trained FastText + IDF
PAPER_TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing is a subfield of linguistics.",
    "Wikipedia is a free online encyclopedia.",
]

PAPER_EMBEDDING_DIM = 300


def _get_paper_fasttext_path():
    """Resolve FastText indexed dir: env override, or try HF registry."""
    env_dir = os.environ.get("WISSE_PAPER_FASTTEXT_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return str(p.resolve())
    return None


def _get_paper_idf_path():
    """Resolve IDF path: env override, or try HF registry."""
    env_path = os.environ.get("WISSE_PAPER_IDF_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p.resolve())
    return None


def _paper_assets_available_via_env():
    """True if paper assets are available via local paths (MEGA-downloaded)."""
    return _get_paper_fasttext_path() is not None and _get_paper_idf_path() is not None


@pytest.mark.skipif(
    not _paper_assets_available_via_env(),
    reason=(
        "Paper Wikipedia assets not found. Download from MEGA (see README): "
        "FastText indexed + IDF, then set WISSE_PAPER_FASTTEXT_DIR and WISSE_PAPER_IDF_PATH."
    ),
)
def test_paper_fasttext_idf_encode_and_similarity():
    """Use the paper's Wikipedia FastText + IDF (local paths) to encode and compute similarity."""
    import wisse

    fasttext_path = _get_paper_fasttext_path()
    idf_path = _get_paper_idf_path()
    assert fasttext_path and idf_path

    model = wisse.SentenceEmbedding(
        model_name_or_path=fasttext_path,
        idf_name_or_path=idf_path,
        combiner="sum",
        use_tfidf_weights=True,
    )

    assert model.dimension == PAPER_EMBEDDING_DIM, "Paper uses 300d FastText"

    embeddings = model.encode(PAPER_TEST_SENTENCES)
    assert embeddings.shape == (len(PAPER_TEST_SENTENCES), PAPER_EMBEDDING_DIM)
    assert embeddings.dtype == np.float32

    sim = model.similarity(embeddings, embeddings)
    assert sim.shape == (len(PAPER_TEST_SENTENCES), len(PAPER_TEST_SENTENCES))
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    # Sanity: first and second sentence should have some similarity (both English text)
    assert sim[0, 1] > -0.5 and sim[0, 1] < 1.01


@pytest.mark.skipif(
    not _paper_assets_available_via_env(),
    reason="Paper Wikipedia assets not available (set WISSE_PAPER_FASTTEXT_DIR and WISSE_PAPER_IDF_PATH).",
)
def test_paper_assets_low_level_wisse():
    """Low-level wisse + vector_space with paper FastText and IDF."""
    import wisse

    fasttext_path = _get_paper_fasttext_path()
    idf_path = _get_paper_idf_path()
    if not fasttext_path or not idf_path:
        pytest.skip("Paper assets only via HF registry in this run; low-level test needs local paths.")
    import pickle
    with open(idf_path, "rb") as f:
        idf = pickle.load(f)
    vs = wisse.vector_space(fasttext_path)
    w = wisse.wisse(vs, vectorizer=idf, tf_tfidf=True, combiner="sum", return_missing=False, generate=True)
    out = w.infer_sentence(PAPER_TEST_SENTENCES[0])
    assert out is not None
    assert isinstance(out, np.ndarray)
    assert out.shape == (PAPER_EMBEDDING_DIM,)


@pytest.mark.skipif(
    os.environ.get("WISSE_TEST_HF_REGISTRY", "").lower() not in ("1", "true", "yes"),
    reason="Set WISSE_TEST_HF_REGISTRY=1 to run tests with Hugging Face registry (network).",
)
def test_paper_assets_via_huggingface_registry():
    """
    Use the same Wikipedia FastText + IDF via Hugging Face registry (wisse-fasttext-300, wisse-idf-en).
    Run when HF repo has the paper assets: WISSE_TEST_HF_REGISTRY=1 pytest tests/test_02_paper_wikipedia_assets.py -v
    """
    import wisse

    model = wisse.SentenceEmbedding(
        model_name_or_path="wisse-fasttext-300",
        idf_name_or_path="wisse-idf-en",
        combiner="sum",
        use_tfidf_weights=True,
    )
    assert model.dimension == PAPER_EMBEDDING_DIM
    embeddings = model.encode(PAPER_TEST_SENTENCES)
    assert embeddings.shape == (len(PAPER_TEST_SENTENCES), PAPER_EMBEDDING_DIM)
    sim = model.similarity(embeddings, embeddings)
    assert sim.shape == (len(PAPER_TEST_SENTENCES), len(PAPER_TEST_SENTENCES))
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)
