# -*- coding: utf-8 -*-
"""
Full workflow test simulating a new user: empty cache, default model keys.
Uses MOCKED download (toy data only). This does NOT use real pretrained FastText/IDF.
For tests with real paper assets, see test_02_paper_wikipedia_assets (requires env or HF).
"""
import pickle
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

# Same toy setup as test_01 (TF-IDF weighted FastText-style)
TOY_VOCAB = [
    "hello", "world", "sentence", "embedding", "test", "one", "two", "three",
    "first", "second", "short", "text",
]
TOY_DIM = 8
TOY_SENTENCES = [
    "hello world",
    "sentence embedding test",
    "one two three",
    "first second sentence",
    "short text",
]


def _build_toy_fasttext_tarball(tmp_path: Path) -> Path:
    """Build a tar.gz containing an indexed embedding dir (one top-level dir with .npy files)."""
    emb_dir = tmp_path / "toy_emb"
    emb_dir.mkdir()
    for w in TOY_VOCAB:
        np.save(emb_dir / f"{w}.npy", np.random.randn(TOY_DIM).astype(np.float32))
    tarball = tmp_path / "fasttext-300-indexed.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(emb_dir, arcname="emb")
    return tarball


def _build_toy_tfidf_pkl(tmp_path: Path) -> Path:
    """Build pickled TfidfVectorizer fit on TOY_SENTENCES (full TF-IDF via .transform())."""
    tv = TfidfVectorizer(vocabulary=TOY_VOCAB)
    tv.fit(TOY_SENTENCES)
    pkl_path = tmp_path / "idf-en.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(tv, f)
    return pkl_path


@pytest.fixture
def new_user_env(tmp_path, monkeypatch):
    """Simulate new user: empty WISSE_HOME, and mock 'download' to serve toy assets."""
    cache_root = tmp_path / "wisse_cache"
    cache_root.mkdir()
    monkeypatch.setenv("WISSE_HOME", str(cache_root))

    toy_tarball = _build_toy_fasttext_tarball(tmp_path)
    toy_pkl = _build_toy_tfidf_pkl(tmp_path)

    original_download = None

    def mock_download(url: str, dest: Path, desc: str = "Downloading") -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.name == "archive.tar.gz" or "fasttext" in url:
            shutil.copy(toy_tarball, dest)
        elif dest.suffix == ".pkl" or "idf" in url:
            shutil.copy(toy_pkl, dest)
        else:
            raise ValueError(f"Unexpected download request: url={url!r} dest={dest}")
        return dest

    import wisse.download as download_module
    monkeypatch.setattr(download_module, "_download_file", mock_download)

    return cache_root


def test_full_workflow_new_user_tfidf_toy_sentences(new_user_env):
    """
    Simulate a completely new user: empty cache, first use.
    Default model (wisse-fasttext-300) and TF-IDF weights (wisse-idf-en) are
    'downloaded' (mocked); then encode same toy sentences with full TF-IDF and run similarity.
    """
    import wisse

    # No args = default embedding + default TF-IDF weights (full TF-IDF, best performing)
    model = wisse.SentenceEmbedding()

    assert model.dimension == TOY_DIM

    # Same toy sentences as in test_01; full TF-IDF weighting is used (tf_tfidf=True)
    embeddings = model.encode(TOY_SENTENCES)
    assert embeddings.shape == (len(TOY_SENTENCES), TOY_DIM)
    assert embeddings.dtype == np.float32

    sim = model.similarity(embeddings, embeddings)
    assert sim.shape == (len(TOY_SENTENCES), len(TOY_SENTENCES))
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    # Single string and normalize
    one = model.encode("hello world", normalize_embeddings=True)
    assert one.shape == (1, TOY_DIM)
    np.testing.assert_allclose(np.linalg.norm(one[0]), 1.0, atol=1e-5)


def test_new_user_uses_tfidf_not_idf_only(new_user_env):
    """Ensure the model uses full TF-IDF (vectorizer.transform), not IDF-only."""
    import wisse

    model = wisse.SentenceEmbedding()
    # Internal wisse instance should have tf_tfidf=True (full TF-IDF)
    assert model._tf_tfidf is True
    assert model._vectorizer is not None
