# -*- coding: utf-8 -*-
"""Additional tests for encode(), similarity(), and low-level API."""
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


def _make_fake_indexed_embedding_dir(tmp_path, dim=8):
    words = ["hello", "world", "sentence", "embedding", "test", "one", "two", "three"]
    d = tmp_path / "emb"
    d.mkdir()
    for w in words:
        np.save(d / f"{w}.npy", np.random.randn(dim).astype(np.float32))
    return str(d)


def _make_fake_idf(tmp_path, vocab=None):
    if vocab is None:
        vocab = ["hello", "world", "sentence", "embedding", "test", "one", "two", "three"]
    tv = TfidfVectorizer(vocabulary=vocab)
    tv.fit([" ".join(vocab)])
    p = tmp_path / "idf.pkl"
    with open(p, "wb") as f:
        pickle.dump(tv, f)
    return str(p)


def test_similarity_standalone():
    from wisse import similarity
    a = np.random.randn(5, 10).astype(np.float32)
    b = np.random.randn(3, 10).astype(np.float32)
    s = similarity(a, b, similarity_fn="cosine")
    assert s.shape == (5, 3)
    assert np.all(s >= -1.01) and np.all(s <= 1.01)
    s2 = similarity(a, None)
    assert s2.shape == (5, 5)
    np.testing.assert_allclose(np.diag(s2), 1.0, atol=1e-5)


def test_similarity_euclidean():
    from wisse import similarity
    a = np.random.randn(4, 6).astype(np.float32)
    s = similarity(a, a, similarity_fn="euclidean")
    assert s.shape == (4, 4)
    np.testing.assert_allclose(np.diag(s), 0.0, atol=1e-5)


def test_wisse_encode_with_local_paths(tmp_path):
    emb_dir = _make_fake_indexed_embedding_dir(tmp_path, dim=8)
    idf_path = _make_fake_idf(tmp_path)
    from wisse import SentenceEmbedding
    model = SentenceEmbedding(
        model_name_or_path=emb_dir,
        idf_name_or_path=idf_path,
        combiner="sum",
        use_tfidf_weights=True,
    )
    assert model.dimension == 8
    sentences = ["hello world", "sentence embedding", "test one two"]
    emb = model.encode(sentences)
    assert emb.shape == (3, 8)
    assert emb.dtype == np.float32
    sim = model.similarity(emb, emb)
    assert sim.shape == (3, 3)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)


def test_encode_single_string(tmp_path):
    emb_dir = _make_fake_indexed_embedding_dir(tmp_path, dim=5)
    idf_path = _make_fake_idf(tmp_path)
    from wisse import SentenceEmbedding
    model = SentenceEmbedding(
        model_name_or_path=emb_dir,
        idf_name_or_path=idf_path,
    )
    emb = model.encode("hello world")
    assert emb.shape == (1, 5)


def test_encode_normalize(tmp_path):
    emb_dir = _make_fake_indexed_embedding_dir(tmp_path, dim=4)
    idf_path = _make_fake_idf(tmp_path)
    from wisse import SentenceEmbedding
    model = SentenceEmbedding(
        model_name_or_path=emb_dir,
        idf_name_or_path=idf_path,
    )
    emb = model.encode(["hello world"], normalize_embeddings=True)
    assert emb.shape == (1, 4)
    np.testing.assert_allclose(np.linalg.norm(emb[0]), 1.0, atol=1e-5)


def test_vector_space_and_wisse_low_level(tmp_path):
    emb_dir = _make_fake_indexed_embedding_dir(tmp_path, dim=6)
    idf_path = _make_fake_idf(tmp_path)
    import wisse
    with open(idf_path, "rb") as f:
        idf = pickle.load(f)
    vs = wisse.vector_space(emb_dir)
    w = wisse.wisse(vs, vectorizer=idf, tf_tfidf=True, combiner="sum", return_missing=False, generate=True)
    out = w.infer_sentence("hello world test")
    assert out is not None
    assert isinstance(out, np.ndarray)
    assert out.size == 6
