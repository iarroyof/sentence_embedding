# -*- coding: utf-8 -*-
"""
Embed a toy set of sentences with full TF-IDF weighted FastText-style word embeddings
(best performing) and exercise all helpers: vector_space, keyed2indexed, streamer,
save_dense, load_dense, wisse, SentenceEmbedding.encode, similarity.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

# Toy vocab for FastText-style embeddings (same format as word2vec/fasttext .vec)
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


def _write_toy_word2vec_vec(path: Path, vocab: list, dim: int):
    """Write a minimal word2vec .vec (text) file - FastText-compatible format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(vocab)} {dim}\n")
        for w in vocab:
            vec = " ".join(f"{x:.6f}" for x in np.random.randn(dim).astype(np.float32))
            f.write(f"{w} {vec}\n")


def _make_toy_idf(tmp_path: Path, vocab: list):
    tv = TfidfVectorizer(vocabulary=vocab)
    tv.fit(TOY_SENTENCES)
    idf_path = tmp_path / "idf_toy.pkl"
    with open(idf_path, "wb") as f:
        pickle.dump(tv, f)
    return str(idf_path)


@pytest.fixture
def toy_fasttext_indexed_dir(tmp_path):
    """Create toy FastText-style word embeddings and convert to indexed dir via keyed2indexed."""
    import wisse
    from gensim.models.keyedvectors import KeyedVectors

    vec_path = tmp_path / "toy_fasttext.vec"
    _write_toy_word2vec_vec(vec_path, TOY_VOCAB, TOY_DIM)
    kv = KeyedVectors.load_word2vec_format(str(vec_path), binary=False)
    indexed_dir = tmp_path / "indexed"
    wisse.keyed2indexed(kv, str(indexed_dir), parallel=False)
    return str(indexed_dir)


@pytest.fixture
def toy_idf_path(tmp_path):
    return _make_toy_idf(tmp_path, TOY_VOCAB)


def test_manual_tfidf_via_idf_per_feature_matches_transform(
    toy_fasttext_indexed_dir, toy_idf_path
):
    """Legacy path (idf_per_feature + no .transform) must match full sklearn TF×IDF."""
    import wisse

    with open(toy_idf_path, "rb") as f:
        idf = pickle.load(f)
    idf_arr = np.asarray(idf.idf_, dtype=np.float64)
    vs = wisse.vector_space(toy_fasttext_indexed_dir)
    w_sklearn = wisse.wisse(
        vs,
        vectorizer=idf,
        tf_tfidf=True,
        combiner="sum",
        return_missing=False,
        generate=True,
        idf_per_feature=None,
    )
    w_manual = wisse.wisse(
        vs,
        vectorizer=idf,
        tf_tfidf=True,
        combiner="sum",
        return_missing=False,
        generate=True,
        idf_per_feature=idf_arr,
    )
    for s in TOY_SENTENCES + ["hello world embedding test", "unknownword x"]:
        a = w_sklearn.infer_sentence(s)
        b = w_manual.infer_sentence(s)
        if a is None and b is None:
            continue
        assert a is not None and b is not None
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_toy_sentences_tfidf_fasttext_embedding(toy_fasttext_indexed_dir, toy_idf_path):
    """Embed toy sentences with full TF-IDF weighted FastText-style embeddings via SentenceEmbedding."""
    import wisse

    model = wisse.SentenceEmbedding(
        model_name_or_path=toy_fasttext_indexed_dir,
        idf_name_or_path=toy_idf_path,
        combiner="sum",
        use_tfidf_weights=True,
    )
    assert model.dimension == TOY_DIM

    embeddings = model.encode(TOY_SENTENCES)
    assert embeddings.shape == (len(TOY_SENTENCES), TOY_DIM)
    assert embeddings.dtype == np.float32

    sim = model.similarity(embeddings, embeddings)
    assert sim.shape == (len(TOY_SENTENCES), len(TOY_SENTENCES))
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)


def test_all_helpers(
    toy_fasttext_indexed_dir,
    toy_idf_path,
    tmp_path,
):
    """Exercise every helper: vector_space, keyed2indexed, streamer, save_dense, load_dense, wisse, encode, similarity."""
    import wisse

    # --- vector_space ---
    vs = wisse.vector_space(toy_fasttext_indexed_dir)
    assert len(vs) > 0
    assert "hello" in vs
    vec_hello = vs["hello"]
    assert isinstance(vec_hello, np.ndarray)
    assert vec_hello.shape == (TOY_DIM,)

    # --- load_dense / save_dense ---
    test_dir = tmp_path / "save_dense_test"
    test_dir.mkdir()
    wisse.save_dense(str(test_dir), "testword", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    loaded = wisse.load_dense(str(test_dir / "testword.npy"))
    np.testing.assert_array_almost_equal(loaded, [1.0, 2.0, 3.0])

    # --- streamer ---
    sentences_file = tmp_path / "sentences.txt"
    sentences_file.write_text("\n".join(TOY_SENTENCES), encoding="utf-8")
    lines = list(wisse.streamer(str(sentences_file)))
    assert len(lines) == len(TOY_SENTENCES)
    assert lines[0] == TOY_SENTENCES[0]

    # --- wisse (low-level) with TF-IDF ---
    with open(toy_idf_path, "rb") as f:
        idf = pickle.load(f)
    w = wisse.wisse(
        vs,
        vectorizer=idf,
        tf_tfidf=True,
        combiner="sum",
        return_missing=False,
        generate=True,
    )
    out = w.infer_sentence(TOY_SENTENCES[0])
    assert out is not None
    assert isinstance(out, np.ndarray)
    assert out.size == TOY_DIM

    # --- wisse with return_missing ---
    w_rm = wisse.wisse(vs, vectorizer=idf, tf_tfidf=True, combiner="avg", return_missing=True, generate=True)
    missing_cbow, missing_bow, sent_vec = w_rm.infer_sentence(TOY_SENTENCES[0])
    assert isinstance(missing_cbow, list)
    assert isinstance(missing_bow, list)
    assert sent_vec.shape == (TOY_DIM,)

    # --- SentenceEmbedding.encode (single string, normalize) ---
    model = wisse.SentenceEmbedding(
        model_name_or_path=toy_fasttext_indexed_dir,
        idf_name_or_path=toy_idf_path,
        combiner="avg",
        use_tfidf_weights=True,
    )
    single = model.encode("hello world", normalize_embeddings=True)
    assert single.shape == (1, TOY_DIM)
    np.testing.assert_allclose(np.linalg.norm(single[0]), 1.0, atol=1e-5)

    # --- similarity (standalone) ---
    emb = model.encode(TOY_SENTENCES[:2])
    s_cosine = wisse.similarity(emb, emb, similarity_fn="cosine")
    assert s_cosine.shape == (2, 2)
    s_dot = wisse.similarity(emb, emb, similarity_fn="dot")
    assert s_dot.shape == (2, 2)
    s_euclidean = wisse.similarity(emb, emb, similarity_fn="euclidean")
    assert s_euclidean.shape == (2, 2)
    s_manhattan = wisse.similarity(emb, emb, similarity_fn="manhattan")
    assert s_manhattan.shape == (2, 2)
    # Self-similarity with one matrix
    s_self = wisse.similarity(emb, None)
    assert s_self.shape == (2, 2)
    np.testing.assert_allclose(np.diag(s_self), 1.0, atol=1e-5)


def test_legacy_sklearn_idf_diag_recovery():
    """Pre-0.18 TfidfTransformer used _idf_diag; idf_ was not always in __dict__."""
    from scipy import sparse

    from wisse.tfidf_compat import extract_idf_feature_array

    n = 64
    idf = np.log(np.arange(2, n + 2, dtype=np.float64)) + 1.0

    class Inner:
        pass

    inner = Inner()
    inner._idf_diag = sparse.spdiags(idf, 0, n, n, format="csr")

    class FakeVec:
        vocabulary_ = {str(i): i for i in range(n)}
        _tfidf = inner

        def transform(self, _X):
            raise RuntimeError("simulates broken sklearn 1.x unpickle")

    arr = extract_idf_feature_array(FakeVec())
    assert arr is not None
    assert arr.shape == (n,)
    np.testing.assert_allclose(arr, idf, rtol=1e-5)
