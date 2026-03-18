# -*- coding: utf-8 -*-
"""Streaming training pipeline (large caps, disk-backed sentences)."""
import pickle
from pathlib import Path


def test_run_train_streaming_corpus_dir(tmp_path):
    from wisse.train import run_train

    (tmp_path / "doc1.txt").write_text(
        "alpha beta gamma. delta beta.\n\nSecond paragraph here.",
        encoding="utf-8",
    )
    idf_out = tmp_path / "idf.pkl"
    emb_out = tmp_path / "indexed"
    run_train(
        corpus_dir=str(tmp_path),
        document_unit="article",
        streaming=True,
        cap_tokens=10_000,
        dim=8,
        window=2,
        min_count=1,
        epochs=1,
        workers=1,
        idf_out=str(idf_out),
        embeddings_out=str(emb_out),
    )
    assert idf_out.is_file()
    with open(idf_out, "rb") as f:
        vec = pickle.load(f)
    assert len(vec.vocabulary_) >= 4
    assert (emb_out / "alpha.npy").is_file() or len(list(emb_out.glob("*.npy"))) >= 1


def test_build_tfidf_from_df_matches_doc_scope():
    from wisse.train_streaming import _build_tfidf_from_document_freq

    # Two docs: doc1 {a,b}, doc2 {b,c} -> df a=1,b=2,c=1
    df = {"a": 1, "b": 2, "c": 1}
    vec = _build_tfidf_from_document_freq(df, n_docs=2, min_df=1)
    assert set(vec.vocabulary_.keys()) == {"a", "b", "c"}
