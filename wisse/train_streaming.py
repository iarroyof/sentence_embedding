# -*- coding: utf-8 -*-
"""
Streaming training: one pass writes sentences to disk; document frequencies for
IDF stay in memory; FastText trains from corpus_file (low RAM vs materializing
all sentences).
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Above this token cap (or with --streaming), use streaming pipeline
STREAMING_TOKEN_THRESHOLD = 15_000_000


def _build_tfidf_from_document_freq(
    df: Dict[str, int],
    n_docs: int,
    min_df: int = 1,
    max_df: float = 1.0,
    max_features: Optional[int] = None,
) -> TfidfVectorizer:
    """
    Build TfidfVectorizer matching sklearn smooth_idf=True idf formula.
    Input space-joined lowercase tokens (same as wisse inference).
    """
    if n_docs < 1:
        raise ValueError("n_docs must be >= 1")

    terms: List[Tuple[str, int]] = []
    max_df_docs = (max_df * n_docs) if isinstance(max_df, float) and max_df <= 1.0 else float("inf")

    for term, dfi in df.items():
        if dfi < min_df:
            continue
        if dfi > max_df_docs:
            continue
        terms.append((term, dfi))

    if not terms:
        raise RuntimeError("No terms passed min_df/max_df filters for IDF")

    if max_features is not None and len(terms) > max_features:
        terms.sort(key=lambda x: -x[1])
        terms = terms[:max_features]
    terms.sort(key=lambda x: x[0])

    vocab = {t: i for i, (t, _) in enumerate(terms)}
    idf = np.array(
        [np.log((1 + n_docs) / (1 + df[t])) + 1 for t, _ in terms],
        dtype=np.float64,
    )

    vec = TfidfVectorizer(
        vocabulary=vocab,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )
    # Minimal fit so CountVectorizer / pipeline is valid; then replace idf
    sample = " ".join(list(vocab.keys())[: min(500, len(vocab))])
    vec.fit([sample] if sample else ["placeholder"])
    vec._tfidf.idf_ = idf
    vec._tfidf._idf_diag = sp.spdiags(idf, 0, len(idf), len(idf), format="csr")
    return vec


def stream_corpus_pass(
    doc_iterator: Iterator[Tuple[List[str], List[List[str]]]],
    sentence_file: Any,
    cap_articles: Optional[int],
    cap_tokens: Optional[int],
) -> Tuple[Dict[str, int], int, int, int, int]:
    """
    Single pass: update document frequencies; write sentences (>=2 tokens) as
    space-separated lines. Returns (df dict, n_docs, n_sentences, n_tokens, n_lines_written).
    """
    df: Dict[str, int] = {}
    n_docs = 0
    n_sentences = 0
    n_tokens = 0
    n_lines = 0

    for doc_tokens, sentence_tokens in doc_iterator:
        if cap_articles is not None and n_docs >= cap_articles:
            break
        if not doc_tokens:
            continue
        seen = set(doc_tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1
        n_docs += 1

        for sent in sentence_tokens:
            if len(sent) < 2:
                continue
            line = " ".join(sent)
            sentence_file.write(line + "\n")
            n_lines += 1
            n_sentences += 1
            n_tokens += len(sent)
            if cap_tokens is not None and n_tokens >= cap_tokens:
                return dict(df), n_docs, n_sentences, n_tokens, n_lines

    return dict(df), n_docs, n_sentences, n_tokens, n_lines


def iter_wikipedia_sequential(
    language: str,
    document_unit: str,
    cap_articles: Optional[int],
    cap_tokens: Optional[int],
    seed: int,
) -> Iterator[Tuple[List[str], List[List[str]]]]:
    """Wikipedia stream (shuffled buffer); sequential until caps — no reservoir."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Training from Wikipedia requires: pip install datasets") from e

    from wisse.train import WIKI_CONFIG_DATE, _doc_paragraphs, _sentence_split, _tokenize_for_fasttext

    config = f"{WIKI_CONFIG_DATE}.{language}"
    logger.info("Streaming Wikipedia %s (sequential until caps; not reservoir sampling)", config)
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=50_000)

    total_tokens = 0
    n_docs = 0
    for item in ds:
        if cap_articles is not None and n_docs >= cap_articles:
            break
        text = (item.get("text") or "").strip()
        if not text:
            continue

        if document_unit == "article":
            doc_tokens = _tokenize_for_fasttext(text)
            sents = _sentence_split(text)
            sentence_tokens = [_tokenize_for_fasttext(s) for s in sents if s]
            if not doc_tokens or not sentence_tokens:
                continue
            yield (doc_tokens, sentence_tokens)
            n_docs += 1
            total_tokens += sum(len(s) for s in sentence_tokens)
            if cap_tokens is not None and total_tokens >= cap_tokens:
                break
        else:
            for para in _doc_paragraphs(text):
                if cap_articles is not None and n_docs >= cap_articles:
                    return
                doc_tokens = _tokenize_for_fasttext(para)
                sents = _sentence_split(para)
                sentence_tokens = [_tokenize_for_fasttext(s) for s in sents if s]
                if not doc_tokens or not sentence_tokens:
                    continue
                yield (doc_tokens, sentence_tokens)
                n_docs += 1
                total_tokens += sum(len(s) for s in sentence_tokens)
                if cap_tokens is not None and total_tokens >= cap_tokens:
                    return


def _import_tokenizers():
    from wisse.train import _doc_paragraphs, _sentence_split, _tokenize_for_fasttext

    return _tokenize_for_fasttext, _sentence_split, _doc_paragraphs


def iter_dir_streaming(
    corpus_dir: str,
    document_unit: str,
    cap_articles: Optional[int],
    cap_tokens: Optional[int],
    encoding: str = "utf-8",
) -> Iterator[Tuple[List[str], List[List[str]]]]:
    _tokenize_for_fasttext, _sentence_split, _doc_paragraphs = _import_tokenizers()
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    n_docs = 0
    total_tokens = 0
    for fp in sorted(corpus_path.rglob("*")):
        if not fp.is_file():
            continue
        if cap_articles is not None and n_docs >= cap_articles:
            break
        try:
            text = fp.read_text(encoding=encoding, errors="replace").strip()
        except Exception as e:
            logger.warning("Skip %s: %s", fp, e)
            continue
        if not text:
            continue

        if document_unit == "article":
            doc_tokens = _tokenize_for_fasttext(text)
            sents = _sentence_split(text)
            sentence_tokens = [_tokenize_for_fasttext(s) for s in sents if s]
            if doc_tokens and sentence_tokens:
                yield (doc_tokens, sentence_tokens)
                n_docs += 1
                total_tokens += sum(len(s) for s in sentence_tokens)
        else:
            for para in _doc_paragraphs(text):
                doc_tokens = _tokenize_for_fasttext(para)
                sents = _sentence_split(para)
                sentence_tokens = [_tokenize_for_fasttext(s) for s in sents if s]
                if doc_tokens and sentence_tokens:
                    yield (doc_tokens, sentence_tokens)
                    n_docs += 1
                    total_tokens += sum(len(s) for s in sentence_tokens)
        if cap_tokens is not None and total_tokens >= cap_tokens:
            break


def run_train_streaming(
    corpus_dir: Optional[str],
    wikipedia_lang: Optional[str],
    document_unit: str,
    idf_out: str,
    embeddings_out: str,
    binary_out: Optional[str],
    dim: int,
    window: int,
    min_count: int,
    epochs: int,
    workers: int,
    cap_articles: Optional[int],
    cap_tokens: Optional[int],
    seed: int,
    idf_min_df: int = 1,
    idf_max_df: float = 1.0,
    idf_max_features: Optional[int] = None,
    sentence_corpus_path: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Streaming training: temp sentence file + df in RAM + FastText from file.
    """
    from gensim.models import FastText
    from .wisse import keyed2indexed

    if corpus_dir:
        it = iter_dir_streaming(
            corpus_dir, document_unit, cap_articles, cap_tokens
        )
    else:
        it = iter_wikipedia_sequential(
            wikipedia_lang or "en",
            document_unit,
            cap_articles,
            cap_tokens,
            seed,
        )

    own_temp = sentence_corpus_path is None
    if own_temp:
        fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="wisse_sentences_")
        os.close(fd)
        sentence_path = tmp_path
    else:
        sentence_path = sentence_corpus_path
        Path(sentence_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            "Streaming pass: writing sentences to %s (caps: articles=%s tokens=%s)",
            sentence_path,
            cap_articles,
            cap_tokens,
        )
        with open(sentence_path, "w", encoding="utf-8", buffering=1024 * 1024) as sf:
            df, n_docs, n_sents, n_tok, n_lines = stream_corpus_pass(
                it, sf, cap_articles, cap_tokens
            )

        if n_docs < 1 or n_lines < 1:
            raise RuntimeError("Streaming produced no documents or sentences")

        logger.info(
            "Pass done: %d docs, %d sentences, ~%d tokens, %d unique terms (df keys)",
            n_docs,
            n_sents,
            n_tok,
            len(df),
        )

        logger.info("Building TF-IDF vectorizer from document frequencies...")
        vec = _build_tfidf_from_document_freq(
            df, n_docs, min_df=idf_min_df, max_df=idf_max_df, max_features=idf_max_features
        )
        idf_p = Path(idf_out)
        idf_p.parent.mkdir(parents=True, exist_ok=True)
        with open(idf_p, "wb") as f:
            pickle.dump(vec, f)
        logger.info("Saved IDF to %s (vocab size %d)", idf_p, len(vec.vocabulary_))

        del df
        logger.info(
            "Training FastText from %s (dim=%d, epochs=%d, workers=%d)",
            sentence_path,
            dim,
            epochs,
            workers,
        )
        model = FastText(
            vector_size=dim,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,
            epochs=epochs,
            negative=5,
        )
        model.build_vocab(corpus_file=sentence_path)
        model.train(
            corpus_file=sentence_path,
            epochs=epochs,
            total_examples=model.corpus_count,
            total_words=model.corpus_total_words,
        )

        out_path = Path(embeddings_out)
        out_path.mkdir(parents=True, exist_ok=True)
        keyed2indexed(model.wv, str(out_path), parallel=workers > 1)
        logger.info(
            "Saved indexed embeddings (vocabulary size: %d words)",
            len(model.wv),
        )
        if binary_out:
            model.wv.save_word2vec_format(binary_out, binary=True)
            logger.info("Saved binary to %s", binary_out)

        return idf_p, out_path
    finally:
        if own_temp and os.path.isfile(sentence_path):
            try:
                os.remove(sentence_path)
            except OSError:
                pass
