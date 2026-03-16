# -*- coding: utf-8 -*-
"""
Train IDF (TF-IDF) and FastText from a directory of text files or the Wikipedia
dataset (Hugging Face). Produces WISSE-ready IDF pickle and indexed embeddings.
"""
from __future__ import annotations

import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple


from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Paper-style defaults (configurable via CLI)
DEFAULT_DIM = 300
DEFAULT_WINDOW = 5
DEFAULT_MIN_COUNT = 5
DEFAULT_EPOCHS = 5
# Typical scale for 2015–2019: ~500k articles or ~100M tokens
DEFAULT_CAP_ARTICLES = 500_000
DEFAULT_CAP_TOKENS = 100_000_000

# HF Wikipedia config: date.language (e.g. 20231101.en)
WIKI_CONFIG_DATE = "20231101"


def _tokenize_for_fasttext(text: str) -> List[str]:
    """Simple tokenizer: lowercase, alphanumeric tokens (no extra deps)."""
    return re.findall(r"\b\w+\b", text.lower())


def _sentence_split(text: str) -> List[str]:
    """Split text into sentence-like chunks (simple, no NLTK)."""
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [c.strip() for c in chunks if c.strip()]


def _doc_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (double newline or single newline blocks)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# -----------------------------------------------------------------------------
# Corpus: directory of plain text files
# -----------------------------------------------------------------------------


def iter_corpus_from_dir(
    corpus_dir: str,
    document_unit: str,
    encoding: str = "utf-8",
) -> Iterator[Tuple[List[str], List[List[str]]]]:
    """
    Yield (doc_tokens, list_of_sentence_tokens) for each document.
    document_unit: "article" (one file = one doc) or "paragraph" (file split by \\n\\n).
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    for fp in sorted(corpus_path.rglob("*")):
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(encoding=encoding, errors="replace")
        except Exception as e:
            logger.warning("Skip %s: %s", fp, e)
            continue
        text = text.strip()
        if not text:
            continue

        if document_unit == "article":
            doc_tokens = _tokenize_for_fasttext(text)
            sents = _sentence_split(text)
            sentence_tokens = [_tokenize_for_fasttext(s) for s in sents if s]
            yield (doc_tokens, sentence_tokens)
        else:  # paragraph
            for para in _doc_paragraphs(text):
                doc_tokens = _tokenize_for_fasttext(para)
                sents = _sentence_split(para)
                sentence_tokens = [_tokenize_for_fasttext(s) for s in sents if s]
                if doc_tokens and sentence_tokens:
                    yield (doc_tokens, sentence_tokens)


# -----------------------------------------------------------------------------
# Corpus: Hugging Face Wikipedia (streaming, optional cap)
# -----------------------------------------------------------------------------


def _wiki_article_pairs(
    ds: Any,  # datasets.IterableDataset
    document_unit: str,
    cap_tokens: Optional[int],
) -> Iterator[Tuple[List[str], List[List[str]]]]:
    """Yield (doc_tokens, sentence_tokens) from HF Wikipedia dataset."""
    total_tokens = 0
    for item in ds:
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
            total_tokens += len(doc_tokens)
        else:  # paragraph
            for para in _doc_paragraphs(text):
                doc_tokens = _tokenize_for_fasttext(para)
                sents = _sentence_split(para)
                sentence_tokens = [_tokenize_for_fasttext(s) for s in sents if s]
                if not doc_tokens or not sentence_tokens:
                    continue
                yield (doc_tokens, sentence_tokens)
                total_tokens += len(doc_tokens)
                if cap_tokens and total_tokens >= cap_tokens:
                    return
        if cap_tokens and total_tokens >= cap_tokens:
            return


def iter_corpus_from_wikipedia(
    language: str = "en",
    document_unit: str = "article",
    cap_articles: Optional[int] = None,
    cap_tokens: Optional[int] = None,
    seed: int = 42,
) -> Iterator[Tuple[List[str], List[List[str]]]]:
    """
    Stream Wikipedia from HF datasets. Yields (doc_tokens, list_of_sentence_tokens).
    Uses reservoir sampling when cap_articles is set.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Training from Wikipedia requires: pip install datasets"
        ) from e

    config = f"{WIKI_CONFIG_DATE}.{language}"
    logger.info("Loading Wikipedia config: %s (streaming)", config)
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=50_000)
    stream = _wiki_article_pairs(ds, document_unit, cap_tokens)

    if cap_articles is None:
        for pair in stream:
            yield pair
        return

    # Reservoir sampling
    rng = random.Random(seed)
    reservoir: List[Tuple[List[str], List[List[str]]]] = []
    n_seen = 0
    for pair in stream:
        n_seen += 1
        if len(reservoir) < cap_articles:
            reservoir.append(pair)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < cap_articles:
                reservoir[j] = pair
    rng.shuffle(reservoir)
    for pair in reservoir:
        yield pair


# -----------------------------------------------------------------------------
# Train IDF (TfidfVectorizer) and save
# -----------------------------------------------------------------------------


def train_idf(
    doc_iterator: Iterator[Tuple[List[str], List[List[str]]]],
    output_path: str,
    max_features: Optional[int] = None,
    min_df: int = 1,
    max_df: float = 1.0,
    tokenizer: Any = None,
) -> Path:
    """
    Fit TfidfVectorizer on documents (each doc = space-joined tokens) and save pickle.
    """
    if tokenizer is None:
        tokenizer = TfidfVectorizer().build_tokenizer()

    docs_for_idf: List[str] = []
    for doc_tokens, _ in doc_iterator:
        docs_for_idf.append(" ".join(doc_tokens))

    logger.info("Fitting IDF on %d documents", len(docs_for_idf))
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        tokenizer=tokenizer,
        token_pattern=None,
    )
    vectorizer.fit(docs_for_idf)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved IDF to %s", out)
    return out


# -----------------------------------------------------------------------------
# Train FastText and export to WISSE indexed (+ optional binary)
# -----------------------------------------------------------------------------


def _sentence_iterator(
    doc_iterator: Iterator[Tuple[List[str], List[List[str]]]],
) -> Iterator[List[str]]:
    """Flatten (doc, sentences) iterator to stream of sentence token lists."""
    for _, sentence_tokens in doc_iterator:
        for sent in sentence_tokens:
            if len(sent) >= 2:  # skip very short
                yield sent


def collect_corpus(
    doc_iterator: Iterator[Tuple[List[str], List[List[str]]]],
    cap_articles: Optional[int] = None,
    cap_tokens: Optional[int] = None,
) -> Tuple[List[str], List[List[str]]]:
    """
    Consume the iterator and return (list of doc strings for IDF, list of sentence
    token lists for FastText). Applies cap_articles/cap_tokens if set.
    """
    docs: List[str] = []
    sentences: List[List[str]] = []
    n_docs = 0
    n_tokens = 0
    for doc_tokens, sents in doc_iterator:
        docs.append(" ".join(doc_tokens))
        for s in sents:
            if len(s) >= 2:
                sentences.append(s)
        n_docs += 1
        n_tokens += sum(len(s) for s in sents)
        if cap_articles and n_docs >= cap_articles:
            break
        if cap_tokens and n_tokens >= cap_tokens:
            break
    return docs, sentences


def train_idf_from_docs(
    docs: List[str],
    output_path: str,
    max_features: Optional[int] = None,
    min_df: int = 1,
    max_df: float = 1.0,
) -> Path:
    """Fit TfidfVectorizer on list of document strings and save pickle."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    vectorizer.fit(docs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved IDF to %s", out)
    return out


def train_fasttext_from_sentences(
    sentences: List[List[str]],
    output_dir: str,
    vector_size: int = DEFAULT_DIM,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    workers: int = 1,
    save_binary_path: Optional[str] = None,
) -> Path:
    """Train FastText on list of tokenized sentences; export to WISSE indexed (+ optional binary)."""
    from gensim.models import FastText

    logger.info("Training FastText on %d sentences (dim=%d, window=%d, min_count=%d, epochs=%d)",
                len(sentences), vector_size, window, min_count, epochs)
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        epochs=epochs,
        negative=5,
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    from .wisse import keyed2indexed
    keyed2indexed(model.wv, str(out_path), parallel=True)
    logger.info("Saved indexed embeddings to %s", out_path)

    if save_binary_path:
        model.wv.save_word2vec_format(save_binary_path, binary=True)
        logger.info("Saved binary embeddings to %s", save_binary_path)

    return out_path


def run_train(
    corpus_dir: Optional[str] = None,
    wikipedia_lang: Optional[str] = None,
    document_unit: str = "article",
    idf_out: str = "idf-en.pkl",
    embeddings_out: str = "fasttext-300-indexed",
    binary_out: Optional[str] = None,
    dim: int = DEFAULT_DIM,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    workers: int = 1,
    cap_articles: Optional[int] = None,
    cap_tokens: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Main entry: load corpus (dir or Wikipedia), train IDF and FastText, write outputs.
    Returns (idf_path, embeddings_dir_path).
    """
    if (corpus_dir is None) == (wikipedia_lang is None):
        raise ValueError("Set exactly one of corpus_dir or wikipedia_lang")

    if corpus_dir is not None:
        it = iter_corpus_from_dir(corpus_dir, document_unit=document_unit)
        docs, sentences = collect_corpus(it, cap_articles=cap_articles, cap_tokens=cap_tokens)
    else:
        it = iter_corpus_from_wikipedia(
            language=wikipedia_lang,
            document_unit=document_unit,
            cap_articles=cap_articles,
            cap_tokens=cap_tokens,
            seed=seed,
        )
        # Cap already applied inside iterator
        docs, sentences = collect_corpus(it)

    if not docs or not sentences:
        raise RuntimeError("Corpus produced no documents or sentences")

    logger.info("Corpus: %d documents, %d sentences", len(docs), len(sentences))
    train_idf_from_docs(docs, idf_out)
    emb_path = train_fasttext_from_sentences(
        sentences,
        embeddings_out,
        vector_size=dim,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=workers,
        save_binary_path=binary_out,
    )
    idf_path = Path(idf_out)
    return idf_path, emb_path
