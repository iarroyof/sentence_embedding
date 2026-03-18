# -*- coding: utf-8 -*-
"""
CLI entry points for encode and keyed2indexed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main_keyed2indexed() -> None:
    parser = argparse.ArgumentParser(
        description="Convert word2vec KeyedVectors to WISSE indexed (.npy per word) format."
    )
    parser.add_argument("--input", "-i", required=True, help="Input embeddings (word2vec .bin or .vec)")
    parser.add_argument("--output", "-o", default="output_indexed", help="Output directory for .npy files")
    parser.add_argument(
        "--txt",
        action="store_true",
        help="Input is text .vec format (default: binary)",
    )
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel export")
    args = parser.parse_args()

    import logging
    from gensim.models.keyedvectors import KeyedVectors

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    binary = not args.txt
    try:
        embedding = KeyedVectors.load_word2vec_format(args.input, binary=binary, encoding="utf-8")
    except Exception as e:
        try:
            embedding = KeyedVectors.load_word2vec_format(args.input, binary=binary, encoding="latin-1")
        except Exception:
            print(f"Error loading embeddings: {e}", file=sys.stderr)
            sys.exit(1)

    from .wisse import keyed2indexed

    logging.info("Indexing embeddings...")
    keyed2indexed(embedding, args.output, parallel=not args.no_parallel)
    logging.info("Done: %s", args.output)


def main_encode() -> None:
    parser = argparse.ArgumentParser(
        description="Encode sentences to vectors using WISSE (SBERT-like)."
    )
    parser.add_argument("--input", "-i", required=True, help="Input file: one sentence per line")
    parser.add_argument("--output", "-o", required=True, help="Output .npy or .txt file for vectors")
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name (e.g. wisse-glove-300) or path to indexed embedding dir",
    )
    parser.add_argument("--idf", default=None, help="IDF name or path to .pkl (optional)")
    parser.add_argument("--comb", choices=("sum", "avg"), default="sum", help="Combiner: sum or avg")
    args = parser.parse_args()

    from .model import SentenceEmbedding
    import numpy as np

    model = SentenceEmbedding(
        model_name_or_path=args.model,
        idf_name_or_path=args.idf,
        combiner=args.comb,
        use_tfidf_weights=True,
    )
    sentences = []
    with open(args.input, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    embeddings = model.encode(sentences, show_progress_bar=True)
    out = Path(args.output)
    if out.suffix.lower() == ".npy":
        np.save(out, embeddings)
    else:
        with open(out, "w", encoding="utf-8") as f:
            for i, vec in enumerate(embeddings):
                f.write(f"{i}\t" + " ".join(f"{x:.6f}" for x in vec) + "\n")
    print(f"Saved {len(embeddings)} vectors to {args.output}")


def main_train() -> None:
    parser = argparse.ArgumentParser(
        description="Train IDF (TF-IDF) and FastText from a text directory or Wikipedia (HF), "
        "producing WISSE-ready IDF pickle and indexed embeddings."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus-dir", type=str, help="Directory of plain text files")
    group.add_argument(
        "--wikipedia",
        type=str,
        metavar="LANG",
        help="Wikipedia via Hugging Face (e.g. en, es). Requires: pip install datasets",
    )
    parser.add_argument(
        "--document-unit",
        choices=("article", "paragraph"),
        default="article",
        help="Document unit for IDF: one file/article per doc, or split by paragraph (default: article)",
    )
    parser.add_argument(
        "--idf-out",
        type=str,
        default=None,
        help="Output path for IDF pickle (default: idf-<lang>.pkl with --wikipedia, else idf-en.pkl)",
    )
    parser.add_argument(
        "--embeddings-out",
        type=str,
        default=None,
        help="Output directory for indexed FastText .npy files (default: fasttext-300-indexed)",
    )
    parser.add_argument(
        "--binary-out",
        type=str,
        default=None,
        help="Optional path to save FastText in Word2Vec binary format",
    )
    parser.add_argument("--dim", type=int, default=300, help="Embedding dimension (default: 300)")
    parser.add_argument("--window", type=int, default=5, help="FastText window (default: 5)")
    parser.add_argument("--min-count", type=int, default=5, help="FastText min_count (default: 5)")
    parser.add_argument("--epochs", type=int, default=5, help="FastText epochs (default: 5)")
    parser.add_argument("--workers", type=int, default=1, help="FastText workers (default: 1)")
    parser.add_argument(
        "--cap-articles",
        type=int,
        default=None,
        help="Max documents (efficient random sampling). Default for Wikipedia: 500000",
    )
    parser.add_argument(
        "--cap-tokens",
        type=int,
        default=None,
        help="Max tokens to use. Default for Wikipedia: 100000000",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Force one-pass streaming (sentences on disk; low RAM). Auto-on when --cap-tokens > 15M.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable auto streaming even for large --cap-tokens (may OOM).",
    )
    parser.add_argument(
        "--sentence-corpus",
        type=str,
        default=None,
        metavar="PATH",
        help="With streaming: keep sentence lines here instead of a temp file (reuse for reruns)",
    )
    parser.add_argument(
        "--idf-min-df",
        type=int,
        default=1,
        help="Streaming IDF: min document frequency (default: 1)",
    )
    parser.add_argument(
        "--idf-max-df",
        type=float,
        default=1.0,
        help="Streaming IDF: max document proportion 0–1 (default: 1.0)",
    )
    parser.add_argument(
        "--idf-max-features",
        type=int,
        default=None,
        help="Streaming IDF: cap vocabulary size by top df (optional)",
    )
    args = parser.parse_args()

    if args.streaming and args.no_streaming:
        print("Error: use only one of --streaming and --no-streaming", file=sys.stderr)
        sys.exit(1)

    from .train import (
        DEFAULT_CAP_ARTICLES,
        DEFAULT_CAP_TOKENS,
        run_train,
    )

    lang = (args.wikipedia or "en").strip().lower() if args.wikipedia else "en"
    idf_out = args.idf_out if args.idf_out is not None else f"idf-{lang}.pkl"
    embeddings_out = args.embeddings_out if args.embeddings_out is not None else "fasttext-300-indexed"

    cap_articles = args.cap_articles
    cap_tokens = args.cap_tokens
    if args.wikipedia and cap_articles is None and cap_tokens is None:
        cap_articles = DEFAULT_CAP_ARTICLES
        cap_tokens = DEFAULT_CAP_TOKENS

    import logging
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

    streaming = None
    if args.streaming:
        streaming = True
    elif args.no_streaming:
        streaming = False

    try:
        idf_path, emb_path = run_train(
            corpus_dir=args.corpus_dir,
            wikipedia_lang=args.wikipedia,
            document_unit=args.document_unit,
            idf_out=idf_out,
            embeddings_out=embeddings_out,
            binary_out=args.binary_out,
            dim=args.dim,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs,
            workers=args.workers,
            cap_articles=cap_articles,
            cap_tokens=cap_tokens,
            seed=args.seed,
            streaming=streaming,
            sentence_corpus_path=args.sentence_corpus,
            idf_min_df=args.idf_min_df,
            idf_max_df=args.idf_max_df,
            idf_max_features=args.idf_max_features,
        )
        print(f"IDF: {idf_path}")
        print(f"Embeddings: {emb_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "keyed2indexed":
        sys.argv.pop(1)
        main_keyed2indexed()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv.pop(1)
        main_train()
    else:
        main_encode()
