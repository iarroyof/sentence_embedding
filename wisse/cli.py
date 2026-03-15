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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "keyed2indexed":
        sys.argv.pop(1)
        main_keyed2indexed()
    else:
        main_encode()
