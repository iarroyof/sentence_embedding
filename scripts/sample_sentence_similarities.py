#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample N lines from a WISSE training sentence corpus (one sentence per line,
space-separated tokens), encode with SentenceEmbedding (TF-IDF weights), and
write pairwise cosine similarities — highlighting most similar and most dissimilar pairs.

Usage (after pip install -e .):
  python scripts/sample_sentence_similarities.py \\
    --sentence-corpus /mnt/wisse-training/corpus/wiki-en-sentences.txt \\
    --model /mnt/wisse-training/models/fasttext-300-indexed \\
    --idf /mnt/wisse-training/models/idf-en.pkl \\
    --output similarities.txt

Defaults for --model / --idf: same registry keys as wisse-encode (may download).
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

# Run from repo root without pip install: python scripts/sample_sentence_similarities.py ...
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def reservoir_sample_lines(
    path: str,
    k: int,
    seed: int,
    encoding: str = "utf-8",
) -> List[str]:
    """Uniform random sample of k non-empty lines from a large text file (one pass)."""
    rng = random.Random(seed)
    reservoir: List[str] = []
    seen = 0
    with open(path, encoding=encoding, errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(s)
            else:
                j = rng.randrange(seen)
                if j < k:
                    reservoir[j] = s
    if len(reservoir) < k:
        raise ValueError(
            f"File {path!r} has only {len(reservoir)} non-empty lines; need {k}"
        )
    return reservoir


def pairwise_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample sentences from corpus file, encode (TF-IDF WISSE), "
        "write similar/dissimilar pair similarities to a file."
    )
    parser.add_argument(
        "--sentence-corpus",
        required=True,
        help="Path to one-sentence-per-line file (e.g. wiki-en-sentences.txt)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Indexed embedding dir, registry key, or .tar.gz (default: package default)",
    )
    parser.add_argument(
        "--idf",
        dest="idf_path",
        default=None,
        help="Pickled TfidfVectorizer path or registry key (default: package default)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="sentence_pair_similarities.txt",
        help="Output report path (UTF-8 text)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of sentences to sample (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reservoir sampling (default: 42)",
    )
    parser.add_argument(
        "--top-similar",
        type=int,
        default=25,
        help="How many highest-similarity pairs to list (default: 25)",
    )
    parser.add_argument(
        "--top-dissimilar",
        type=int,
        default=25,
        help="How many lowest-similarity pairs to list (default: 25)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Also append full sorted list of all pairs (can be long)",
    )
    parser.add_argument(
        "--combiner",
        choices=("sum", "avg"),
        default="sum",
        help="WISSE combiner (default: sum)",
    )
    parser.add_argument(
        "--no-tfidf",
        action="store_true",
        help="Disable TF-IDF weights (uniform word weights)",
    )
    args = parser.parse_args()

    corpus = Path(args.sentence_corpus)
    if not corpus.is_file():
        print(f"Error: sentence corpus not found: {corpus}", file=sys.stderr)
        return 1

    if args.n < 2:
        print("Error: --n must be at least 2 for pairs.", file=sys.stderr)
        return 1

    # Import after argparse so --help works without wisse installed in some edge cases
    from wisse import SentenceEmbedding

    print(f"Sampling {args.n} lines from {corpus} (seed={args.seed})...")
    sentences = reservoir_sample_lines(str(corpus), args.n, args.seed)
    print("Loading model and encoding (TF-IDF=%s, combiner=%s)..." % (not args.no_tfidf, args.combiner))
    model = SentenceEmbedding(
        model_name_or_path=args.model,
        idf_name_or_path=args.idf_path,
        combiner=args.combiner,
        use_tfidf_weights=not args.no_tfidf,
    )
    emb = model.encode(sentences, show_progress_bar=True, normalize_embeddings=False)
    sim = model.similarity(emb, similarity_fn="cosine")

    pairs = pairwise_pairs(len(sentences))
    scored: List[Tuple[float, int, int]] = []
    for i, j in pairs:
        scored.append((float(sim[i, j]), i, j))
    scored.sort(key=lambda x: -x[0])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"# WISSE sentence pair similarities (cosine)\n")
        out.write(f"# generated_utc: {ts}\n")
        out.write(f"# corpus: {corpus.resolve()}\n")
        out.write(f"# n_sampled: {args.n}  seed: {args.seed}\n")
        out.write(f"# tfidf: {not args.no_tfidf}  combiner: {args.combiner}\n")
        out.write(f"# model: {args.model!r}  idf: {args.idf_path!r}\n\n")

        out.write("## Sampled sentences (index \\t line)\n")
        for idx, s in enumerate(sentences):
            out.write(f"{idx}\t{s}\n")

        out.write("\n## Most similar pairs (highest cosine)\n")
        out.write("# rank\tcosine\tidx_i\tidx_j\tsentence_i\tsentence_j\n")
        for rank, (cos, i, j) in enumerate(scored[: args.top_similar], start=1):
            out.write(
                f"{rank}\t{cos:.6f}\t{i}\t{j}\t{sentences[i]}\t{sentences[j]}\n"
            )

        out.write("\n## Most dissimilar pairs (lowest cosine)\n")
        out.write("# rank\tcosine\tidx_i\tidx_j\tsentence_i\tsentence_j\n")
        dis = list(reversed(scored[-args.top_dissimilar :]))
        for rank, (cos, i, j) in enumerate(dis, start=1):
            out.write(
                f"{rank}\t{cos:.6f}\t{i}\t{j}\t{sentences[i]}\t{sentences[j]}\n"
            )

        if args.all_pairs:
            out.write("\n## All pairs (cosine descending)\n")
            out.write("# cosine\tidx_i\tidx_j\tsentence_i\tsentence_j\n")
            for cos, i, j in scored:
                out.write(f"{cos:.6f}\t{i}\t{j}\t{sentences[i]}\t{sentences[j]}\n")

    print(f"Wrote report to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
