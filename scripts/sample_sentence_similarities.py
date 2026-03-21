#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample N lines from a WISSE training sentence corpus (one sentence per line,
space-separated tokens), encode with SentenceEmbedding (TF-IDF weights), and
write pairwise cosine similarities — highlighting most similar and most dissimilar pairs.

WISSE is a *bag-of-words* FastText + TF-IDF model (not a transformer). High cosine
between two random Wikipedia lines often reflects shared common tokens / templates,
not “same topic” in the SBERT sense. Use --max-tokens / --filter-wiki-boilerplate
for more interpretable demos.

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
from typing import List, Optional, Tuple

# Run from repo root without pip install: python scripts/sample_sentence_similarities.py ...
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _looks_like_wiki_boilerplate(s: str) -> bool:
    """
    Heuristic: very long lines that mix many Wikipedia navbox/footer phrases
    dominate token overlap with unrelated sentences — skip for clearer demos.
    """
    low = s.lower()
    tail_markers = (
        "external links",
        "references further reading",
        "living people",
        "year of birth missing",
        "place of birth missing",
    )
    if sum(1 for m in tail_markers if m in low) >= 2:
        return True
    if "track listing" in low and ("vinyl" in low or "demo" in low or "uk cd" in low):
        return True
    return False


def reservoir_sample_lines(
    path: str,
    k: int,
    seed: int,
    encoding: str = "utf-8",
    min_tokens: Optional[int] = 10,
    max_tokens: Optional[int] = 100,
    filter_wiki_boilerplate: bool = True,
) -> Tuple[List[str], int]:
    """
    Uniform random sample of k eligible non-empty lines from a large text file.

    Returns (reservoir, n_eligible_seen).
    """
    rng = random.Random(seed)
    reservoir: List[str] = []
    seen_eligible = 0
    with open(path, encoding=encoding, errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            toks = s.split()
            if min_tokens is not None and len(toks) < min_tokens:
                continue
            if max_tokens is not None and len(toks) > max_tokens:
                continue
            if filter_wiki_boilerplate and _looks_like_wiki_boilerplate(s):
                continue
            seen_eligible += 1
            if len(reservoir) < k:
                reservoir.append(s)
            else:
                j = rng.randrange(seen_eligible)
                if j < k:
                    reservoir[j] = s
    if len(reservoir) < k:
        raise ValueError(
            f"Only {len(reservoir)} lines matched filters after scanning {path!r}; "
            f"need {k}. Try --no-length-filter and/or --keep-boilerplate, or a larger corpus."
        )
    return reservoir, seen_eligible


def pairwise_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def scores_from_embeddings(sim_matrix) -> List[Tuple[float, int, int]]:
    n = sim_matrix.shape[0]
    pairs = pairwise_pairs(n)
    scored: List[Tuple[float, int, int]] = []
    for i, j in pairs:
        scored.append((float(sim_matrix[i, j]), i, j))
    scored.sort(key=lambda x: -x[0])
    return scored


def write_report(
    out_path: Path,
    *,
    sentences: List[str],
    corpus: Path,
    eligible: int,
    seed: int,
    n: int,
    min_tok: Optional[int],
    max_tok: Optional[int],
    filter_bp: bool,
    use_tfidf: bool,
    combiner: str,
    model_repr: str,
    idf_repr: str,
    scored: List[Tuple[float, int, int]],
    top_similar: int,
    top_dissimilar: int,
    all_pairs: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# WISSE sentence pair similarities (cosine)\n")
        out.write(f"# generated_utc: {ts}\n")
        out.write(f"# corpus: {corpus.resolve()}\n")
        out.write(f"# n_sampled: {n}  seed: {seed}\n")
        out.write(f"# eligible_lines_matching_filters: {eligible}\n")
        out.write(
            f"# length_filter: min_tokens={min_tok} max_tokens={max_tok} "
            f"wiki_boilerplate_filter={filter_bp}\n"
        )
        out.write(f"# tfidf: {use_tfidf}  combiner: {combiner}\n")
        out.write(f"# model: {model_repr!r}  idf: {idf_repr!r}\n\n")
        out.write(REPORT_CAVEATS)
        out.write("\n")

        out.write("## Sampled sentences (index \\t line)\n")
        for idx, s in enumerate(sentences):
            out.write(f"{idx}\t{s}\n")

        out.write("\n## Most similar pairs (highest cosine)\n")
        out.write("# rank\tcosine\tidx_i\tidx_j\tsentence_i\tsentence_j\n")
        for rank, (cos, i, j) in enumerate(scored[:top_similar], start=1):
            out.write(f"{rank}\t{cos:.6f}\t{i}\t{j}\t{sentences[i]}\t{sentences[j]}\n")

        out.write("\n## Most dissimilar pairs (lowest cosine)\n")
        out.write("# rank\tcosine\tidx_i\tidx_j\tsentence_i\tsentence_j\n")
        dis = list(reversed(scored[-top_dissimilar:]))
        for rank, (cos, i, j) in enumerate(dis, start=1):
            out.write(f"{rank}\t{cos:.6f}\t{i}\t{j}\t{sentences[i]}\t{sentences[j]}\n")

        if all_pairs:
            out.write("\n## All pairs (cosine descending)\n")
            out.write("# cosine\tidx_i\tidx_j\tsentence_i\tsentence_j\n")
            for cos, i, j in scored:
                out.write(f"{cos:.6f}\t{i}\t{j}\t{sentences[i]}\t{sentences[j]}\n")


REPORT_CAVEATS = """## How to read this (WISSE vs SBERT)

WISSE builds one vector per sentence as a *weighted sum/average of word vectors*
(FastText) with TF-IDF weights. Cosine similarity reflects *overlap of weighted
word directions*, not full-sentence semantics like SBERT / sentence-transformers.

Random pairs of Wikipedia lines often show *moderately high* cosine because they
share many function words and recurring encyclopedic vocabulary. *Very long*
lines (navboxes, filmography footers, track listings) share even more tokens with
unrelated sentences — the sampling defaults try to exclude those.

For topic-coherent pairs, prefer *curated* sentences or smaller random samples
with strict `--max-tokens` and boilerplate filtering enabled.
"""


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
        "--min-tokens",
        type=int,
        default=10,
        help="Skip shorter lines (default: 10). Use with --no-length-filter to disable.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Skip longer lines (default: 100). Use with --no-length-filter to disable.",
    )
    parser.add_argument(
        "--no-length-filter",
        action="store_true",
        help="Do not filter by token count (any length line may be sampled).",
    )
    parser.add_argument(
        "--keep-boilerplate",
        action="store_true",
        help="Do not skip long Wikipedia footer / track-list style lines.",
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
        help="WISSE combiner (default: sum). Try avg for length-robust demos.",
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

    min_tok: Optional[int] = None if args.no_length_filter else args.min_tokens
    max_tok: Optional[int] = None if args.no_length_filter else args.max_tokens
    filter_bp = not args.keep_boilerplate

    # Import after argparse so --help works without wisse installed in some edge cases
    from wisse import SentenceEmbedding

    print(
        f"Sampling {args.n} lines from {corpus} (seed={args.seed}, "
        f"min_tok={min_tok}, max_tok={max_tok}, filter_boilerplate={filter_bp})...",
        flush=True,
    )
    try:
        sentences, eligible = reservoir_sample_lines(
            str(corpus),
            args.n,
            args.seed,
            min_tokens=min_tok,
            max_tokens=max_tok,
            filter_wiki_boilerplate=filter_bp,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Reservoir filled from {eligible} eligible lines matching filters.", flush=True)
    print(
        "Loading model and encoding (TF-IDF=%s, combiner=%s)..."
        % (not args.no_tfidf, args.combiner),
        flush=True,
    )
    model = SentenceEmbedding(
        model_name_or_path=args.model,
        idf_name_or_path=args.idf_path,
        combiner=args.combiner,
        use_tfidf_weights=not args.no_tfidf,
    )
    emb = model.encode(sentences, show_progress_bar=True, normalize_embeddings=False)
    sim = model.similarity(emb, similarity_fn="cosine")
    scored = scores_from_embeddings(sim)

    out_path = Path(args.output)
    write_report(
        out_path,
        sentences=sentences,
        corpus=corpus,
        eligible=eligible,
        seed=args.seed,
        n=args.n,
        min_tok=min_tok,
        max_tok=max_tok,
        filter_bp=filter_bp,
        use_tfidf=not args.no_tfidf,
        combiner=args.combiner,
        model_repr=str(args.model),
        idf_repr=str(args.idf_path),
        scored=scored,
        top_similar=args.top_similar,
        top_dissimilar=args.top_dissimilar,
        all_pairs=args.all_pairs,
    )
    print(f"Wrote report to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
