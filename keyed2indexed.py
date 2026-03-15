#!/usr/bin/env python3
"""
Convert word2vec KeyedVectors to WISSE indexed format.
Use: python keyed2indexed.py --input model.bin --output output_indexed
     python keyed2indexed.py --input model.vec --txt --output output_indexed
"""
import argparse
import logging
import sys

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)

parser = argparse.ArgumentParser(
    description="Convert word2vec embeddings to WISSE indexed (.npy per word) format."
)
parser.add_argument("--input", "-i", required=True, help="Input embeddings (word2vec .bin or .vec)")
parser.add_argument("--output", "-o", default="output_indexed", help="Output directory")
parser.add_argument(
    "--txt",
    action="store_true",
    help="Input is text .vec format (default: binary .bin)",
)
args = parser.parse_args()

import wisse
from gensim.models.keyedvectors import KeyedVectors

binary = not args.txt
try:
    embedding = KeyedVectors.load_word2vec_format(
        args.input, binary=binary, encoding="utf-8"
    )
except Exception:
    embedding = KeyedVectors.load_word2vec_format(
        args.input, binary=binary, encoding="latin-1"
    )

logging.info("Indexing embeddings, this may take a while...")
wisse.keyed2indexed(embedding, args.output)
logging.info("Embeddings indexed: %s", args.output)
