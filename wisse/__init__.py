# -*- coding: utf-8 -*-
"""
WISSE: sentence embeddings with TF-IDF-weighted word vectors.
SBERT-like API via SentenceEmbedding for downstream NLP.
"""
from .wisse import (
    wisse,
    vector_space,
    keyed2indexed,
    streamer,
    load_dense,
    save_dense,
)
from .model import SentenceEmbedding, similarity
from . import similarity as similarity_module

__all__ = [
    "wisse",
    "vector_space",
    "keyed2indexed",
    "streamer",
    "load_dense",
    "save_dense",
    "SentenceEmbedding",
    "similarity",
    "similarity_module",
]

__version__ = "0.1.0"
