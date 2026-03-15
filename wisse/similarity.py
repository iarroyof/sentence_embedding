# -*- coding: utf-8 -*-
"""
Pairwise similarity/distance for sentence embeddings (SBERT-compatible).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def similarity(
    embeddings_a: np.ndarray,
    embeddings_b: Optional[np.ndarray] = None,
    similarity_fn: str = "cosine",
) -> np.ndarray:
    """
    Pairwise similarity or distance between embedding matrices.

    Parameters
    ----------
    embeddings_a : np.ndarray
        Shape (n_a, dim).
    embeddings_b : np.ndarray or None
        Shape (n_b, dim). If None, use embeddings_a (self-similarity).
    similarity_fn : str
        "cosine" | "dot" | "euclidean" | "manhattan"

    Returns
    -------
    np.ndarray
        (n_a, n_b) or (n_a, n_a). Cosine/dot: higher = more similar;
        euclidean/manhattan: lower = closer.
    """
    if embeddings_b is None:
        embeddings_b = embeddings_a

    a = np.asarray(embeddings_a, dtype=np.float64)
    b = np.asarray(embeddings_b, dtype=np.float64)

    if similarity_fn == "cosine":
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        na = np.where(na < 1e-12, 1.0, na)
        nb = np.where(nb < 1e-12, 1.0, nb)
        return (a / na) @ (b / nb).T
    if similarity_fn == "dot":
        return a @ b.T
    if similarity_fn == "euclidean":
        # Return negative distance so higher = closer (SBERT convention)
        from sklearn.metrics.pairwise import euclidean_distances
        return -euclidean_distances(a, b)
    if similarity_fn == "manhattan":
        from sklearn.metrics.pairwise import manhattan_distances
        return -manhattan_distances(a, b)

    raise ValueError(
        f"similarity_fn must be one of cosine, dot, euclidean, manhattan; got {similarity_fn!r}"
    )
