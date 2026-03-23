# -*- coding: utf-8 -*-
"""
Sklearn TfidfVectorizer pickles from the paper era (typically **sklearn < 0.18**)
often fail under **sklearn 1.x**: the inner ``TfidfTransformer`` unpickles as
*not fitted*, so ``.transform()`` raises ``NotFittedError`` and the ``idf_``
property breaks.

This module resolves that **once** when loading: probe ``transform()``, and if it
fails, recover a per-feature ``idf_`` vector from ``__dict__`` / ``df_`` / ``n_samples_``
(or fall back to ones). ``SentenceEmbedding`` passes that into ``wisse``, which applies
the same TF×IDF weighting as sklearn (without calling the broken ``transform()``).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Shown in sklearn InconsistentVersionWarning when unpickling very old pickles
LEGACY_TFIDF_SKLEARN_HINT = (
    "Paper-era TF-IDF pickles were often created with sklearn **pre-0.18**. "
    "Under sklearn 1.x, use ``prepare_tfidf_vectorizer_for_inference`` (called "
    "from SentenceEmbedding) to recover IDF weights without calling ``transform()``."
)


def _n_features_from_vocabulary(vocab: dict) -> int:
    return int(max(vocab.values())) + 1 if vocab else 0


def _smooth_idf_from_df(df: Any, n_samples: Any) -> Optional[np.ndarray]:
    if df is None or n_samples is None:
        return None
    df_a = np.asarray(df, dtype=np.float64).ravel()
    n = float(n_samples)
    return np.log((1.0 + n) / (1.0 + df_a)) + 1.0


def _idf_from_legacy_tfidf_idf_diag(inner: Any) -> Optional[np.ndarray]:
    """
    Sklearn **<= 0.18** ``TfidfTransformer`` stored IDF as a sparse ``_idf_diag``
    matrix; ``idf_`` was a **property** (not always in ``__dict__``). Unpickling
    under sklearn 1.x leaves ``_idf_diag`` but ``transform()`` / ``idf_`` break.
    """
    if inner is None:
        return None
    diag = getattr(inner, "_idf_diag", None)
    if diag is None and hasattr(inner, "__dict__"):
        diag = inner.__dict__.get("_idf_diag")
    if diag is None:
        return None
    try:
        import scipy.sparse as sp

        if sp.issparse(diag):
            # Same as old property: np.ravel(self._idf_diag.sum(axis=0))
            try:
                flat = np.asarray(diag.diagonal(), dtype=np.float64).ravel()
                if flat.size > 0:
                    return flat
            except Exception:
                pass
            return np.ravel(np.asarray(diag.sum(axis=0), dtype=np.float64))
    except Exception:
        pass
    try:
        return np.ravel(np.asarray(diag.sum(axis=0), dtype=np.float64))
    except Exception:
        return None


def extract_idf_feature_array(vectorizer: Any) -> Optional[np.ndarray]:
    """
    Best-effort recovery of shape ``(n_features,)`` IDF weights without using the
    ``idf_`` property (which may delegate to an unfitted inner transformer).
    """
    if vectorizer is None:
        return None

    inner = getattr(vectorizer, "_tfidf", None)

    if inner is not None:
        legacy = _idf_from_legacy_tfidf_idf_diag(inner)
        if legacy is not None and legacy.size > 0:
            return legacy

    if inner is not None:
        raw = inner.__dict__.get("idf_")
        if raw is not None:
            return np.asarray(raw, dtype=np.float64).ravel()

    raw = vectorizer.__dict__.get("idf_")
    if raw is not None:
        return np.asarray(raw, dtype=np.float64).ravel()

    if inner is not None:
        df = getattr(inner, "df_", None) or inner.__dict__.get("df_")
        n_samples = (
            getattr(inner, "n_samples_", None)
            or inner.__dict__.get("n_samples_")
            or inner.__dict__.get("_n_samples")
        )
        out = _smooth_idf_from_df(df, n_samples)
        if out is not None:
            return out

    df = getattr(vectorizer, "df_", None) or vectorizer.__dict__.get("df_")
    n_samples = getattr(vectorizer, "n_samples_", None) or vectorizer.__dict__.get(
        "n_samples_"
    )
    out = _smooth_idf_from_df(df, n_samples)
    if out is not None:
        return out

    return None


def align_idf_to_vocab(idf: Optional[np.ndarray], n_features: int) -> np.ndarray:
    """Ensure length ``n_features``; pad or truncate; default 1.0 if unknown."""
    if n_features <= 0:
        return np.zeros(0, dtype=np.float64)
    if idf is None or idf.size == 0:
        logger.warning(
            "Could not recover IDF from legacy pickle; using weight 1.0 for all %d features.",
            n_features,
        )
        return np.ones(n_features, dtype=np.float64)
    a = np.asarray(idf, dtype=np.float64).ravel()
    if a.size == n_features:
        return a.copy()
    if a.size > n_features:
        return a[:n_features].copy()
    out = np.ones(n_features, dtype=np.float64)
    out[: a.size] = a
    logger.warning(
        "Recovered IDF length %d < vocabulary size %d; padded trailing features with 1.0.",
        a.size,
        n_features,
    )
    return out


def tfidf_transform_is_working(vectorizer: Any) -> bool:
    """True if ``transform`` runs (modern sklearn + fitted / compatible pickle)."""
    if vectorizer is None:
        return False
    try:
        vectorizer.transform(["the"])
        return True
    except Exception:
        return False


def prepare_tfidf_vectorizer_for_inference(
    vectorizer: Any,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Decide how WISSE should obtain term weights.

    Returns
    -------
    use_transform, idf_per_feature
        If ``use_transform`` is True, pass ``idf_per_feature=None`` to ``wisse`` and
        use full ``TfidfVectorizer.transform`` (TF×IDF).
        If False, pass ``idf_per_feature`` (shape ``(n_features,)``): ``wisse`` rebuilds
        the same TF×IDF pipeline (counts / binary / sublinear × ``idf_`` / norm) without
        calling ``.transform()`` on the broken inner transformer.
    """
    if vectorizer is None:
        return True, None

    vocab = getattr(vectorizer, "vocabulary_", None)
    if not vocab:
        logger.warning("TfidfVectorizer has no vocabulary_; cannot recover IDF.")
        return True, None

    n_features = _n_features_from_vocabulary(vocab)
    if n_features <= 0:
        return True, None

    if tfidf_transform_is_working(vectorizer):
        return True, None

    logger.info(
        "TF-IDF artifact: .transform() failed under this sklearn (typical for pickles "
        "from sklearn pre-0.18). Recovering per-feature IDF for manual TF×IDF (incl. "
        "legacy _idf_diag if present)."
    )

    raw = extract_idf_feature_array(vectorizer)
    aligned = align_idf_to_vocab(raw, n_features)
    return False, aligned
