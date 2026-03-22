# -*- coding: utf-8 -*-
"""
SBERT-like facade: SentenceEmbedding with encode() and similarity().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .download import (
    DEFAULT_EMBEDDING_KEY,
    DEFAULT_IDF_KEY,
    get_embedding_path,
    get_idf_path,
    load_idf,
)
from .tfidf_compat import prepare_tfidf_vectorizer_for_inference
from .wisse import vector_space, wisse
from . import similarity as sim_module

logger = logging.getLogger(__name__)


class SentenceEmbedding:
    """
    SBERT-compatible interface for WISSE sentence embeddings.
    Use encode() for batch encoding and similarity() for pairwise scores.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        idf_name_or_path: Optional[str] = None,
        combiner: str = "sum",
        use_tfidf_weights: bool = True,
    ):
        """
        Load a WISSE model from a registry name or local path.

        Parameters
        ----------
        model_name_or_path : str or None
            Registry key (e.g. "wisse-glove-300") or path to indexed embedding
            directory / .tar.gz. If None, uses default embedding.
        idf_name_or_path : str or None
            Registry key (e.g. "wisse-idf-en") or path to pickled TfidfVectorizer.
            If None, uses default TF-IDF weights when use_tfidf_weights is True.
        combiner : "sum" or "avg"
            How to combine word vectors into sentence vector.
        use_tfidf_weights : bool
            If True, apply full TF-IDF weighting (best performing); if False, use uniform weights.
        """
        if model_name_or_path is None:
            model_name_or_path = DEFAULT_EMBEDDING_KEY

        try:
            emb_path = get_embedding_path(model_name_or_path)
        except FileNotFoundError as e:
            logger.warning(
                "Default embedding not found or download failed. "
                "Use a local path or set WISSE_HOME. %s",
                e,
            )
            raise

        self._embedding = vector_space(str(emb_path), sparse=False)
        self._combiner = combiner
        self._use_tfidf = use_tfidf_weights
        self._vectorizer = None
        self._tf_tfidf = False

        if use_tfidf_weights:
            if idf_name_or_path is None:
                idf_name_or_path = DEFAULT_IDF_KEY
            try:
                idf_path = get_idf_path(idf_name_or_path)
                self._vectorizer = load_idf(idf_path)
                self._tf_tfidf = True
                _, self._idf_per_feature = prepare_tfidf_vectorizer_for_inference(
                    self._vectorizer
                )
            except FileNotFoundError as e:
                logger.warning(
                    "TF-IDF artifact not found or download failed; using uniform weights. %s",
                    e,
                )
                self._use_tfidf = False
                self._idf_per_feature = None
        else:
            self._idf_per_feature = None

        if not self._use_tfidf:
            self._idf_per_feature = None

        self._wisse = wisse(
            self._embedding,
            vectorizer=self._vectorizer,
            tf_tfidf=self._tf_tfidf,
            combiner=combiner,
            return_missing=False,
            generate=True,
            idf_per_feature=self._idf_per_feature,
        )

        # One .npy load (or tar member), not a full vocabulary scan
        try:
            self._dim = int(self._embedding.get_embedding_dimension())
        except Exception:
            self._dim = 0

    @property
    def dimension(self) -> int:
        """Sentence embedding dimension."""
        return self._dim

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        Encode sentences into embedding vectors (SBERT-like API).

        Parameters
        ----------
        sentences : str or list of str
            One or more sentences.
        batch_size : int
            Batch size for processing (for progress only; WISSE is CPU-bound).
        show_progress_bar : bool
            If True, log progress (no tqdm by default).
        normalize_embeddings : bool
            L2-normalize embeddings to unit length.
        convert_to_numpy : bool
            Return numpy array (always True for WISSE).

        Returns
        -------
        np.ndarray
            Shape (n_sentences, dim) of dtype float32.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        out = []
        for i, sent in enumerate(sentences):
            if show_progress_bar and (i + 1) % max(1, batch_size) == 0:
                logger.info("Encoded %d / %d", i + 1, len(sentences))
            vec = self._wisse.infer_sentence(sent)
            if vec is None:
                # Fallback zero vector if no words in vocab
                vec = np.zeros(self._dim, dtype=np.float32)
            elif isinstance(vec, tuple):
                vec = vec[2]
            vec = np.asarray(vec, dtype=np.float32)
            if normalize_embeddings and vec.size:
                n = np.linalg.norm(vec)
                if n > 1e-12:
                    vec = vec / n
            out.append(vec)

        if not out:
            return np.zeros((0, self._dim), dtype=np.float32)

        emb = np.vstack(out)
        return emb

    def similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: Optional[np.ndarray] = None,
        similarity_fn: str = "cosine",
    ) -> np.ndarray:
        """
        Compute pairwise similarity between two sets of embeddings (SBERT-like).

        Parameters
        ----------
        embeddings_a : np.ndarray
            Shape (n_a, dim).
        embeddings_b : np.ndarray or None
            Shape (n_b, dim). If None, uses embeddings_a (self-similarity).
        similarity_fn : "cosine" | "dot" | "euclidean" | "manhattan"
            Similarity or distance function.

        Returns
        -------
        np.ndarray
            If embeddings_b is None: (n_a, n_a); else (n_a, n_b).
            For cosine/dot: higher is more similar; for euclidean/manhattan: lower is closer.
        """
        return sim_module.similarity(embeddings_a, embeddings_b, similarity_fn)


def similarity(
    embeddings_a: np.ndarray,
    embeddings_b: Optional[np.ndarray] = None,
    similarity_fn: str = "cosine",
) -> np.ndarray:
    """
    Standalone pairwise similarity (SBERT-like).
    """
    return sim_module.similarity(embeddings_a, embeddings_b, similarity_fn)
