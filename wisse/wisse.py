# -*- coding: utf-8 -*-
"""
WISSE: sentence embeddings via TF-IDF-weighted combination of word embeddings.
Python 3 only.
"""
from __future__ import annotations

import logging
import os
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


class wisse:
    """
    TF-IDF-weighted sentence embeddings from a word embedding space and optional
    TF-IDF vectorizer. Both can be pretrained or fitted on a corpus.
    """

    def __init__(
        self,
        embeddings: Any,
        vectorizer: Optional[TfidfVectorizer] = None,
        tf_tfidf: Optional[bool] = None,
        combiner: str = "sum",
        verbose: bool = False,
        return_missing: bool = False,
        generate: bool = False,
        idf_per_feature: Optional[np.ndarray] = None,
    ):
        if vectorizer is not None:
            self.tokenize = vectorizer.build_tokenizer()
        else:
            self.tokenize = TfidfVectorizer().build_tokenizer()

        self.tfidf = vectorizer
        self.embedding = embeddings
        self.tf_tfidf = tf_tfidf if vectorizer is not None else False
        # Pre-aligned IDF from tfidf_compat when sklearn .transform() is unusable (legacy pickles)
        self._idf_per_feature = idf_per_feature
        self.rm = return_missing
        self.generate = generate
        self.verbose = verbose

        if combiner.startswith("avg"):
            self.comb = partial(np.mean, axis=0)
        else:
            self.comb = partial(np.sum, axis=0)

    def fit(self, X: Union[List[str], Tuple[str, ...]], y: Any = None) -> "wisse":
        if isinstance(X, (list, tuple)):
            self.sentences = X
            if not self.generate and not self.rm:
                S = [self._embed_one(s) for s in self.sentences]
                nulls = [
                    i
                    for i, v in enumerate(S)
                    if (v is None or (isinstance(v, np.ndarray) and v.size == 0))
                ]
                if nulls:
                    a_idx = next(
                        (i for i, v in enumerate(S) if v is not None and getattr(v, "shape", None)),
                        None,
                    )
                    if a_idx is not None:
                        dim = np.asarray(S[a_idx]).shape[0]
                        for n in nulls:
                            S[n] = np.zeros(dim, dtype=np.float64)
                return np.vstack([np.asarray(s, dtype=np.float64) for s in S])
        return self

    def transform(self, X: Union[str, List[str], Tuple[str, ...]]) -> Any:
        if isinstance(X, (list, tuple)):
            return self.fit(X)
        if isinstance(X, str):
            return self.infer_sentence(X)
        return None

    def fit_transform(
        self, X: Union[List[str], Tuple[str, ...]], y: Any = None
    ) -> Any:
        return self.transform(X)

    def _embed_one(self, sent: str) -> Optional[np.ndarray]:
        out = self.infer_sentence(sent)
        if out is None:
            return None
        if self.rm:
            return out[2]
        return out

    def infer_sentence(
        self, sent: str
    ) -> Union[Optional[np.ndarray], Tuple[List[str], List[str], np.ndarray]]:
        try:
            if self.tfidf is not None and getattr(self.tfidf, "lowercase", True):
                sent = sent.lower()
        except Exception:
            sent = sent.lower()

        ss = self.tokenize(sent)
        self.missing_bow: List[str] = []
        self.missing_cbow: List[str] = []
        series: dict = {}

        if not ss:
            return None

        if self.tf_tfidf is False and self.tfidf is None:
            self.weights, m = dict(zip(ss, [1.0] * len(ss))), []
        else:
            self.weights, m = self._infer_tfidf_weights(ss)

        self.missing_bow += m

        for w in self.weights:
            try:
                vec = self.embedding[w]
                if hasattr(vec, "toarray"):
                    vec = vec.toarray().ravel()
                series[w] = (self.weights[w], np.asarray(vec, dtype=np.float64))
            except (KeyError, IndexError):
                self.missing_cbow.append(w)
                continue

        if not series:
            return None

        weighted = np.array(
            [weight * vec for weight, vec in series.values()],
            dtype=np.float64,
        )
        sentence_vec = self.comb(weighted)

        if self.verbose:
            logging.info("Sentence weights: %s", self.weights)
        if self.rm:
            return self.missing_cbow, self.missing_bow, sentence_vec
        return sentence_vec

    def _raw_idf_array(self) -> Optional[np.ndarray]:
        from .tfidf_compat import extract_idf_feature_array

        return extract_idf_feature_array(self.tfidf)

    def _infer_idf_table_weights(
        self, sentence: List[str], idf_arr: np.ndarray
    ) -> Tuple[dict, List[str]]:
        """IDF-only weights from a pre-aligned (n_features,) array and vocabulary_."""
        existent: dict = {}
        missing: List[str] = []
        vocab = self.tfidf.vocabulary_
        for word in sentence:
            try:
                idx = int(vocab[word])
                if idx < 0 or idx >= len(idf_arr):
                    missing.append(word)
                    continue
                w = float(idf_arr[idx])
                existent[word] = w if w > 2.0 else 0.01
            except KeyError:
                missing.append(word)
        return existent, missing

    def _infer_idf_only_weights(self, sentence: List[str]) -> Tuple[dict, List[str]]:
        """Recover IDF table without .transform() (e.g. ad-hoc wisse use, legacy pickle)."""
        from .tfidf_compat import align_idf_to_vocab, extract_idf_feature_array

        vocab = self.tfidf.vocabulary_
        n_features = int(max(vocab.values())) + 1
        raw = extract_idf_feature_array(self.tfidf)
        idf_arr = align_idf_to_vocab(raw, n_features)
        return self._infer_idf_table_weights(sentence, idf_arr)

    def _infer_tfidf_weights(
        self, sentence: List[str]
    ) -> Tuple[dict, List[str]]:
        """Compute weights: full TF-IDF (best) when tf_tfidf=True, else IDF-only."""
        existent: dict = {}
        missing: List[str] = []

        if self.tfidf is None:
            for word in sentence:
                existent[word] = 1.0
            return existent, missing

        # Set once at SentenceEmbedding load for sklearn pre-0.18 pickles (see tfidf_compat)
        if self._idf_per_feature is not None:
            return self._infer_idf_table_weights(sentence, self._idf_per_feature)

        if self.tf_tfidf:
            try:
                unseen = self.tfidf.transform([" ".join(sentence)]).toarray()
            except NotFittedError:
                logging.warning(
                    "TfidfVectorizer.transform failed (NotFittedError). "
                    "Using recovered IDF table (pass idf_per_feature from tfidf_compat "
                    "at init to avoid this path)."
                )
                return self._infer_idf_only_weights(sentence)
            vocab = self.tfidf.vocabulary_
            for word in sentence:
                try:
                    existent[word] = float(unseen[0][vocab[word]])
                except KeyError:
                    missing.append(word)
        else:
            return self._infer_idf_only_weights(sentence)

        return existent, missing

    def __iter__(self):
        if hasattr(self, "sentences"):
            for s in self.sentences:
                yield self.transform(s)


def embedding_filename_stem(word: str) -> Optional[str]:
    """
    Basename stem for an indexed embedding file (must match keyed2indexed / save_dense).
    """
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in word)
    return safe if safe else None


def save_dense(directory: str, filename: str, array: np.ndarray) -> Optional[None]:
    directory = os.path.normpath(directory) + os.sep
    try:
        stem = embedding_filename_stem(filename)
        if not stem:
            return None
        path = os.path.join(directory, stem + ".npy")
        np.save(path, array)
    except (UnicodeEncodeError, OSError):
        return None
    return None


def load_dense(filename: Union[str, Any]) -> np.ndarray:
    return np.load(filename, allow_pickle=True)


def _iter_keyed_vocab(keyed_model: Any):
    """Iterate over (word, _) for gensim 3.x (.vocab) and 4.x (.key_to_index)."""
    if hasattr(keyed_model, "key_to_index"):
        for w in keyed_model.key_to_index:
            yield w, None
    else:
        for w, _ in keyed_model.vocab.items():
            yield w, None


class vector_space:
    """
    Word embedding space backed by a directory of .npy files or a .tar.gz archive.

    For large indexed directories (millions of words), use ``lazy_index=True`` (default
    for directories): no full ``os.listdir`` at startup; each ``__getitem__`` loads
    only that word's ``.npy`` (same as inference). Vectors are never all loaded into RAM.

    ``lazy_index=False`` builds a word→path dict (faster repeated lookups, high RAM
    and slow startup for huge vocabs). Archives (``.tar.gz``) always use an in-memory
    member index.
    """

    def __init__(
        self,
        directory: str,
        sparse: bool = False,
        lazy_index: Optional[bool] = None,
    ):
        self.sparse = sparse
        self._tar = False
        self._lazy = False
        self._directory: Optional[str] = None
        ext = ".npz" if sparse else ".npy"
        self._ext = ext

        if directory.endswith(".tar.gz"):
            self._tar = True
            import tarfile
            self._tarfile = tarfile.open(directory, "r:*")
            file_list = self._tarfile.getnames()
            self.words = {
                os.path.basename(p).replace(ext, ""): p for p in file_list
            }
        else:
            directory = os.path.normpath(directory) + os.sep
            if lazy_index is None:
                lazy_index = True
            self._lazy = bool(lazy_index)
            if self._lazy:
                self._directory = directory
                self.words = {}  # unused in lazy mode; __getitem__ uses path resolution
            else:
                self.words = {
                    f.replace(ext, ""): os.path.join(directory, f)
                    for f in os.listdir(directory)
                    if f.endswith(ext)
                }

    def _path_for_word(self, item: str) -> Optional[str]:
        if self._tar:
            try:
                return self.words[item]
            except KeyError:
                return None
        stem = embedding_filename_stem(item)
        if not stem or self._directory is None:
            return None
        path = os.path.join(self._directory, stem + self._ext)
        return path if os.path.isfile(path) else None

    def get_embedding_dimension(self) -> int:
        """Infer vector size from one stored embedding (no full vocabulary scan in lazy mode)."""
        if self._tar:
            if not self.words:
                return 0
            first_key = next(iter(self.words.keys()))
            return int(np.asarray(self[first_key]).size)
        if self._lazy and self._directory is not None:
            with os.scandir(self._directory) as it:
                for entry in it:
                    if entry.name.endswith(self._ext) and entry.is_file():
                        arr = load_dense(entry.path)
                        return int(arr.size) if arr.ndim == 1 else int(arr.shape[-1])
            return 0
        if not self.words:
            return 0
        first_key = next(iter(self.words.keys()))
        return int(np.asarray(self[first_key]).size)

    def __getitem__(self, item: str) -> np.ndarray:
        if self._tar:
            path = self.words[item]
            f = self._tarfile.extractfile(self._tarfile.getmember(path))
            return np.load(f, allow_pickle=True)
        if self._lazy:
            path = self._path_for_word(item)
            if path is None:
                raise KeyError(item)
            return load_dense(path)
        path = self.words[item]
        return load_dense(path)

    def __contains__(self, item: str) -> bool:
        if self._tar or not self._lazy:
            return item in self.words
        return self._path_for_word(item) is not None

    def __len__(self) -> int:
        if self._tar or not self._lazy:
            return len(self.words)
        if self._directory is None:
            return 0
        if not hasattr(self, "_lazy_len"):
            n = 0
            with os.scandir(self._directory) as it:
                for entry in it:
                    if entry.name.endswith(self._ext) and entry.is_file():
                        n += 1
            self._lazy_len = n
        return int(self._lazy_len)


def keyed2indexed(
    keyed_model: Any,
    output_dir: str = "word_embeddings/",
    parallel: bool = True,
    n_jobs: int = -1,
) -> None:
    output_dir = os.path.normpath(output_dir) + os.sep
    os.makedirs(output_dir, exist_ok=True)

    if parallel:
        from joblib import Parallel, delayed
        words = list(keyed_model.key_to_index) if hasattr(keyed_model, "key_to_index") else list(keyed_model.vocab.keys())
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(save_dense)(output_dir, word, keyed_model[word])
            for word in words
        )
    else:
        for word, _ in _iter_keyed_vocab(keyed_model):
            save_dense(output_dir, word, keyed_model[word])


class streamer:
    def __init__(self, file_name: str, encoding: str = "utf-8"):
        self.file_name = file_name
        self._encoding = encoding

    def __iter__(self):
        with open(self.file_name, encoding=self._encoding, errors="replace") as f:
            for line in f:
                yield line.strip()
