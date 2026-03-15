# -*- coding: utf-8 -*-
"""
Model and TF-IDF weight registry and autodownload from Hugging Face.
Weights are full TF-IDF (best performing). On first use of SentenceEmbedding(),
assets are downloaded to ~/.wisse (or WISSE_HOME).
"""
from __future__ import annotations

import logging
import os
import pickle
import tarfile
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve

import requests

logger = logging.getLogger(__name__)

# Hugging Face repo hosting the paper's assets (Wikipedia FastText + IDF)
# Override with WISSE_HF_REPO=org/repo (repo_type is inferred: dataset or model)
WISSE_HF_REPO = os.environ.get("WISSE_HF_REPO", "iarroyof/wisse-models")
WISSE_HF_REPO_TYPE = os.environ.get("WISSE_HF_REPO_TYPE", "dataset")  # "dataset" or "model"


def _cache_dir() -> Path:
    base = os.environ.get("WISSE_HOME", os.path.expanduser("~/.wisse"))
    return Path(base)


def _hf_url(filename: str) -> str:
    """Build Hugging Face resolve URL for a file in the WISSE repo."""
    return f"https://huggingface.co/{WISSE_HF_REPO_TYPE}s/{WISSE_HF_REPO}/resolve/main/{filename}"


def _embedding_url(entry: dict, key: str) -> str:
    """Resolve URL for an embedding: env override or HF default."""
    if key == "wisse-fasttext-300":
        return os.environ.get("WISSE_FASTTEXT_URL") or _hf_url(entry["filename"])
    return os.environ.get("WISSE_EMBEDDING_URL") or _hf_url(entry["filename"])


def _idf_url(entry: dict) -> str:
    """Resolve URL for IDF: env override or HF default."""
    return os.environ.get("WISSE_IDF_URL") or _hf_url(entry["filename"])


# Registry: name -> {filename on HF, format}. URLs built from WISSE_HF_REPO for transparent autodownload.
EMBEDDING_REGISTRY = {
    "wisse-glove-300": {
        "filename": "glove-300-indexed.tar.gz",
        "format": "indexed_tar",
    },
    "wisse-fasttext-300": {
        "filename": "fasttext-300-indexed.tar.gz",
        "format": "indexed_tar",
    },
}

# TF-IDF weights (fitted TfidfVectorizer; .transform() yields full TF-IDF)
IDF_REGISTRY = {
    "wisse-idf-en": {
        "filename": "idf-en.pkl",
        "format": "pickle",
    },
}

# Default: Wikipedia FastText + TF-IDF weights (auto-downloaded from HF on first use)
DEFAULT_EMBEDDING_KEY = "wisse-fasttext-300"
DEFAULT_IDF_KEY = "wisse-idf-en"


def _download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    logger.info("%s from Hugging Face to %s", desc, dest)
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
        logger.info("Downloaded %s (%s bytes)", dest.name, written)
    except Exception:
        try:
            urlretrieve(url, str(dest))
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {url}. "
                f"If the repo is private or URL changed, set WISSE_HF_REPO or use a local path. {e}"
            ) from e
    return dest


def get_embedding_path(name_or_path: str) -> Path:
    """
    Resolve embedding to a local path. If it's a registry name, download to cache.
    If it's a path, return it as-is (must exist).
    """
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()

    if name_or_path in EMBEDDING_REGISTRY:
        entry = EMBEDDING_REGISTRY[name_or_path]
        url = entry.get("url") or _embedding_url(entry, name_or_path)
        fmt = entry.get("format", "indexed_tar")
        cache = _cache_dir() / "embeddings" / name_or_path
        if fmt == "indexed_tar":
            archive = cache / "archive.tar.gz"
            extract_dir = cache / "extracted"
            if not extract_dir.exists():
                _download_file(url, archive, desc=name_or_path)
                cache.mkdir(parents=True, exist_ok=True)
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, "r:*") as tf:
                    tf.extractall(extract_dir)
            # If archive had a single top-level dir with .npy files, use it
            subdirs = [x for x in extract_dir.iterdir() if x.is_dir()]
            npy_in_root = list(extract_dir.glob("*.npy"))
            if not npy_in_root and len(subdirs) == 1:
                return subdirs[0]
            return extract_dir
        return cache

    raise FileNotFoundError(f"Embedding not found: {name_or_path}")


def get_idf_path(name_or_path: str) -> Path:
    """
    Resolve IDF artifact to a local path. If registry name, download from HF to cache.
    """
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()

    if name_or_path in IDF_REGISTRY:
        entry = IDF_REGISTRY[name_or_path]
        url = entry.get("url") or _idf_url(entry)
        cache = _cache_dir() / "idf" / name_or_path
        cache.parent.mkdir(parents=True, exist_ok=True)
        dest = cache.with_suffix(Path(entry["filename"]).suffix or ".pkl")
        _download_file(url, dest, desc=name_or_path)
        return dest

    raise FileNotFoundError(f"IDF artifact not found: {name_or_path}")


def load_idf(path: Path):
    """Load pickled TfidfVectorizer from path (used for full TF-IDF weighting via .transform())."""
    path = Path(path)
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin-1")
