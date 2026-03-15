# -*- coding: utf-8 -*-
"""First test: ensure the package installs and public API is importable."""
import pytest


def test_package_installs():
    """Package must be importable and expose version and all public helpers."""
    import wisse

    assert hasattr(wisse, "__version__")
    assert wisse.__version__
    assert isinstance(wisse.__version__, str)

    # All public API from __all__
    assert hasattr(wisse, "wisse")
    assert hasattr(wisse, "vector_space")
    assert hasattr(wisse, "keyed2indexed")
    assert hasattr(wisse, "streamer")
    assert hasattr(wisse, "load_dense")
    assert hasattr(wisse, "save_dense")
    assert hasattr(wisse, "SentenceEmbedding")
    assert hasattr(wisse, "similarity")
    assert hasattr(wisse, "similarity_module")

    # Core deps used by the package
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    assert np is not None
    assert TfidfVectorizer is not None
