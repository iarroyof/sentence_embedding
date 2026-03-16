"""Legacy setup.py for editable installs; prefer building via pyproject.toml."""
import os
from setuptools import setup

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()

setup(
    name="wisse",
    version="0.1.0",
    description="Sentence embeddings (WISSE) with SBERT-like API for downstream NLP.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Ignacio Arroyo-Fernandez",
    author_email="iaf@ciencias.unam.mx",
    url="https://github.com/iarroyof/sentence_embedding",
    license="BSD-3-Clause",
    packages=["wisse"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scikit-learn>=1.0",
        "gensim>=4.0",
        "joblib>=1.0",
        "requests>=2.25",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
