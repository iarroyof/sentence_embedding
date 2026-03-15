---
license: bsd-3-clause
language:
  - en
tags:
  - sentence-transformers
  - sentence-similarity
  - embeddings
  - tfidf
  - wisse
  - nlp
datasets:
  - wikipedia
---

# WISSE — Sentence embeddings (model artifacts)

This Hugging Face repo hosts **default model artifacts** for [WISSE](https://github.com/iarroyof/sentence_embedding): indexed word embeddings and **TF-IDF weights** for sentence embeddings (full TF-IDF, best performing). The code and documentation live on **GitHub**; this repo mirrors the project for easy autodownload via the `wisse` Python package.

- **GitHub (code & docs):** [iarroyof/sentence_embedding](https://github.com/iarroyof/sentence_embedding)
- **Paper:** [Unsupervised Sentence Representations as Word Information Series: Revisiting TF–IDF](https://arxiv.org/abs/1710.06524) (Arroyo-Fernández et al., 2017)

---

## Model card summary

- **Model type:** Sentence embeddings (WISSE: TF-IDF–weighted word-embedding combination)
- **Language:** English
- **Training data:** English Wikipedia (word embeddings and TF-IDF fit)
- **Intended use:** Sentence representation for similarity, retrieval, clustering, and other downstream NLP tasks. Use with the `wisse` Python package.

---

## What’s in this repo

| Asset | Description |
|--------|-------------|
| `glove-300-indexed.tar.gz` | Indexed GloVe 300d word embeddings (WISSE format) |
| `fasttext-300-indexed.tar.gz` | Indexed FastText 300d word embeddings (WISSE format) |
| `idf-en.pkl` | Pickled TfidfVectorizer for full TF-IDF weighting (trained on English Wikipedia) |

They **auto-download on first use** (when you call `SentenceEmbedding()` or use registry keys); no download at install time. Cache: `~/.wisse` or `$WISSE_HOME`.

---

## Quick start (mirrors GitHub README)

Install the package from GitHub:

```bash
pip install git+https://github.com/iarroyof/sentence_embedding.git
```

Then use the SBERT-like API; artifacts will download from this Hugging Face repo on first use:

```python
from wisse import SentenceEmbedding

model = SentenceEmbedding()  # uses default embedding + TF-IDF weights from this repo

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)  # shape (3, 300)
sim = model.similarity(embeddings, embeddings)
```

Using a specific embedding and local/HF paths:

```python
# From this HF repo (autodownload)
model = SentenceEmbedding(
    model_name_or_path="wisse-fasttext-300",
    idf_name_or_path="wisse-idf-en",
)
embeddings = model.encode(["First sentence.", "Second sentence."])
```

---

## Installation & requirements

- Python ≥3.8  
- `numpy`, `scikit-learn`, `gensim`, `joblib`, `requests`

See the [GitHub README](https://github.com/iarroyof/sentence_embedding#readme) for full installation, CLI, and low-level usage.

---

## Citation

```bibtex
@article{arroyo2017unsupervised,
  title={Unsupervised Sentence Representations as Word Information Series: Revisiting TF--IDF},
  author={Arroyo-Fern{\'a}ndez, Ignacio and M{\'e}ndez-Cruz, Carlos-Francisco and Sierra, Gerardo and Torres-Moreno, Juan-Manuel and Sidorov, Grigori},
  journal={arXiv preprint arXiv:1710.06524},
  year={2017}
}
```

---

## Links

- **Source & docs:** [github.com/iarroyof/sentence_embedding](https://github.com/iarroyof/sentence_embedding)
- **Paper:** [arXiv:1710.06524](https://arxiv.org/abs/1710.06524)
