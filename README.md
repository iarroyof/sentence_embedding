# WISSE — Sentence embeddings

Sentence embeddings via **entropy-weighted series** (TF-IDF–weighted word embeddings). No language or knowledge resources required. **Python 3.8+**.

SBERT-like API: `encode()` and `similarity()`. **Default embeddings and TF-IDF weights are on Hugging Face and auto-download on first use** (not at install time).

---

## Quick start

```python
from wisse import SentenceEmbedding

model = SentenceEmbedding()  # downloads to ~/.wisse on first use

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)  # shape (3, 300)
sim = model.similarity(embeddings, embeddings)
```

Local paths (no download):

```python
model = SentenceEmbedding(
    model_name_or_path="/path/to/indexed_embeddings/",
    idf_name_or_path="/path/to/tfidf.pkl",
)
embeddings = model.encode(["First sentence.", "Second sentence."])
```

Similarity: `"cosine"`, `"dot"`, `"euclidean"`, `"manhattan"`.

```python
from wisse import similarity
import numpy as np
s = similarity(embeddings, embeddings, similarity_fn="cosine")
```

---

## Installation

```bash
pip install -e .
```

Requirements: Python ≥3.8, `numpy`, `scikit-learn`, `gensim`, `joblib`, `requests`.

---

## Default models (Hugging Face)

Wikipedia-trained **FastText (300d)** and **TF-IDF weights** (full TF-IDF, best performing) are on the Hub. They download on **first use** to `~/.wisse` (or `$WISSE_HOME`).

- **Default**: `wisse-fasttext-300` + `wisse-idf-en` → `SentenceEmbedding()` works out of the box.
- **Registry**: `wisse-glove-300`, `wisse-fasttext-300` (embeddings); `wisse-idf-en` (TF-IDF).
- **Override**: `WISSE_HF_REPO`, `WISSE_HF_REPO_TYPE`, or `WISSE_FASTTEXT_URL`, `WISSE_IDF_URL`, `WISSE_EMBEDDING_URL`.

---

## Repository layout

```
sentence_embedding/
├── wisse/                 # Package
│   ├── __init__.py
│   ├── wisse.py           # Core: TF-IDF weighting, vector_space, keyed2indexed
│   ├── model.py           # SentenceEmbedding (SBERT-like API)
│   ├── similarity.py      # Pairwise similarity helpers
│   ├── download.py        # HF registry and autodownload
│   └── cli.py             # wisse-encode, keyed2indexed entry points
├── tests/
├── hf_model/              # Hugging Face model card and push instructions
├── setup.py
├── pyproject.toml
├── requirements.txt
├── run_tests.py
├── keyed2indexed.py       # Standalone script (or use: keyed2indexed after install)
└── README.md
```

---

## CLI

After `pip install`:

```bash
wisse-encode --input sentences.txt --output vectors.npy
wisse-encode --input sentences.txt --output out.npy --model wisse-fasttext-300 --idf wisse-idf-en

keyed2indexed --input model.bin --output output_indexed
keyed2indexed --input model.vec --txt --output output_indexed
```

From repo without installing:

```bash
python keyed2indexed.py --input model.bin --output output_indexed
```

---

## Low-level usage

Convert word2vec to indexed format and use the WISSE combiner:

```python
import wisse
from gensim.models.keyedvectors import KeyedVectors

kv = KeyedVectors.load_word2vec_format("/path/to/embeddings.bin", binary=True)
wisse.keyed2indexed(kv, "/path/to/output_dir/")

embedding = wisse.vector_space("/path/to/output_dir/")
# embedding["word"] → array

import pickle
with open("/path/to/tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)
w = wisse.wisse(embedding, vectorizer=vectorizer, tf_tfidf=True, combiner="sum", generate=True)
vec = w.infer_sentence("this is a sentence")
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
# or
python run_tests.py
```

| Test | Description |
|------|-------------|
| `test_00_install` | Package and public API import |
| `test_01_toy_tfidf_fasttext_and_helpers` | Toy TF-IDF FastText + all helpers |
| `test_02_paper_wikipedia_assets` | Paper assets (skip unless paths or HF) |
| `test_new_user_full_workflow` | New user: mock download, default model, toy sentences |
| `test_encode_similarity` | encode/similarity and low-level API |

Paper assets (optional): set `WISSE_PAPER_FASTTEXT_DIR` and `WISSE_PAPER_IDF_PATH` to local paths (e.g. from MEGA links below), or `WISSE_TEST_HF_REGISTRY=1` for HF registry test.

---

## Hugging Face model

Artifacts are intended to live on the Hub for autodownload. The `hf_model/` folder is the model card; see `hf_model/PUSH_TO_HF.md` for upload steps.

- **Repo**: [huggingface.co/datasets/iarroyof/wisse-models](https://huggingface.co/datasets/iarroyof/wisse-models)

---

## Pretrained assets (manual download)

If you prefer not to use the Hub, download and extract manually, then pass paths to `SentenceEmbedding(...)` or `wisse.vector_space(path)`:

- **FastText** (Wikipedia 300d): [idx_FastText](https://mega.nz/#!zKBUzL7J!V2BN6hsb2_I61WbM3C8OIrSnJotFyxaqfBmapddns4Y)
- **Word2Vec** (Wikipedia 300d): [idx_Word2Vec](https://mega.nz/#!yS4mHTDT!QF28R9jIVRnpGr3kwRYlMMqaJoT-1QMoGwNbkDmac3E)
- **GloVe** (840B 300d): [idx_Glove](https://mega.nz/#!Pa4GQC7Y!ccQ9398j234ixYcqhbIqEUPj-jS-aC3HXdExMk5PyQs)
- **Dep2Vec** (300d): [idx_Dep2Vec](https://mega.nz/#!CHYXjbrb!jk3gW5DaVOW4yksq-B4eGKJDQv9LSVPxmBJqM68rZHs)
- **TF-IDF** (Wikipedia, stop words ignored): [pretrained_idf](https://mega.nz/#!WPx1iYwA!okha3WRVIksZJuq7cJKeKzplxuDYqOa0aq31hyMHvAo)

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
