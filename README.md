# WISSE — Sentence embeddings

Sentence embeddings via **entropy-weighted series** (TF-IDF–weighted word embeddings). No language or knowledge resources required. **Python 3.8+**.

SBERT-like API: `encode()` and `similarity()`. **Default model keys** (`wisse-fasttext-300`, `wisse-idf-en`) point to the Hugging Face repo; once the **paper’s Wikipedia FastText and TF-IDF assets are uploaded** there (one-time, see [Uploading assets](#uploading-assets-to-hugging-face)), they auto-download on first use to `~/.wisse`. Until then, use local paths (e.g. after [downloading from MEGA](#pretrained-assets-manual-download)).

---

## Quick start

**With assets on Hugging Face** (after one-time upload of the paper assets):

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

**With local paths** (e.g. after downloading from MEGA):

```python
model = SentenceEmbedding(
    model_name_or_path="/path/to/indexed_fasttext/",
    idf_name_or_path="/path/to/idf-en.pkl",
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

From PyPI:

```bash
pip install wisse-sentence
```

From the repo (editable):

```bash
pip install -e .
```

Requirements: Python ≥3.8, `numpy`, `scikit-learn`, `gensim`, `joblib`, `requests`.

---

## Default models (Hugging Face)

The package expects **Wikipedia-trained FastText (300d)** and **TF-IDF weights** at the Hugging Face repo. Once those files are uploaded (see [Uploading assets](#uploading-assets-to-hugging-face)), they download on **first use** to `~/.wisse` (or `$WISSE_HOME`).

- **Registry keys**: `wisse-fasttext-300`, `wisse-idf-en` (and optionally `wisse-glove-300`).
- **Override**: `WISSE_HF_REPO`, `WISSE_HF_REPO_TYPE`, or `WISSE_FASTTEXT_URL`, `WISSE_IDF_URL`, `WISSE_EMBEDDING_URL`.

---

## Repository layout

Clean Python package layout:

```
sentence_embedding/
├── wisse/                 # Package
│   ├── __init__.py
│   ├── wisse.py           # Core: TF-IDF weighting, vector_space, keyed2indexed
│   ├── model.py           # SentenceEmbedding (SBERT-like API)
│   ├── similarity.py      # Pairwise similarity helpers
│   ├── download.py        # HF registry and autodownload
│   └── cli.py             # wisse-encode, keyed2indexed entry points
├── tests/                 # Pytest suite
├── hf_model/              # Hugging Face model card (README.md) and push script
├── setup.py
├── pyproject.toml
├── requirements.txt
├── run_tests.py
├── keyed2indexed.py      # Standalone script (or use CLI after install)
├── LICENSE
├── .gitignore
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

wisse-train --wikipedia en --idf-out idf-en.pkl --embeddings-out fasttext-300-indexed
wisse-train --corpus-dir ./my_texts --document-unit paragraph --idf-out idf.pkl
```

**Train IDF + FastText** (new operating mode): from a directory of plain text files or from Wikipedia (Hugging Face). Produces WISSE-ready IDF pickle and indexed FastText embeddings. For Wikipedia you need `pip install ".[train]"` (adds `datasets`).

- `--corpus-dir PATH` — directory of plain text files, or  
- `--wikipedia LANG` — e.g. `en`, `es` (downloads from HF `wikimedia/wikipedia`).
- `--document-unit article|paragraph` — one doc per file/article vs per paragraph.
- `--idf-out`, `--embeddings-out` — explicit output paths (defaults: `idf-<lang>.pkl`, `fasttext-300-indexed`).
- `--binary-out PATH` — optionally save FastText in Word2Vec binary format.
- `--dim`, `--window`, `--min-count`, `--epochs` — paper defaults (300, 5, 5, 5), all configurable.
- `--cap-articles`, `--cap-tokens` — optional cap with efficient random sampling; default for Wikipedia: 500k articles / 100M tokens.

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

| Test | Description | Needs real pretrained assets? |
|------|-------------|-------------------------------|
| `test_00_install` | Package and public API import | No |
| `test_01_toy_tfidf_fasttext_and_helpers` | Toy TF-IDF FastText + all helpers (synthetic data) | No |
| `test_02_paper_wikipedia_assets` | **Paper’s Wikipedia FastText + TF-IDF** | **Yes** (see below) |
| `test_new_user_full_workflow` | New user: **mocked** download, toy sentences | No (uses mocks) |
| `test_encode_similarity` | encode/similarity and low-level API (synthetic) | No |

**Tests that use the real pretrained models** (optional): `test_02_paper_wikipedia_assets` runs only when assets are available:

- Set `WISSE_PAPER_FASTTEXT_DIR` and `WISSE_PAPER_IDF_PATH` to local paths (e.g. after downloading from the MEGA links below), or  
- Set `WISSE_TEST_HF_REGISTRY=1` to use the Hugging Face registry (only works **after** the paper assets have been uploaded to the HF repo).

---

## Uploading assets to Hugging Face

For `SentenceEmbedding()` to work with defaults (no local paths), the **paper’s** FastText and TF-IDF files must be on the Hub. There are no synthetic “minimal” defaults — only the real pretrained assets are useful.

1. **Get the assets**: Download from MEGA (links in [Pretrained assets](#pretrained-assets-manual-download)): indexed FastText and the TF-IDF pickle. Optionally pack the FastText directory as `fasttext-300-indexed.tar.gz`.
2. **Upload**: Use `hf_model/upload_assets_to_hf.py` with local paths, or the Hub UI/CLI. Full steps: **`hf_model/PUSH_TO_HF.md`**.
3. **Repo**: [huggingface.co/datasets/iarroyof/wisse-models](https://huggingface.co/datasets/iarroyof/wisse-models)

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
