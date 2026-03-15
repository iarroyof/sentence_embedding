# Pushing this model to Hugging Face Hub

This folder mirrors the **Hugging Face model repo** [iarroyof/wisse-models](https://huggingface.co/iarroyof/wisse-models) (or your chosen org/name).

## 1. Create the repo on the Hub

- Go to [huggingface.co/new](https://huggingface.co/new) and create a model repo (e.g. `iarroyof/wisse-models`).
- Optionally make it a dataset repo if you prefer; the `wisse` package uses the same resolve URLs for datasets.

## 2. Clone and add this folder’s content

```bash
# Install Hugging Face CLI: pip install huggingface_hub
huggingface-cli login

# Clone the empty repo (replace with your repo id)
git clone https://huggingface.co/iarroyof/wisse-models hf_wisse_models
cd hf_wisse_models
```

Copy the contents of this folder (`hf_model/`) into the cloned repo root:

- `README.md` → model card
- (Optional) `.gitattributes` for Git LFS if you add large files

## 3. Upload artifact files

Upload the actual files (e.g. from your build or external URLs):

- `glove-300-indexed.tar.gz`
- `fasttext-300-indexed.tar.gz`
- `idf-en.pkl`

Using the Hub web UI (drag & drop) or CLI:

```bash
huggingface-cli upload iarroyof/wisse-models glove-300-indexed.tar.gz .
huggingface-cli upload iarroyof/wisse-models fasttext-300-indexed.tar.gz .
huggingface-cli upload iarroyof/wisse-models idf-en.pkl .
```

Or from Python:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj="path/to/glove-300-indexed.tar.gz", path_in_repo="glove-300-indexed.tar.gz", repo_id="iarroyof/wisse-models", repo_type="model")
```

## 4. Keep in sync with GitHub

When the GitHub README or usage changes, update this folder’s `README.md` and push to the HF repo so the model card stays in sync.

## 5. Run the same tests after HF setup

From the **main sentence_embedding repo** (GitHub clone), run the full test suite to confirm nothing regressed:

```bash
pip install -e ".[dev]"
pytest tests/test_00_install.py tests/test_01_toy_tfidf_fasttext_and_helpers.py tests/test_encode_similarity.py -v
# or
python run_tests.py
```

This runs: (1) package install check, (2) toy TF-IDF FastText embeddings + all helpers, (3) encode/similarity and low-level tests.
