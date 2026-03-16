# Pushing this model to Hugging Face Hub

This folder mirrors the **Hugging Face model repo** [iarroyof/wisse-models](https://huggingface.co/iarroyof/wisse-models) (or your chosen org/name).

## 1. Create the repo on the Hub

- Go to [huggingface.co/new](https://huggingface.co/new) and create a model repo (e.g. `iarroyof/wisse-models`).
- Optionally make it a dataset repo if you prefer; the `wisse` package uses the same resolve URLs for datasets.

## 2. Push the model card (README) only

From the **sentence_embedding** repo root, after `huggingface-cli login`:

```bash
pip install huggingface_hub
python hf_model/push_model_card_to_hf.py
```

This uploads `hf_model/README.md` to the Hub. To use a different repo: `WISSE_HF_REPO=org/repo python hf_model/push_model_card_to_hf.py`.

## 2b. Or clone and add this folder’s content (git workflow)

```bash
# Install Hugging Face CLI: pip install huggingface_hub
huggingface-cli login

# Clone the repo (replace with your repo id)
git clone https://huggingface.co/datasets/iarroyof/wisse-models hf_wisse_models
cd hf_wisse_models
```

Copy the contents of this folder (`hf_model/`) into the cloned repo root:

- `README.md` → model card
- (Optional) `.gitattributes` for Git LFS if you add large files

## 3. Upload artifact files (required for autodownload)

The **wisse** package has no synthetic “minimal” defaults. Autodownload works only with the **paper’s real assets** (Wikipedia FastText + TF-IDF). Upload these to the Hub so `SentenceEmbedding()` can autodownload on first use:

- `fasttext-300-indexed.tar.gz` — Wikipedia 300d FastText in indexed (WISSE) format
- `idf-en.pkl` — TF-IDF weights (fitted TfidfVectorizer) from Wikipedia

**Option A — From local files** (recommended: download from MEGA first, then upload):

```bash
# 1) Download from MEGA (see README “Pretrained assets” for links), then:
huggingface-cli login
python hf_model/upload_assets_to_hf.py --fasttext /path/to/fasttext-300-indexed.tar.gz --idf /path/to/idf-en.pkl
```

**Option B — From MEGA directly** (requires mega.py; may fail on some networks):

```bash
pip install mega.py huggingface_hub
huggingface-cli login
python hf_model/upload_assets_to_hf.py --from-mega
```

**Option C — Hub UI or CLI:** Upload the two files via [huggingface.co/datasets/iarroyof/wisse-models](https://huggingface.co/datasets/iarroyof/wisse-models) (drag & drop) or:

```bash
huggingface-cli upload iarroyof/wisse-models fasttext-300-indexed.tar.gz . --repo-type dataset
huggingface-cli upload iarroyof/wisse-models idf-en.pkl . --repo-type dataset
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
