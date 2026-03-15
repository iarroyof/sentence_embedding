#!/usr/bin/env python3
"""
Push the model card (README.md) from hf_model/ to the Hugging Face repo.
Requires: pip install huggingface_hub; huggingface-cli login

Usage (from repo root):
  python hf_model/push_model_card_to_hf.py
  # Or with custom repo:
  WISSE_HF_REPO=yourorg/wisse-models python hf_model/push_model_card_to_hf.py
"""
import os
from pathlib import Path

REPO_ID = os.environ.get("WISSE_HF_REPO", "iarroyof/wisse-models")
REPO_TYPE = os.environ.get("WISSE_HF_REPO_TYPE", "dataset")  # or "model"

def main():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install: pip install huggingface_hub")
        print("Then: huggingface-cli login")
        raise SystemExit(1)

    script_dir = Path(__file__).resolve().parent
    readme_path = script_dir / "README.md"
    if not readme_path.exists():
        print(f"Not found: {readme_path}")
        raise SystemExit(1)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Update model card (README) from sentence_embedding/hf_model",
    )
    print(f"Uploaded README.md to {REPO_TYPE}s/{REPO_ID}")
    print(f"View: https://huggingface.co/{REPO_TYPE}s/{REPO_ID}")

if __name__ == "__main__":
    main()
