#!/usr/bin/env python3
"""
Upload IDF and FastText weights to Hugging Face so wisse autodownload works after install.

Options:
  1) From local files (after downloading from MEGA manually):
       python hf_model/upload_assets_to_hf.py --fasttext /path/to/fasttext-300-indexed.tar.gz --idf /path/to/idf-en.pkl
  2) From MEGA (requires: pip install mega.py huggingface_hub; then huggingface-cli login):
       python hf_model/upload_assets_to_hf.py --from-mega

Repo: https://huggingface.co/datasets/iarroyof/wisse-models
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

# MEGA links from paper (README)
MEGA_FASTTEXT_URL = "https://mega.nz/#!zKBUzL7J!V2BN6hsb2_I61WbM3C8OIrSnJotFyxaqfBmapddns4Y"
MEGA_IDF_URL = "https://mega.nz/#!WPx1iYwA!okha3WRVIksZJuq7cJKeKzplxuDYqOa0aq31hyMHvAo"

REPO_ID = os.environ.get("WISSE_HF_REPO", "iarroyof/wisse-models")
REPO_TYPE = os.environ.get("WISSE_HF_REPO_TYPE", "dataset")
HF_FASTTEXT_FILENAME = "fasttext-300-indexed.tar.gz"
HF_IDF_FILENAME = "idf-en.pkl"


def download_from_mega(url: str, dest_dir: Path) -> Path:
    """Download a file from MEGA URL into dest_dir. Returns path to downloaded file/folder."""
    try:
        from mega import Mega
    except ImportError:
        print("For --from-mega install: pip install mega.py", file=sys.stderr)
        raise SystemExit(1)
    mega = Mega()
    m = mega.login()
    # download_url(url, dest_path) downloads to dest_path; may return path or None
    result = m.download_url(url, str(dest_dir))
    if result is not None:
        p = Path(result) if isinstance(result, str) else Path(result[0])
        if p.exists():
            return p
    # Else find the newest file/dir in dest_dir (MEGA often creates a subdir or file)
    entries = list(dest_dir.iterdir())
    if not entries:
        raise RuntimeError(f"MEGA download failed for {url}")
    # Prefer a .tar.gz or .pkl file, else the single subdir or file
    for e in entries:
        if e.suffix in (".gz", ".tar.gz", ".pkl", ".pk"):
            return e
    if len(entries) == 1:
        return entries[0]
    raise RuntimeError(f"Unexpected MEGA output: {entries}")


def upload_to_hf(local_path: Path, path_in_repo: str, repo_id: str, repo_type: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Add {path_in_repo}",
    )
    print(f"Uploaded {local_path.name} -> {path_in_repo}")


def main():
    ap = argparse.ArgumentParser(description="Upload FastText and IDF assets to Hugging Face")
    ap.add_argument("--fasttext", type=str, help="Local path to fasttext-300-indexed.tar.gz")
    ap.add_argument("--idf", type=str, help="Local path to idf-en.pkl (or pretrained_idf.pk)")
    ap.add_argument("--from-mega", action="store_true", help="Download from MEGA first, then upload")
    ap.add_argument("--repo", type=str, default=REPO_ID, help="HF repo id (default: iarroyof/wisse-models)")
    ap.add_argument("--repo-type", type=str, default=REPO_TYPE, choices=("dataset", "model"))
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
        HfApi()
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        print("Then: huggingface-cli login", file=sys.stderr)
        raise SystemExit(1)

    fasttext_path = None
    idf_path = None

    if args.from_mega:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            print("Downloading FastText from MEGA...")
            fasttext_path = download_from_mega(MEGA_FASTTEXT_URL, tmp)
            # MEGA may return a folder; if so, tar it
            if fasttext_path.is_dir():
                tarball = tmp / HF_FASTTEXT_FILENAME
                with tarfile.open(tarball, "w:gz") as tf:
                    tf.add(fasttext_path, arcname=fasttext_path.name)
                fasttext_path = tarball
            elif not fasttext_path.suffix == ".gz":
                dest = tmp / HF_FASTTEXT_FILENAME
                shutil.copy(fasttext_path, dest)
                fasttext_path = dest
            print("Downloading IDF from MEGA...")
            idf_path = download_from_mega(MEGA_IDF_URL, tmp)
            if idf_path.is_dir():
                # single file in dir
                files = list(idf_path.iterdir())
                if len(files) == 1 and files[0].suffix in (".pkl", ".pk"):
                    idf_path = files[0]
            if not str(idf_path).endswith(".pkl") and not str(idf_path).endswith(".pk"):
                dest = tmp / HF_IDF_FILENAME
                shutil.copy(idf_path, dest)
                idf_path = dest
            print("Uploading to Hugging Face...")
            upload_to_hf(fasttext_path, HF_FASTTEXT_FILENAME, args.repo, args.repo_type)
            upload_to_hf(idf_path, HF_IDF_FILENAME, args.repo, args.repo_type)
    else:
        if not args.fasttext or not args.idf:
            print("Provide --fasttext and --idf paths, or use --from-mega", file=sys.stderr)
            raise SystemExit(1)
        fasttext_path = Path(args.fasttext)
        idf_path = Path(args.idf)
        if not fasttext_path.exists():
            print(f"Not found: {fasttext_path}", file=sys.stderr)
            raise SystemExit(1)
        if not idf_path.exists():
            print(f"Not found: {idf_path}", file=sys.stderr)
            raise SystemExit(1)
        print("Uploading to Hugging Face...")
        upload_to_hf(fasttext_path, HF_FASTTEXT_FILENAME, args.repo, args.repo_type)
        upload_to_hf(idf_path, HF_IDF_FILENAME, args.repo, args.repo_type)

    print(f"Done. View: https://huggingface.co/{args.repo_type}s/{args.repo}")


if __name__ == "__main__":
    main()
