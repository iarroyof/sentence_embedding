#!/usr/bin/env python3
"""
Run the full test suite in order:
  1. Package installs (import + public API)
  2. Toy TF-IDF FastText embeddings + all helpers
  3. Additional encode/similarity and low-level tests
"""
import subprocess
import sys


def main():
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_00_install.py",
        "tests/test_01_toy_tfidf_fasttext_and_helpers.py",
        "tests/test_02_paper_wikipedia_assets.py",
        "tests/test_new_user_full_workflow.py",
        "tests/test_encode_similarity.py",
        "-v",
        "--tb=short",
    ]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
