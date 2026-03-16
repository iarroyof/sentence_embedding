# Train WISSE IDF + FastText from Wikipedia, register assets, run toy example.
# Build: docker build -t wisse-train .
# Run:   docker run --rm wisse-train
# Optional: mount a volume to keep models: docker run --rm -v wisse-models:/workspace/models wisse-train

FROM python:3.11-slim

WORKDIR /workspace

# Install system deps used by Python libs (optional, for faster builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project and install package with train extra (datasets for Wikipedia)
COPY pyproject.toml setup.py README.md LICENSE ./
COPY wisse/ ./wisse/

RUN pip install --no-cache-dir -e ".[train]"

# Output dir for training; WISSE_HOME for registered assets
ENV WISSE_HOME=/workspace/models
ENV OUTPUT_DIR=/workspace/output

RUN mkdir -p /workspace/output /workspace/models

# Train from Wikipedia (small cap so the container finishes in reasonable time)
# Override with docker run -e CAP_ARTICLES=50000 etc.
ENV CAP_ARTICLES=2000
ENV CAP_TOKENS=5000000

RUN wisse-train --wikipedia en \
    --document-unit article \
    --idf-out /workspace/output/idf-en.pkl \
    --embeddings-out /workspace/output/fasttext-300-indexed \
    --cap-articles ${CAP_ARTICLES} \
    --cap-tokens ${CAP_TOKENS} \
    --dim 300 --window 5 --min-count 5 --epochs 3

# Copy script that registers assets and runs toy example (build context = repo root)
COPY docker/train_and_demo.py /workspace/train_and_demo.py

# Register trained files under WISSE_HOME and run toy example
CMD ["python", "/workspace/train_and_demo.py"]
