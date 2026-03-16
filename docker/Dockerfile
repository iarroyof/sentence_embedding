# Train WISSE IDF + FastText from Wikipedia, register assets, run toy example.
# Build once (no training at build time):  docker build -f docker/Dockerfile -t wisse-train .
# Run with default cap (quick):            docker run --rm wisse-train
# Run with 6B tokens (GloVe-style):       docker run --rm -e CAP_TOKENS=6000000000 wisse-train
# Run with 16B tokens (FastText-style):   docker run --rm -e CAP_TOKENS=16000000000 wisse-train
# Keep models on a volume:                 docker run --rm -v wisse-models:/workspace/models wisse-train

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

# Default cap (quick run). Override at run time: -e CAP_TOKENS=6000000000
ENV CAP_ARTICLES=2000
ENV CAP_TOKENS=5000000
ENV WIKIPEDIA_LANG=en

# Training runs at container start, not at build (so you can change cap without rebuilding)
COPY docker/train_and_demo.py /workspace/train_and_demo.py
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
