FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY . .

# Pre-index the committed corpus into both stores (embedding model is downloaded and cached at build time)
RUN python -m src.cli ingest --store lancedb && python -m src.cli ingest --store qdrant

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD curl -fs http://localhost:${PORT:-8000}/health || exit 1

ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
