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

# Pre-generate corpus and pre-index into LanceDB
RUN python scripts/generate_corpus.py && python -m src.cli ingest --store lancedb

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
