from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Groq LLM configuration (generator)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    # Judge model for the answer-quality evaluation. Deliberately a different model
    # family from the generator to limit self-enhancement bias.
    judge_model: str = Field(default="openai/gpt-oss-120b", alias="JUDGE_MODEL")

    # Vector store selection
    default_vector_store: Literal["lancedb", "qdrant"] = Field(default="lancedb", alias="DEFAULT_VECTOR_STORE")
    lancedb_dir: str = Field(default=str(BASE_DIR / "data" / "lancedb"), alias="LANCEDB_DIR")
    qdrant_dir: str = Field(default=str(BASE_DIR / "data" / "qdrant"), alias="QDRANT_DIR")

    # Embedding configuration
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    # Ingestion defaults
    default_chunk_size: int = Field(default=500, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=50, alias="DEFAULT_CHUNK_OVERLAP")
    max_upload_mb: int = Field(default=10, alias="MAX_UPLOAD_MB")

    # Retrieval: minimum cosine similarity of the top chunks to attempt an answer.
    # Calibrated on the evaluation set (see "similarity_gate" in evaluation/results_summary.md).
    similarity_threshold: float = Field(default=0.68, alias="SIMILARITY_THRESHOLD")

    # Server configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    # Comma-separated list of allowed browser origins ("*" = any; fine for a public read-only demo)
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    # If set, /ingest endpoints and diagnostics require header "X-Admin-Token: <value>"
    admin_token: str = Field(default="", alias="ADMIN_TOKEN")

    # Corpus path
    corpus_dir: str = Field(default=str(BASE_DIR / "data" / "corpus"), alias="CORPUS_DIR")

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key) and self.groq_api_key != "your_groq_api_key_here"

settings = Settings()
