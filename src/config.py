import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq LLM Configuration
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Vector Store Selection
    default_vector_store: Literal["lancedb", "qdrant"] = Field(default="lancedb", alias="DEFAULT_VECTOR_STORE")
    lancedb_dir: str = Field(default=str(BASE_DIR / "data" / "lancedb"), alias="LANCEDB_DIR")
    qdrant_dir: str = Field(default=str(BASE_DIR / "data" / "qdrant"), alias="QDRANT_DIR")

    # Embedding Configuration
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")

    # Ingestion Defaults
    default_chunk_size: int = Field(default=500, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=50, alias="DEFAULT_CHUNK_OVERLAP")

    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Corpus & Data paths
    corpus_dir: str = Field(default=str(BASE_DIR / "data" / "corpus"))

settings = Settings()
