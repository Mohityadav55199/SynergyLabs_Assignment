from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk

# Only these fields may be used in a metadata filter. This is a whitelist because the
# LanceDB backend builds a SQL-like WHERE clause from the keys.
ALLOWED_FILTER_FIELDS = ("format", "doc_id", "filename", "section")

def validate_metadata_filter(metadata_filter: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Validates a metadata filter and returns it with string values. Raises ValueError if invalid."""
    if not metadata_filter:
        return {}
    clean: Dict[str, str] = {}
    for key, val in metadata_filter.items():
        if key not in ALLOWED_FILTER_FIELDS:
            raise ValueError(f"Unsupported filter field '{key}'. Allowed: {', '.join(ALLOWED_FILTER_FIELDS)}")
        if not isinstance(val, str) or not val:
            raise ValueError(f"Filter value for '{key}' must be a non-empty string")
        clean[key] = val
    return clean

class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    format: str
    section: str
    text: str
    score: float
    metadata: Dict[str, Any]

class BaseVectorStore(ABC):
    """
    Abstract Vector Store interface ensuring consistent behavior across
    LanceDB, Qdrant Embedded, and any benchmarked backend.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize connection, directories, and tables/collections."""

    @abstractmethod
    def insert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> int:
        """
        Inserts new chunks and their corresponding embeddings.
        Returns the count of newly inserted chunks (must be idempotent).
        """

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Top-k cosine-similarity search with an optional metadata filter."""

    @abstractmethod
    def get_existing_hashes(self) -> Set[str]:
        """All chunk_hash values currently stored (used to guarantee idempotency)."""

    @abstractmethod
    def delete_stale_versions(self, filename: str, keep_doc_id: str) -> int:
        """
        Deletes vectors of `filename` whose doc_id differs from `keep_doc_id`
        (i.e. an older version of an edited file). Returns number of vectors deleted.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of vectors stored."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Total vectors, disk footprint, model metadata, etc."""

    @abstractmethod
    def clear(self) -> None:
        """Resets the vector collection."""
