from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk

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
        pass

    @abstractmethod
    def insert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> int:
        """
        Inserts new chunks and their corresponding embeddings.
        Returns the count of newly inserted chunks (must be idempotent).
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform top-k vector similarity search with optional metadata filter.
        """
        pass

    @abstractmethod
    def get_existing_hashes(self) -> Set[str]:
        """
        Retrieve all chunk_hash values currently in the store to guarantee idempotency.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Returns store statistics: total vectors, disk footprint, model metadata, etc.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Resets the vector collection."""
        pass
