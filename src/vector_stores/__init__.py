from typing import Optional, Literal
from src.config import settings
from src.vector_stores.base import BaseVectorStore, SearchResult
from src.vector_stores.lance_store import LanceDBVectorStore
from src.vector_stores.qdrant_store import QdrantVectorStore

_store_cache = {}

def get_vector_store(store_type: Optional[str] = None) -> BaseVectorStore:
    target_store = (store_type or settings.default_vector_store).lower()
    if target_store not in _store_cache:
        if target_store == "lancedb":
            _store_cache[target_store] = LanceDBVectorStore()
        elif target_store == "qdrant":
            _store_cache[target_store] = QdrantVectorStore()
        else:
            raise ValueError(f"Unknown vector store '{target_store}'. Choose 'lancedb' or 'qdrant'.")
    return _store_cache[target_store]

__all__ = ["BaseVectorStore", "SearchResult", "LanceDBVectorStore", "QdrantVectorStore", "get_vector_store"]
