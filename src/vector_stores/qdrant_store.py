from __future__ import annotations
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)
from src.config import settings
from src.vector_stores.base import BaseVectorStore, SearchResult

if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk

class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant Embedded implementation of BaseVectorStore.
    Runs in-process, storing vector indices directly on disk without a network daemon.
    Serves as the secondary benchmark store.
    """
    COLLECTION_NAME = "synergy_rag_chunks"

    def __init__(self, db_dir: Optional[str] = None):
        self.db_dir = Path(db_dir or settings.qdrant_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.db_dir))
        self.initialize()

    def initialize(self) -> None:
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    def get_existing_hashes(self) -> Set[str]:
        if not self.client.collection_exists(self.COLLECTION_NAME):
            return set()
        existing_hashes = set()
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["chunk_hash"],
                with_vectors=False
            )
            for r in records:
                if r.payload and "chunk_hash" in r.payload:
                    existing_hashes.add(r.payload["chunk_hash"])
            if next_offset is None or not records:
                break
            offset = next_offset
        return existing_hashes

    def insert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> int:
        if not chunks:
            return 0

        existing_hashes = self.get_existing_hashes()
        points: List[PointStruct] = []

        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_hash in existing_hashes:
                continue  # Idempotent skip: prevent duplicate vectors

            # Deterministic UUID from chunk_hash
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_hash))

            payload = {
                "chunk_id": chunk.chunk_id,
                "chunk_hash": chunk.chunk_hash,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "format": chunk.format,
                "section": chunk.section,
                "text": chunk.text,
                **chunk.metadata
            }

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
            existing_hashes.add(chunk.chunk_hash)

        if not points:
            return 0

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
            wait=True
        )
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if not self.client.collection_exists(self.COLLECTION_NAME):
            return []

        query_filter = None
        if metadata_filter:
            must_conditions = []
            for key, val in metadata_filter.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=val))
                )
            if must_conditions:
                query_filter = Filter(must=must_conditions)

        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        results: List[SearchResult] = []
        for p in response.points:
            payload = p.payload or {}
            # Score from Qdrant with Cosine distance is cosine similarity in range [-1, 1]
            score = max(0.0, float(p.score)) if p.score is not None else 0.0

            results.append(SearchResult(
                chunk_id=payload.get("chunk_id", str(p.id)),
                doc_id=payload.get("doc_id", ""),
                filename=payload.get("filename", ""),
                format=payload.get("format", ""),
                section=payload.get("section", ""),
                text=payload.get("text", ""),
                score=round(score, 4),
                metadata=payload
            ))

        return results

    def get_stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.COLLECTION_NAME)
        total_size = sum(f.stat().st_size for f in self.db_dir.glob("**/*") if f.is_file())

        return {
            "store_type": "qdrant",
            "mode": "embedded_local",
            "storage_path": str(self.db_dir),
            "total_chunks": info.points_count or 0,
            "embedding_dim": settings.embedding_dim,
            "disk_size_bytes": total_size,
            "disk_size_mb": round(total_size / (1024 * 1024), 3)
        }

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.initialize()
