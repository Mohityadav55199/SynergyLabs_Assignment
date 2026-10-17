from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
import lancedb
import pyarrow as pa
from src.config import settings
from src.vector_stores.base import BaseVectorStore, SearchResult

if TYPE_CHECKING:
    from src.ingestion.chunker import Chunk

class LanceDBVectorStore(BaseVectorStore):
    """
    LanceDB implementation of BaseVectorStore.
    Runs locally/embedded using Apache Arrow/Parquet columnar format.
    Zero always-on infrastructure cost, fast vector scan, and metadata filtering.
    """
    TABLE_NAME = "chunks"

    def __init__(self, db_dir: Optional[str] = None):
        self.db_dir = Path(db_dir or settings.lancedb_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_dir))
        self.table = None
        self.initialize()

    def _get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("vector", pa.list_(pa.float32(), settings.embedding_dim)),
            pa.field("chunk_id", pa.string()),
            pa.field("chunk_hash", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("format", pa.string()),
            pa.field("section", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata_json", pa.string())
        ])

    def initialize(self) -> None:
        try:
            self.table = self.db.open_table(self.TABLE_NAME)
        except Exception:
            self.table = self.db.create_table(
                self.TABLE_NAME,
                schema=self._get_schema(),
                mode="create"
            )

    def get_existing_hashes(self) -> Set[str]:
        if self.table is None or self.table.count_rows() == 0:
            return set()
        try:
            arrow_tbl = self.table.to_arrow()
            if "chunk_hash" in arrow_tbl.column_names:
                return set(arrow_tbl["chunk_hash"].to_pylist())
            return set()
        except Exception as e:
            return set()

    def insert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> int:
        if not chunks:
            return 0

        existing_hashes = self.get_existing_hashes()
        new_records = []

        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_hash in existing_hashes:
                continue  # Idempotent skip: prevent duplicate vectors

            new_records.append({
                "vector": vector,
                "chunk_id": chunk.chunk_id,
                "chunk_hash": chunk.chunk_hash,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "format": chunk.format,
                "section": chunk.section,
                "text": chunk.text,
                "metadata_json": json.dumps(chunk.metadata)
            })
            existing_hashes.add(chunk.chunk_hash)

        if not new_records:
            return 0

        self.table.add(new_records)
        return len(new_records)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.table is None or self.table.count_rows() == 0:
            return []

        query = self.table.search(query_vector).metric("cosine").limit(top_k)

        # Apply metadata filter (e.g. format = 'pdf', doc_id = '...')
        if metadata_filter:
            filter_clauses = []
            for key, val in metadata_filter.items():
                if isinstance(val, str):
                    clean_val = val.replace("'", "''")
                    filter_clauses.append(f"{key} = '{clean_val}'")
                elif isinstance(val, (int, float)):
                    filter_clauses.append(f"{key} = {val}")
                elif isinstance(val, bool):
                    filter_clauses.append(f"{key} = {str(val).lower()}")
            if filter_clauses:
                where_clause = " AND ".join(filter_clauses)
                query = query.where(where_clause)

        results = query.to_list()
        search_results: List[SearchResult] = []

        for r in results:
            # Cosine distance to similarity conversion: similarity = 1 - distance
            distance = r.get("_distance", 0.0)
            score = max(0.0, 1.0 - distance)

            meta = {}
            if "metadata_json" in r and r["metadata_json"]:
                try:
                    meta = json.loads(r["metadata_json"])
                except Exception:
                    pass

            search_results.append(SearchResult(
                chunk_id=r.get("chunk_id", ""),
                doc_id=r.get("doc_id", ""),
                filename=r.get("filename", ""),
                format=r.get("format", ""),
                section=r.get("section", ""),
                text=r.get("text", ""),
                score=round(score, 4),
                metadata=meta
            ))

        return search_results

    def get_stats(self) -> Dict[str, Any]:
        count = self.table.count_rows() if self.table else 0
        total_size = sum(f.stat().st_size for f in self.db_dir.glob("**/*") if f.is_file())

        return {
            "store_type": "lancedb",
            "mode": "embedded_columnar",
            "storage_path": str(self.db_dir),
            "total_chunks": count,
            "embedding_dim": settings.embedding_dim,
            "disk_size_bytes": total_size,
            "disk_size_mb": round(total_size / (1024 * 1024), 3)
        }

    def clear(self) -> None:
        try:
            self.db.drop_table(self.TABLE_NAME)
        except Exception:
            pass
        self.initialize()
