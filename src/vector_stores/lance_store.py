from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
import lancedb
import pyarrow as pa
from src.config import settings
from src.vector_stores.base import BaseVectorStore, SearchResult, validate_metadata_filter

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
        # Errors propagate on purpose: returning an empty set on failure would silently
        # re-insert every chunk and break idempotency.
        if self.table.count_rows() == 0:
            return set()
        rows = self.table.search().select(["chunk_hash"]).limit(self.table.count_rows()).to_arrow()
        return set(rows["chunk_hash"].to_pylist())

    def count(self) -> int:
        return self.table.count_rows()

    def list_documents(self) -> List[Dict[str, Any]]:
        n = self.table.count_rows()
        if n == 0:
            return []
        rows = self.table.search().select(["filename", "format"]).limit(n).to_arrow().to_pylist()
        counts: Dict[tuple, int] = {}
        for r in rows:
            key = (r["filename"], r["format"])
            counts[key] = counts.get(key, 0) + 1
        return [{"filename": f, "format": fmt, "chunks": c} for (f, fmt), c in sorted(counts.items())]

    def delete_stale_versions(self, filename: str, keep_doc_id: str) -> int:
        before = self.table.count_rows()
        fn = filename.replace("'", "''")
        did = keep_doc_id.replace("'", "''")
        self.table.delete(f"filename = '{fn}' AND doc_id != '{did}'")
        return before - self.table.count_rows()

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
        clean_filter = validate_metadata_filter(metadata_filter)  # validate even when the table is empty
        if self.table is None or self.table.count_rows() == 0:
            return []

        query = self.table.search(query_vector).metric("cosine").limit(top_k)

        if clean_filter:
            where_clause = " AND ".join(
                f"{key} = '{val.replace(chr(39), chr(39) * 2)}'" for key, val in clean_filter.items()
            )
            query = query.where(where_clause, prefilter=True)

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
