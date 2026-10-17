import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.config import settings
from src.embeddings.embedder import get_embedder
from src.ingestion.parsers import DocumentParser, ParsedDocument
from src.ingestion.chunker import DocumentChunker, Chunk
from src.vector_stores import get_vector_store, BaseVectorStore

class FileIngestResult(BaseModel):
    filename: str
    format: str
    doc_id: str
    total_chunks: int
    inserted_chunks: int
    skipped_duplicates: int
    time_taken_ms: float

class IngestionReport(BaseModel):
    total_files: int
    total_chunks: int
    newly_inserted: int
    skipped_duplicates: int
    store_type: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dim: int
    total_time_ms: float
    file_results: List[FileIngestResult]

class IngestionPipeline:
    """
    End-to-end multi-format ingestion pipeline supporting PDF, HTML, and Markdown.
    Guarantees idempotency (zero duplicate vectors) via SHA-256 fingerprinting.
    """
    def __init__(self, store: Optional[BaseVectorStore] = None):
        self.store = store
        self.embedder = get_embedder()

    def ingest_file(
        self,
        file_path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        store_type: Optional[str] = None
    ) -> FileIngestResult:
        start_time = time.perf_counter()
        target_store = self.store or get_vector_store(store_type)

        c_size = chunk_size or settings.default_chunk_size
        c_overlap = chunk_overlap or settings.default_chunk_overlap
        chunker = DocumentChunker(chunk_size=c_size, chunk_overlap=c_overlap)

        # 1. Parse document
        parsed: ParsedDocument = DocumentParser.parse_file(file_path)

        # 2. Chunk document with metadata
        chunks: List[Chunk] = chunker.chunk_document(
            doc_id=parsed.doc_id,
            filename=parsed.filename,
            doc_format=parsed.format,
            text=parsed.text,
            base_metadata=parsed.metadata
        )

        # 3. Check for existing hashes to prevent duplicate embeddings and vectors
        existing_hashes = target_store.get_existing_hashes()
        new_chunks = [c for c in chunks if c.chunk_hash not in existing_hashes]
        skipped_count = len(chunks) - len(new_chunks)

        inserted_count = 0
        if new_chunks:
            # 4. Generate embeddings only for new chunks
            chunk_texts = [c.text for c in new_chunks]
            vectors = self.embedder.embed_documents(chunk_texts)

            # 5. Insert into vector store
            inserted_count = target_store.insert_chunks(new_chunks, vectors)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FileIngestResult(
            filename=file_path.name,
            format=parsed.format,
            doc_id=parsed.doc_id,
            total_chunks=len(chunks),
            inserted_chunks=inserted_count,
            skipped_duplicates=skipped_count,
            time_taken_ms=round(elapsed_ms, 2)
        )

    def ingest_directory(
        self,
        dir_path: Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        store_type: Optional[str] = None
    ) -> IngestionReport:
        start_time = time.perf_counter()
        target_store_type = store_type or settings.default_vector_store
        supported_extensions = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}

        files = [
            f for f in Path(dir_path).rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        file_results: List[FileIngestResult] = []
        total_chunks = 0
        total_inserted = 0
        total_skipped = 0

        for file_path in files:
            res = self.ingest_file(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                store_type=target_store_type
            )
            file_results.append(res)
            total_chunks += res.total_chunks
            total_inserted += res.inserted_chunks
            total_skipped += res.skipped_duplicates

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return IngestionReport(
            total_files=len(files),
            total_chunks=total_chunks,
            newly_inserted=total_inserted,
            skipped_duplicates=total_skipped,
            store_type=target_store_type,
            chunk_size=chunk_size or settings.default_chunk_size,
            chunk_overlap=chunk_overlap or settings.default_chunk_overlap,
            embedding_model=self.embedder.model_name,
            embedding_dim=self.embedder.embedding_dim,
            total_time_ms=round(total_elapsed_ms, 2),
            file_results=file_results
        )
