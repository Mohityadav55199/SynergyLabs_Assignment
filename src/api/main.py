import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.embeddings.embedder import get_embedder
from src.ingestion.pipeline import IngestionPipeline, IngestionReport, FileIngestResult
from src.generator.llm import GroundedRAGService, QueryResponse
from src.vector_stores import get_vector_store

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Cost-Efficient RAG Service",
    description="Low-Cost QA Service backed by LanceDB & Qdrant Embedded with honest IR & Answer Evaluation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to answer", example="What is HNSW and how does it compare to IVF-PQ?")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of top chunks to retrieve", example=4)
    store_type: Optional[Literal["lancedb", "qdrant"]] = Field(
        default=None,
        description="Vector store backend to query (defaults to configured store in .env)"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter (e.g. {'format': 'pdf'} or {'doc_id': '...'})",
        example={"format": "pdf"}
    )
    similarity_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold to trigger answer generation"
    )

class IngestDirectoryRequest(BaseModel):
    directory_path: Optional[str] = Field(
        default=None,
        description="Path to documents directory (defaults to configured corpus_dir)"
    )
    chunk_size: Optional[int] = Field(default=500, description="Chunk size in characters")
    chunk_overlap: Optional[int] = Field(default=50, description="Chunk overlap in characters")
    store_type: Optional[Literal["lancedb", "qdrant"]] = Field(default=None)

rag_service = GroundedRAGService()
ingest_pipeline = IngestionPipeline()

@app.get("/health", tags=["System"])
def health_check():
    embedder = get_embedder()
    return {
        "status": "healthy",
        "default_vector_store": settings.default_vector_store,
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.embedding_dim,
        "groq_model": settings.groq_model,
        "groq_configured": bool(settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here")
    }

@app.get("/stats", tags=["System"])
def get_store_stats(store: Optional[Literal["lancedb", "qdrant"]] = None):
    target_store = get_vector_store(store or settings.default_vector_store)
    return target_store.get_stats()

@app.post("/ingest", response_model=IngestionReport, tags=["Ingestion"])
def ingest_directory(req: IngestDirectoryRequest):
    dir_path = Path(req.directory_path or settings.corpus_dir)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus directory not found: {dir_path}")

    logger.info(f"Starting directory ingestion from {dir_path} into store={req.store_type or settings.default_vector_store}")
    report = ingest_pipeline.ingest_directory(
        dir_path=dir_path,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        store_type=req.store_type
    )
    logger.info(f"Ingestion completed: {report.newly_inserted} chunks inserted, {report.skipped_duplicates} duplicates skipped in {report.total_time_ms}ms")
    return report

@app.post("/ingest/upload", response_model=FileIngestResult, tags=["Ingestion"])
async def ingest_uploaded_file(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(500),
    chunk_overlap: Optional[int] = Form(50),
    store_type: Optional[Literal["lancedb", "qdrant"]] = Form(None)
):
    upload_dir = Path(settings.corpus_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / file.filename

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    logger.info(f"Processing uploaded file: {file.filename}")
    result = ingest_pipeline.ingest_file(
        file_path=temp_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        store_type=store_type
    )
    return result

@app.post("/query", response_model=QueryResponse, tags=["Retrieval & QA"])
def query_rag(req: QueryRequest):
    logger.info(f"Received query: '{req.query}' | top_k={req.top_k} | store={req.store_type or settings.default_vector_store}")
    response = rag_service.query(
        query_text=req.query,
        top_k=req.top_k,
        store_type=req.store_type,
        metadata_filter=req.metadata_filter,
        similarity_threshold=req.similarity_threshold
    )
    # Structured per-query logging as required
    logger.info(
        f"Query processed: latency={response.total_latency_ms}ms (retrieval={response.retrieval_latency_ms}ms, "
        f"gen={response.generation_latency_ms}ms) | chunks={response.retrieved_chunks_count} | "
        f"tokens={response.token_usage.get('total_tokens', 0)} | citations={len(response.citations)}"
    )
@app.get("/groq-debug", tags=["Diagnostics"])
def groq_debug():
    key = settings.groq_api_key
    if not key or key == "your_groq_api_key_here":
        return {
            "configured": False,
            "key_preview": "None / Placeholder",
            "error": "GROQ_API_KEY is not set or still set to 'your_groq_api_key_here'."
        }

    preview = f"{key[:7]}...{key[-4:]}" if len(key) > 11 else "Invalid key length"
    try:
        from groq import Groq
        client = Groq(api_key=key)
        model_list = [m.id for m in client.models.list().data]
        return {
            "configured": True,
            "key_preview": preview,
            "connected": True,
            "models_count": len(model_list),
            "accessible_models": model_list
        }
    except Exception as e:
        return {
            "configured": True,
            "key_preview": preview,
            "connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port, reload=True)
