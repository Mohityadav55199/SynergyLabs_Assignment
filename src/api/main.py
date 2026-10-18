import hmac
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.embeddings.embedder import get_embedder
from src.ingestion.pipeline import IngestionPipeline, IngestionReport, FileIngestResult
from src.generator.llm import GroundedRAGService, QueryResponse
from src.vector_stores import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Cost-Efficient RAG Service",
    description="QA service over PDF/HTML/Markdown backed by LanceDB (primary) or embedded Qdrant (benchmark).",
    version="1.1.0",
)

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,  # no cookies are used; wildcard origins + credentials is invalid anyway
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}

def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Protects write/diagnostic endpoints when ADMIN_TOKEN is set. Open when unset (local dev)."""
    if settings.admin_token and not hmac.compare_digest(x_admin_token or "", settings.admin_token):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Token header")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to answer",
                       examples=["What is HNSW and what recall does it provide?"])
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")
    store_type: Optional[Literal["lancedb", "qdrant"]] = Field(default=None, description="Backend (defaults to DEFAULT_VECTOR_STORE)")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None, description="Filter on format | doc_id | filename | section (exact match)",
        examples=[{"format": "pdf"}])
    similarity_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override SIMILARITY_THRESHOLD for this request")

class IngestDirectoryRequest(BaseModel):
    directory_path: Optional[str] = Field(default=None, description="Defaults to CORPUS_DIR")
    chunk_size: Optional[int] = Field(default=None, ge=50, le=8000, description="Characters per chunk (default DEFAULT_CHUNK_SIZE=500)")
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=2000, description="Overlap in characters (default DEFAULT_CHUNK_OVERLAP=50)")
    store_type: Optional[Literal["lancedb", "qdrant"]] = None

rag_service = GroundedRAGService()
ingest_pipeline = IngestionPipeline()

@app.get("/health", tags=["System"])
def health_check():
    embedder = get_embedder()
    store = get_vector_store(settings.default_vector_store)
    return {
        "status": "healthy",
        "default_vector_store": settings.default_vector_store,
        "vectors_indexed": store.count(),
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.embedding_dim,
        "generator_model": settings.groq_model,
        "llm_configured": settings.groq_configured,
        "similarity_threshold": settings.similarity_threshold,
    }

@app.get("/stats", tags=["System"])
def get_store_stats(store: Optional[Literal["lancedb", "qdrant"]] = None):
    return get_vector_store(store or settings.default_vector_store).get_stats()

@app.post("/ingest", response_model=IngestionReport, tags=["Ingestion"], dependencies=[Depends(require_admin)])
def ingest_directory(req: IngestDirectoryRequest):
    dir_path = Path(req.directory_path or settings.corpus_dir)
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    try:
        report = ingest_pipeline.ingest_directory(
            dir_path=dir_path, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap, store_type=req.store_type)
    except ValueError as exc:  # e.g. overlap >= chunk size
        raise HTTPException(status_code=422, detail=str(exc))
    logger.info("Ingestion: files=%d inserted=%d skipped=%d failed=%d in %sms",
                report.total_files, report.newly_inserted, report.skipped_duplicates,
                report.failed_files, report.total_time_ms)
    return report

@app.post("/ingest/upload", response_model=FileIngestResult, tags=["Ingestion"], dependencies=[Depends(require_admin)])
async def ingest_uploaded_file(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    store_type: Optional[Literal["lancedb", "qdrant"]] = Form(None),
):
    safe_name = Path(file.filename or "").name  # strip any path components (path traversal)
    if Path(safe_name).suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {sorted(SUPPORTED_UPLOAD_EXTENSIONS)}")

    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    upload_dir = Path(settings.corpus_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / safe_name
    dest.write_bytes(data)

    try:
        return ingest_pipeline.ingest_file(
            file_path=dest, chunk_size=chunk_size, chunk_overlap=chunk_overlap, store_type=store_type)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # corrupt PDF etc.
        dest.unlink(missing_ok=True)
        logger.exception("Upload ingestion failed for %s", safe_name)
        raise HTTPException(status_code=422, detail=f"Could not parse '{safe_name}': {type(exc).__name__}: {exc}")

@app.post("/query", response_model=QueryResponse, tags=["Retrieval & QA"])
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank")
    try:
        response = rag_service.query(
            query_text=req.query, top_k=req.top_k, store_type=req.store_type,
            metadata_filter=req.metadata_filter, similarity_threshold=req.similarity_threshold)
    except ValueError as exc:  # invalid filter etc.
        raise HTTPException(status_code=422, detail=str(exc))
    # Per-query structured log: latency, chunk count, token usage
    logger.info(
        "query store=%s k=%d mode=%s top_score=%.3f chunks=%d latency_ms(total=%s embed=%s search=%s gen=%s) "
        "tokens(prompt=%d completion=%d total=%d) citations=%d",
        response.store_type, req.top_k, response.generation_mode, response.top_score,
        response.retrieved_chunks_count, response.total_latency_ms, response.embedding_latency_ms,
        response.retrieval_latency_ms, response.generation_latency_ms,
        response.token_usage.get("prompt_tokens", 0), response.token_usage.get("completion_tokens", 0),
        response.token_usage.get("total_tokens", 0), len(response.citations))
    return response

@app.get("/llm-status", tags=["Diagnostics"], dependencies=[Depends(require_admin)])
def llm_status():
    """Checks the Groq key and that the configured generator model is reachable. Never returns key material."""
    if not settings.groq_configured:
        return {"configured": False, "error": "GROQ_API_KEY is not set"}
    try:
        from groq import Groq
        models = [m.id for m in Groq(api_key=settings.groq_api_key, timeout=15.0).models.list().data]
        return {"configured": True, "connected": True, "generator_model": settings.groq_model,
                "generator_model_available": settings.groq_model in models, "available_models": models}
    except Exception as exc:
        return {"configured": True, "connected": False, "error": f"{type(exc).__name__}: {exc}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.host, port=settings.port)
