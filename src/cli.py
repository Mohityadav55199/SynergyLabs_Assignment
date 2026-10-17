import argparse
import json
import sys
from pathlib import Path
from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.generator.llm import GroundedRAGService
from src.vector_stores import get_vector_store

def handle_ingest(args):
    pipeline = IngestionPipeline()
    corpus_dir = Path(args.dir or settings.corpus_dir)
    print(f"[*] Ingesting documents from: {corpus_dir}")
    print(f"[*] Target Store: {args.store}")
    print(f"[*] Chunk Size: {args.chunk_size} | Overlap: {args.overlap}")

    report = pipeline.ingest_directory(
        dir_path=corpus_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        store_type=args.store
    )

    print("\n" + "=" * 50)
    print("INGESTION REPORT")
    print("=" * 50)
    print(f"Total files processed : {report.total_files}")
    print(f"Total chunks generated: {report.total_chunks}")
    print(f"Newly inserted chunks : {report.newly_inserted}")
    print(f"Duplicates skipped    : {report.skipped_duplicates} (Idempotent)")
    print(f"Vector Store Backend  : {report.store_type}")
    print(f"Embedding Model       : {report.embedding_model} ({report.embedding_dim}-dim)")
    print(f"Total time elapsed    : {report.total_time_ms} ms")
    print("=" * 50)

def handle_query(args):
    rag = GroundedRAGService(store_type=args.store)
    metadata_filter = None
    if args.filter:
        try:
            metadata_filter = json.loads(args.filter)
        except Exception:
            # support simple key=value syntax
            if "=" in args.filter:
                k, v = args.filter.split("=", 1)
                metadata_filter = {k.strip(): v.strip()}

    print(f"\n[*] Querying [{args.store}]: \"{args.query}\" (top_k={args.k})")
    response = rag.query(
        query_text=args.query,
        top_k=args.k,
        store_type=args.store,
        metadata_filter=metadata_filter
    )

    print("\n" + "=" * 60)
    print("GROUNDED ANSWER:")
    print("=" * 60)
    print(response.answer)
    print("=" * 60)
    print("CITATIONS & CHUNKS USED:")
    if response.citations:
        for c in response.citations:
            print(f"  * Chunk ID: {c.chunk_id} | File: {c.filename} | Section: {c.section} | Sim: {c.score}")
    else:
        print("  (No relevant context / citations)")

    print("-" * 60)
    print("PERFORMANCE TELEMETRY:")
    print(f"  Retrieval Latency : {response.retrieval_latency_ms} ms")
    print(f"  Generation Latency: {response.generation_latency_ms} ms")
    print(f"  Total Latency     : {response.total_latency_ms} ms")
    print(f"  Retrieved Chunks  : {response.retrieved_chunks_count}")
    print(f"  Token Usage       : {response.token_usage}")
    print("=" * 60 + "\n")

def handle_stats(args):
    store = get_vector_store(args.store)
    stats = store.get_stats()
    print("\n" + "=" * 40)
    print(f"VECTOR STORE STATS ({args.store.upper()})")
    print("=" * 40)
    for k, v in stats.items():
        print(f"  {k:20}: {v}")
    print("=" * 40 + "\n")

def handle_serve(args):
    import uvicorn
    print(f"[*] Starting FastAPI RAG Server on {args.host}:{args.port}...")
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=True)

def main():
    parser = argparse.ArgumentParser(description="Cost-Efficient RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest corpus into vector store")
    ingest_parser.add_argument("--dir", type=str, default=None, help="Directory containing documents")
    ingest_parser.add_argument("--store", type=str, default="lancedb", choices=["lancedb", "qdrant"])
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters (default: 500)")
    ingest_parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters (default: 50)")
    ingest_parser.set_defaults(func=handle_ingest)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG service")
    query_parser.add_argument("query", type=str, help="Search query or question")
    query_parser.add_argument("--store", type=str, default="lancedb", choices=["lancedb", "qdrant"])
    query_parser.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve")
    query_parser.add_argument("--filter", type=str, default=None, help="Metadata filter (JSON string or key=val)")
    query_parser.set_defaults(func=handle_query)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="View vector store stats")
    stats_parser.add_argument("--store", type=str, default="lancedb", choices=["lancedb", "qdrant"])
    stats_parser.set_defaults(func=handle_stats)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI server")
    serve_parser.add_argument("--host", type=str, default=settings.host)
    serve_parser.add_argument("--port", type=int, default=settings.port)
    serve_parser.set_defaults(func=handle_serve)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
