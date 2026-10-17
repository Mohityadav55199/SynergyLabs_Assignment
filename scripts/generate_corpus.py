import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

def generate_pdf():
    pdf_path = CORPUS_DIR / "vector_db_architecture.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor("#1e293b"))
    h2_style = ParagraphStyle('DocH2', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor("#0f766e"), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=9.5, leading=13.5, textColor=colors.HexColor("#334155"))

    elements = []

    elements.append(Paragraph("Vector Database Architecture & Indexing Fundamentals", title_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f766e"), spaceAfter=12))

    elements.append(Paragraph("Section 1: HNSW vs IVF-PQ Vector Indexing Mechanics", h2_style))
    elements.append(Paragraph(
        "Approximate Nearest Neighbor (ANN) search algorithms navigate a trade-off between recall accuracy, query latency, and index size. "
        "Hierarchical Navigable Small World (HNSW) is a graph-based index structure that provides state-of-the-art recall (>98%) and sub-millisecond query latency. "
        "HNSW builds a multi-layer graph where upper layers contain sparse long-distance skip links for rapid routing, and the bottom layer contains dense local connections. "
        "However, HNSW graphs require holding both vector vectors and graph edges in memory (RAM), requiring roughly 1.5 to 2.0 KB per 1536-dimensional vector. "
        "Conversely, Inverted File with Product Quantization (IVF-PQ) partitions the vector space into Voronoi cells and compresses vectors into compact byte codes. "
        "IVF-PQ reduces memory footprint by up to 95% compared to raw HNSW, but incurs an initial index training phase and suffers lower recall on out-of-distribution queries.",
        body_style
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Section 2: LanceDB Columnar Architecture and Zero-Pod Economics", h2_style))
    elements.append(Paragraph(
        "LanceDB is built on top of Lance, a modern columnar data format tailored for machine learning and multimodal data inspired by Apache Arrow and Parquet. "
        "Unlike traditional vector databases that run persistent container pods with gigabytes of allocated RAM (such as Pinecone Standard or Weaviate clusters), "
        "LanceDB is fundamentally serverless and disk-native. It executes vector distance calculations directly on NVMe SSDs or cloud object storage like AWS S3 or Google Cloud Storage. "
        "Lance organizes data into columnar batches with fast dictionary encoding, zero-copy reads via Arrow, and inverted vector indices stored directly inside the dataset files. "
        "Because there is no background daemon or always-on pod requirement, an idle LanceDB index incurs zero compute cost, making it the most cost-efficient choice for lightly-queried corpora.",
        body_style
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Section 3: Qdrant Vector Engine Architecture and Payload Filtering", h2_style))
    elements.append(Paragraph(
        "Qdrant is an open-source vector search engine written in Rust that specializes in real-time vector search with comprehensive payload filtering. "
        "Qdrant combines modified HNSW graphs with memory-mapped files (mmap) and on-disk payload storage. "
        "A key architectural advantage of Qdrant is its single-stage filtered search: instead of pre-filtering (filtering before vector search, which destroys HNSW graph connectivity) "
        "or post-filtering (filtering top-k results after vector search, which risks returning fewer than k items), Qdrant evaluates payload filters during the HNSW graph traversal. "
        "In embedded mode, Qdrant runs inside the application process without network latency, persisting index files and payloads to a local directory.",
        body_style
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Section 4: Memory vs Disk Trade-offs across Vector Scales", h2_style))
    elements.append(Paragraph(
        "At 100K vectors with 384 dimensions, raw vector data occupies approximately 150 MB. Both RAM-based and disk-based engines deliver sub-5ms retrieval. "
        "However, at 10M vectors, raw embeddings consume 15.36 GB, and HNSW indices expand the memory footprint to over 25 GB. "
        "On managed cloud providers, maintaining a 32GB RAM pod costs between $70 and $250 per month even if query traffic is zero. "
        "By contrast, storing 10M vectors on disk or S3 with LanceDB costs less than $0.50 per month in storage, with query latencies staying under 15ms on modern NVMe drives.",
        body_style
    ))

    doc.build(elements)
    print(f"Generated PDF: {pdf_path}")

def generate_html():
    html_path = CORPUS_DIR / "rag_chunking_guide.html"
    content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RAG Document Chunking and Context Optimization Guide</title>
</head>
<body>
    <h1>RAG Document Chunking and Context Optimization Guide</h1>
    
    <h2>Section 1: Chunking Granularity and Window Strategies</h2>
    <p>
        Document chunking is the process of breaking large text corpora into discrete, semantically coherent segments suitable for vector embedding and LLM context injection.
        The choice of chunk size directly impacts both retrieval precision and generation groundedness.
        Small chunks (100 to 300 characters) offer high semantic precision: the embedding represents a single concept with minimal noise, making it easy to achieve high cosine similarity.
        However, small chunks often lose surrounding context, leading to incomplete answers during generation.
        Large chunks (800 to 2000 characters) preserve rich surrounding context and multi-sentence arguments, but dilute the vector representation because the embedding model must compress multiple disparate ideas into a single fixed-dimensional vector.
        For technical and engineering documents, a chunk size of 400 to 600 characters strikes the empirical sweet spot between semantic focus and contextual completeness.
    </p>

    <h2>Section 2: Chunk Overlap Mechanics and Boundary Preservation</h2>
    <p>
        Chunk overlap (typically 10% to 20% of the chunk size) prevents critical information from being split across chunk boundaries.
        When a factual statement spans two paragraphs, splitting strictly at fixed character counts can cut a sentence in half, causing both chunks to fail vector matching.
        A default overlap of 50 characters ensures that boundary tokens are duplicated in adjacent chunks, preserving semantic continuity without causing excessive vector index bloat.
        Idempotency is preserved by calculating SHA-256 hashes on the composite tuple of (doc_id, chunk_index, chunk_text), ensuring that repeated re-ingestion does not pollute the vector store with duplicate entries.
    </p>

    <h2>Section 3: Context Window Sizing and Anti-Hallucination Guardrails</h2>
    <p>
        Injecting irrelevant chunks into the LLM context prompt significantly increases the risk of hallucination and 'lost-in-the-middle' attention degradation.
        Top-k retrieval should be constrained (typically k=3 to 5) to only the most relevant passages.
        Furthermore, production RAG services must enforce a strict similarity threshold (e.g. cosine similarity >= 0.25).
        When the top retrieved chunk fails to meet this threshold, the system must trigger an explicit fallback response: 'I do not have sufficient information in the provided context to answer this question.'
        This deterministic fallback stops the generator from fabricating plausible-sounding hallucinations when user queries are out-of-domain.
    </p>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated HTML: {html_path}")

def generate_markdown():
    md_path = CORPUS_DIR / "low_cost_indexing_strategies.md"
    content = """# Low-Cost Vector Indexing and Infrastructure Cost Analysis

## Section 1: The Economics of Managed vs Low-Cost Vector Stores
Fully managed vector database services (such as Pinecone Standard, Qdrant Cloud Dedicated, and Databricks Vector Search) charge customers primarily for allocated cluster compute and persistent RAM pods.
In an enterprise setting where a knowledge base contains 1M to 10M vectors but receives sporadic internal queries (e.g. 50 to 500 queries per day), more than 98% of the monthly invoice pays for idle compute and reserved RAM rather than active query execution.
By deploying low-cost embedded or serverless vector stores such as LanceDB or Qdrant Embedded on local NVMe or cloud object storage (S3/GCS), organizations decouple storage costs from compute costs.

## Section 2: Comprehensive Monthly Cost Model Across Vector Scales
Below is the empirical cost comparison between a managed pod-based vector database and a low-cost embedded/disk store:

| Vector Scale | Managed Pods (Pinecone/Qdrant Cloud) | Low-Cost Store (LanceDB / Local NVMe) | S3-Backed LanceDB | Monthly Savings |
| :--- | :--- | :--- | :--- | :--- |
| **100K Vectors** (384-dim, ~150 MB) | $70.00 / month (1 x starter pod) | $0.00 (embedded on host) | $0.01 / month (S3 standard) | **~100% savings** |
| **1M Vectors** (384-dim, ~1.5 GB) | $96.00 / month (1 x 4GB pod) | $0.20 / month (EBS/NVMe disk) | $0.04 / month (S3 standard) | **99.8% savings** |
| **10M Vectors** (384-dim, ~15.4 GB) | $288.00 / month (2 x 16GB pods) | $2.00 / month (100GB gp3 disk) | $0.35 / month (S3 standard) | **99.3% savings** |

### Assumptions in Cost Model:
1. Embeddings: 384 dimensions (float32 = 1,536 bytes per vector without index overhead).
2. Managed DB baseline: Pinecone standard pod pricing ($0.096/hour for s1/p1 pods) or Qdrant Cloud standard cluster.
3. Storage pricing: AWS S3 Standard at $0.023/GB-month or AWS EBS gp3 at $0.08/GB-month.
4. Compute: Co-located on existing application server (FastAPI container), adding negligible marginal CPU overhead.

## Section 3: Trade-Off Analysis and When to Switch Back to Managed
While low-cost embedded stores like LanceDB and Qdrant Embedded provide orders of magnitude cost reductions, they introduce specific trade-offs:
1. Concurrent Write Contention: Embedded stores rely on file locks. High-throughput distributed write workloads (>100 concurrent writers) require central coordination or a managed cluster.
2. Cold Starts over Object Storage: Querying LanceDB directly over S3 introduces a 50-100ms first-query latency while downloading metadata chunks, whereas always-on pods have indices pre-warmed in RAM.
3. High QPS Real-Time SLA: When query throughput exceeds 1,000 queries per second (QPS) with strict sub-5ms p99 SLAs, dedicated memory-resident vector clusters become necessary.

**Recommendation:** For document QA, customer support knowledge bases, internal enterprise search, and read-heavy RAG pipelines with <100 QPS, low-cost stores like LanceDB and Qdrant Embedded deliver identical retrieval accuracy at a fraction of 1% of the infrastructure cost.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated Markdown: {md_path}")

if __name__ == "__main__":
    generate_pdf()
    generate_html()
    generate_markdown()
    print("Corpus generation complete.")
