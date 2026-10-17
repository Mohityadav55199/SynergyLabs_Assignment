# Low-Cost Vector Indexing and Infrastructure Cost Analysis

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
