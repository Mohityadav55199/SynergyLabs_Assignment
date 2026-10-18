"""Retrieval latency & disk footprint at 100K / 1M vectors (synthetic, unit-normalised 384-d, random).

The 100-chunk corpus is far too small to say anything about latency at scale, so this script builds
synthetic indexes and measures top-k search (vector search only, query embedding excluded).
Caveat: random vectors are a worst case for ANN recall and say nothing about recall on real text; this
benchmark measures speed and size only. LanceDB is measured brute-force (flat) and with IVF_PQ.

  python -m evaluation.scale_benchmark --sizes 100000 1000000 --out evaluation/results_scale.json
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIM, K = 384, 4
N_QUERIES = 100


def vectors(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, DIM), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dir_mb(p: Path) -> float:
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)


def timed_queries(fn, queries) -> dict:
    for q in queries[:5]:
        fn(q)
    lat = []
    for q in queries:
        t = time.perf_counter()
        fn(q)
        lat.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2)}


def bench_lance(n: int, tmp: Path) -> dict:
    import lancedb
    db = lancedb.connect(str(tmp / f"lance_{n}"))
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), DIM)), pa.field("id", pa.int64()),
                        pa.field("format", pa.string())])
    tbl = db.create_table("t", schema=schema)
    t0, batch = time.perf_counter(), 100_000
    for s in range(0, n, batch):
        m = min(batch, n - s)
        v = vectors(m, s)
        tbl.add(pa.table({"vector": pa.FixedSizeListArray.from_arrays(pa.array(v.ravel()), DIM),
                          "id": pa.array(np.arange(s, s + m)), "format": pa.array(["pdf" if i % 3 == 0 else "md" for i in range(m)])},
                         schema=schema))
    load_s = time.perf_counter() - t0
    queries = vectors(N_QUERIES, 999)
    out = {"load_seconds": round(load_s, 1)}
    out["flat_scan"] = timed_queries(lambda q: tbl.search(q).metric("cosine").limit(K).to_list(), queries)
    out["flat_scan_filtered"] = timed_queries(
        lambda q: tbl.search(q).metric("cosine").where("format = 'pdf'", prefilter=True).limit(K).to_list(), queries)
    out["disk_mb_flat"] = dir_mb(tmp / f"lance_{n}")
    if n >= 100_000:
        t1 = time.perf_counter()
        tbl.create_index(metric="cosine", index_type="IVF_PQ", num_partitions=max(8, int(n ** 0.5) // 2),
                         num_sub_vectors=48)  # deprecated-kwargs API still works in lancedb 0.40
        out["ivf_pq_build_seconds"] = round(time.perf_counter() - t1, 1)
        out["ivf_pq"] = timed_queries(lambda q: tbl.search(q).metric("cosine").nprobes(20).limit(K).to_list(), queries)
        out["disk_mb_with_index"] = dir_mb(tmp / f"lance_{n}")
    return out


def bench_qdrant(n: int, tmp: Path) -> dict:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    c = QdrantClient(path=str(tmp / f"qdrant_{n}"))
    c.create_collection("t", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    t0, batch = time.perf_counter(), 10_000
    for s in range(0, n, batch):
        v = vectors(min(batch, n - s), s)
        c.upsert("t", points=[PointStruct(id=s + i, vector=v[i].tolist()) for i in range(len(v))], wait=True)
    out = {"load_seconds": round(time.perf_counter() - t0, 1)}
    queries = vectors(N_QUERIES, 999)
    out["search"] = timed_queries(lambda q: c.query_points("t", query=q.tolist(), limit=K), queries)
    out["disk_mb"] = dir_mb(tmp / f"qdrant_{n}")
    c.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    ap.add_argument("--qdrant-n", type=int, default=20_000, help="Qdrant local mode is brute-force numpy and slow to load; measured at this size only")
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results_scale.json")
    ap.add_argument("--queries", type=int, default=100)
    args = ap.parse_args()
    global N_QUERIES
    N_QUERIES = args.queries
    tmp = Path(tempfile.mkdtemp(prefix="scale_bench_"))
    results = {"dim": DIM, "k": K, "queries": args.queries, "note": "synthetic random unit vectors; speed/size only"}
    try:
        for n in args.sizes:
            print(f"== {n:,} vectors: LanceDB"); results[f"lancedb_{n}"] = bench_lance(n, tmp); print(results[f"lancedb_{n}"])
            args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if args.qdrant_n:
            print(f"== {args.qdrant_n:,} vectors: Qdrant (local mode)")
            results[f"qdrant_{args.qdrant_n}"] = bench_qdrant(args.qdrant_n, tmp)
            print(results[f"qdrant_{args.qdrant_n}"])
            args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
