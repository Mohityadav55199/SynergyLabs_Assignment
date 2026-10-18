"""Retrieval quality vs chunk size / overlap (retrieval layer only, no LLM). Answers "what degraded at other settings?".

  python -m evaluation.chunking_ablation            # writes evaluation/results_chunking_ablation.json + prints a table

Each setting builds a throw-away LanceDB index in a temp dir; relevance labels are recomputed against that setting's
chunks (same `evidence` rule as the main harness), so numbers are comparable across settings.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.ground_truth import build_corpus_chunks, relevant_chunk_ids  # noqa: E402
from evaluation.ir_metrics import all_metrics  # noqa: E402
from src.config import settings  # noqa: E402
from src.embeddings.embedder import get_embedder  # noqa: E402
from src.vector_stores.lance_store import LanceDBVectorStore  # noqa: E402

SETTINGS = [(200, 20), (350, 35), (500, 0), (500, 50), (500, 100), (800, 80), (1200, 120)]
K = 4


def main() -> None:
    cases = [c for c in json.loads((ROOT / "evaluation" / "dataset.json").read_text(encoding="utf-8")) if c["is_answerable"]]
    emb = get_embedder()
    qvecs = {c["id"]: emb.embed_query(c["question"]) for c in cases}
    rows = []
    for size, overlap in SETTINGS:
        chunks = build_corpus_chunks(Path(settings.corpus_dir), size, overlap)
        with tempfile.TemporaryDirectory() as tmp:
            store = LanceDBVectorStore(tmp)
            store.insert_chunks(chunks, emb.embed_documents([c.text for c in chunks]))
            per = []
            for c in cases:
                rel = relevant_chunk_ids(c, chunks)
                if not rel:  # evidence string split across a chunk boundary at this setting
                    per.append(None)
                    continue
                ids = [r.chunk_id for r in store.search(qvecs[c["id"]], K)]
                per.append(all_metrics(ids, rel, K))
        valid = [m for m in per if m]
        row = {"chunk_size": size, "overlap": overlap, "n_chunks": len(chunks),
               "questions_with_a_relevant_chunk": f"{len(valid)}/{len(cases)}",
               **{m: round(float(np.mean([v[m] for v in valid])), 4) for m in ("hit_rate", "recall", "mrr", "ndcg")}}
        rows.append(row)
        print(row)
    (ROOT / "evaluation" / "results_chunking_ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
