"""Three-layer evaluation harness.

  Layer 1  Retrieval  : Hit Rate, Recall@k, MRR, nDCG@k, Precision@k, Context Precision (no LLM needed)
  Layer 2  Answer     : LLM-judge faithfulness / relevance / correctness + EM / token-F1 vs gold (needs GROQ_API_KEY)
  Layer 3  Latency    : p50 / p95 of query-embedding, vector search, generation and end-to-end

Usage:
  python -m evaluation.harness                      # all layers, both stores (answer layer on the primary store)
  python -m evaluation.harness --skip-answer-eval   # retrieval + latency only (no API key needed)
  python -m evaluation.harness --k 4 --repeats 5

Results are written to evaluation/results_<store>.json and evaluation/results_summary.md.
If the answer layer cannot run (no key) it is recorded as NOT_RUN - numbers are never simulated.
"""
import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ground_truth import build_corpus_chunks, relevant_chunk_ids  # noqa: E402
from evaluation.ir_metrics import all_metrics  # noqa: E402
from src.config import settings  # noqa: E402
from src.embeddings.embedder import get_embedder  # noqa: E402
from src.generator.llm import FALLBACK_ANSWER, GroundedRAGService, is_fallback_answer  # noqa: E402
from src.vector_stores import get_vector_store  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
_CITE = re.compile(r"\[?Chunk-\d+\]?", re.I)


# ----------------------------- answer string metrics (SQuAD-style) -----------------------------
def normalize_text(s: str) -> str:
    s = _CITE.sub(" ", s.lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def pct(values: List[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 2) if values else float("nan")


def mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


# ----------------------------- layer 1 + 3: retrieval & latency -----------------------------
def evaluate_retrieval(store_type: str, cases: List[Dict], chunks, k: int, repeats: int) -> Dict[str, Any]:
    embedder, store = get_embedder(), get_vector_store(store_type)
    for _ in range(3):  # warm-up (ONNX session, disk cache) so p95 is not dominated by first-call cost
        store.search(embedder.embed_query("warm up"), k)

    per_case, embed_ms, search_ms = [], [], []
    for tc in cases:
        relevant = relevant_chunk_ids(tc, chunks)
        row: Dict[str, Any] = {"id": tc["id"], "is_answerable": tc["is_answerable"], "n_relevant_chunks": len(relevant)}
        for variant, flt in (("unfiltered", None), ("filtered", tc.get("metadata_filter"))):
            lat_e, lat_s, results = [], [], None
            for _ in range(repeats):
                t0 = time.perf_counter()
                qv = embedder.embed_query(tc["question"])
                t1 = time.perf_counter()
                results = store.search(qv, k, flt)
                t2 = time.perf_counter()
                lat_e.append((t1 - t0) * 1000)
                lat_s.append((t2 - t1) * 1000)
            if variant == "unfiltered":
                embed_ms += lat_e
                search_ms += lat_s
            ids = [r.chunk_id for r in results]
            row[variant] = {
                "retrieved_chunk_ids": ids,
                "top_score": results[0].score if results else 0.0,
                **(all_metrics(ids, relevant, k) if tc["is_answerable"] else {}),
            }
        per_case.append(row)

    answerable = [r for r in per_case if r["is_answerable"]]
    agg = {}
    for variant in ("unfiltered", "filtered"):
        agg[variant] = {m: mean([r[variant][m] for r in answerable])
                        for m in ("hit_rate", "recall", "precision", "mrr", "ndcg", "context_precision")}

    # Threshold behaviour: how well does top-1 similarity separate answerable from unanswerable questions?
    ans_top = [r["unfiltered"]["top_score"] for r in answerable]
    una_top = [r["unfiltered"]["top_score"] for r in per_case if not r["is_answerable"]]
    thr = settings.similarity_threshold
    gate = {
        "threshold": thr,
        "min_top_score_answerable": min(ans_top),
        "max_top_score_unanswerable": max(una_top),
        "answerable_passing_gate": f"{sum(s >= thr for s in ans_top)}/{len(ans_top)}",
        "unanswerable_blocked_by_gate": f"{sum(s < thr for s in una_top)}/{len(una_top)}",
    }
    return {
        "per_case": per_case,
        "aggregate": agg,
        "similarity_gate": gate,
        "latency_ms": {
            "samples": len(search_ms),
            "query_embedding_p50": pct(embed_ms, 50), "query_embedding_p95": pct(embed_ms, 95),
            "vector_search_p50": pct(search_ms, 50), "vector_search_p95": pct(search_ms, 95),
        },
    }


# ----------------------------- layer 2: answer quality -----------------------------
def evaluate_answers(store_type: str, cases: List[Dict], k: int) -> Dict[str, Any]:
    if not settings.groq_configured:
        return {"status": "NOT_RUN", "reason": "GROQ_API_KEY not configured - no generated answers, nothing to judge"}
    from evaluation.judge import AnswerJudge  # imported lazily so retrieval-only runs need no key

    rag, judge = GroundedRAGService(store_type=store_type), AnswerJudge()
    rows, e2e = [], []
    for tc in cases:
        t0 = time.perf_counter()
        resp = rag.query(tc["question"], top_k=k, store_type=store_type)  # unfiltered: the honest setting
        e2e.append((time.perf_counter() - t0) * 1000)
        refused = is_fallback_answer(resp.answer)
        row: Dict[str, Any] = {
            "id": tc["id"], "question": tc["question"], "is_answerable": tc["is_answerable"],
            "generation_mode": resp.generation_mode, "refused": refused, "answer": resp.answer,
            "gold_answer": tc["gold_answer"], "citations": [c.chunk_id for c in resp.citations],
            "tokens": resp.token_usage, "latency_ms": {"search": resp.retrieval_latency_ms,
                                                       "generation": resp.generation_latency_ms,
                                                       "total": resp.total_latency_ms},
        }
        if resp.generation_mode == "extractive_fallback":
            row["note"] = "LLM unavailable - excluded from answer metrics"
            rows.append(row)
            continue
        if tc["is_answerable"] and not refused:
            row["exact_match"] = exact_match(resp.answer, tc["gold_answer"])
            row["token_f1"] = round(token_f1(resp.answer, tc["gold_answer"]), 4)
            verdict = judge.judge(tc["id"], tc["question"], [c.text for c in resp.raw_chunks], tc["gold_answer"], resp.answer)
            row["judge"] = verdict
        rows.append(row)
        print(f"  [{tc['id']}] refused={refused} mode={resp.generation_mode} "
              f"{'judge=' + json.dumps(row.get('judge', {}).get('scores')) if 'judge' in row else ''}")

    usable = [r for r in rows if r["generation_mode"] != "extractive_fallback"]
    ans = [r for r in usable if r["is_answerable"]]
    ans_scored = [r for r in ans if "judge" in r]
    una = [r for r in usable if not r["is_answerable"]]
    s = lambda r, key: (r["judge"]["scores"] or {}).get(key)  # noqa: E731
    judge_failures = sum(1 for r in ans_scored if r["judge"]["scores"] is None)
    return {
        "status": "OK", "generator_model": settings.groq_model, "judge_model": judge.model,
        "judge_calls": judge.calls, "judge_tokens": judge.tokens, "judge_unparseable": judge_failures,
        "answerable_total": len(ans), "answerable_answered": len(ans_scored),
        "false_refusal_rate_on_answerable": round(1 - len(ans_scored) / len(ans), 4) if ans else None,
        "refusal_accuracy_on_unanswerable": round(sum(r["refused"] for r in una) / len(una), 4) if una else None,
        "faithfulness_mean": mean([s(r, "faithfulness") for r in ans_scored if r["judge"]["scores"]]),
        "answer_relevance_mean": mean([s(r, "relevance") for r in ans_scored if r["judge"]["scores"]]),
        "correctness_vs_gold_mean": mean([s(r, "correctness") for r in ans_scored if r["judge"]["scores"]]),
        "exact_match_mean": mean([r["exact_match"] for r in ans_scored]),
        "token_f1_mean": mean([r["token_f1"] for r in ans_scored]),
        "mean_total_tokens_per_query": mean([r["tokens"]["total_tokens"] for r in usable]),
        "end_to_end_latency_ms": {"p50": pct(e2e, 50), "p95": pct(e2e, 95)},
        "per_case": rows,
    }


# ----------------------------- orchestration -----------------------------
def run(stores: List[str], k: int, repeats: int, skip_answer: bool, dataset: Path) -> None:
    cases = json.loads(dataset.read_text(encoding="utf-8"))
    chunks = build_corpus_chunks(Path(settings.corpus_dir), settings.default_chunk_size, settings.default_chunk_overlap)
    n_vec = {s: get_vector_store(s).count() for s in stores}
    for s in stores:  # the index must match the chunking the ground truth was built with
        if n_vec[s] != len(chunks):
            sys.exit(f"[!] {s} holds {n_vec[s]} vectors but the corpus yields {len(chunks)} chunks at "
                     f"size={settings.default_chunk_size}/overlap={settings.default_chunk_overlap}. "
                     f"Re-ingest (python -m src.cli ingest --store {s}) before evaluating.")

    summary_rows = []
    for i, store_type in enumerate(stores):
        print(f"\n=== {store_type.upper()}  k={k}  questions={len(cases)}  chunks={len(chunks)} ===")
        retrieval = evaluate_retrieval(store_type, cases, chunks, k, repeats)
        answer = ({"status": "NOT_RUN", "reason": "--skip-answer-eval"} if skip_answer
                  else evaluate_answers(store_type, cases, k) if i == 0
                  else {"status": "NOT_RUN", "reason": "answer layer is run on the primary store only; retrieval is the only difference between stores"})
        result = {
            "store_type": store_type, "k": k, "questions": len(cases),
            "answerable": sum(c["is_answerable"] for c in cases),
            "corpus": {"documents": len({c.filename for c in chunks}), "chunks": len(chunks),
                       "chunk_size": settings.default_chunk_size, "chunk_overlap": settings.default_chunk_overlap},
            "embedding": {"model": settings.embedding_model, "dim": settings.embedding_dim},
            "retrieval_evaluation": retrieval, "answer_evaluation": answer,
        }
        (EVAL_DIR / f"results_{store_type}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_rows.append(result)
        agg, lat = retrieval["aggregate"], retrieval["latency_ms"]
        print(f"Retrieval (unfiltered)  hit={agg['unfiltered']['hit_rate']} recall={agg['unfiltered']['recall']} "
              f"mrr={agg['unfiltered']['mrr']} ndcg={agg['unfiltered']['ndcg']} ctx_prec={agg['unfiltered']['context_precision']}")
        print(f"Retrieval (filtered)    hit={agg['filtered']['hit_rate']} mrr={agg['filtered']['mrr']} ndcg={agg['filtered']['ndcg']}")
        print(f"Latency  search p50/p95 = {lat['vector_search_p50']}/{lat['vector_search_p95']} ms | "
              f"embed p50/p95 = {lat['query_embedding_p50']}/{lat['query_embedding_p95']} ms")
        print(f"Gate: {retrieval['similarity_gate']}")
        if answer["status"] == "OK":
            print({k_: v for k_, v in answer.items() if k_ != "per_case"})
        else:
            print(f"Answer layer: {answer['status']} ({answer['reason']})")
    write_summary(summary_rows)


def write_summary(results: List[Dict]) -> None:
    L = ["# Evaluation summary (auto-generated by `python -m evaluation.harness`)\n"]
    r0 = results[0]
    L.append(f"{r0['questions']} questions ({r0['answerable']} answerable) | corpus: {r0['corpus']['documents']} docs, "
             f"{r0['corpus']['chunks']} chunks (size {r0['corpus']['chunk_size']}, overlap {r0['corpus']['chunk_overlap']}) | "
             f"embeddings: {r0['embedding']['model']} ({r0['embedding']['dim']}-d) | k={r0['k']}\n")
    L.append("## Retrieval (answerable questions, unfiltered search)\n")
    L.append("| Store | Hit Rate@k | Recall@k | MRR | nDCG@k | Precision@k | Context Precision |\n|---|---|---|---|---|---|---|")
    for r in results:
        a = r["retrieval_evaluation"]["aggregate"]["unfiltered"]
        L.append(f"| {r['store_type']} | {a['hit_rate']} | {a['recall']} | {a['mrr']} | {a['ndcg']} | {a['precision']} | {a['context_precision']} |")
    L.append("\n## Retrieval with the metadata filter the dataset specifies (format = pdf/html/md)\n")
    L.append("| Store | Hit Rate@k | Recall@k | MRR | nDCG@k |\n|---|---|---|---|---|")
    for r in results:
        a = r["retrieval_evaluation"]["aggregate"]["filtered"]
        L.append(f"| {r['store_type']} | {a['hit_rate']} | {a['recall']} | {a['mrr']} | {a['ndcg']} |")
    L.append("\n## Latency (ms, warm, per-query samples)\n")
    L.append("| Store | search p50 | search p95 | query-embed p50 | query-embed p95 | samples |\n|---|---|---|---|---|---|")
    for r in results:
        l = r["retrieval_evaluation"]["latency_ms"]
        L.append(f"| {r['store_type']} | {l['vector_search_p50']} | {l['vector_search_p95']} | {l['query_embedding_p50']} | {l['query_embedding_p95']} | {l['samples']} |")
    L.append("\n## Similarity gate\n")
    for r in results[:1]:
        L.append("```\n" + json.dumps(r["retrieval_evaluation"]["similarity_gate"], indent=2) + "\n```")
    L.append("\n## Answer quality (primary store)\n")
    a = results[0]["answer_evaluation"]
    if a["status"] != "OK":
        L.append(f"**NOT RUN** - {a['reason']}")
    else:
        L.append("| Metric | Value |\n|---|---|")
        for key in ("generator_model", "judge_model", "faithfulness_mean", "answer_relevance_mean", "correctness_vs_gold_mean",
                    "exact_match_mean", "token_f1_mean", "false_refusal_rate_on_answerable", "refusal_accuracy_on_unanswerable",
                    "mean_total_tokens_per_query", "judge_calls", "judge_tokens", "judge_unparseable"):
            L.append(f"| {key} | {a[key]} |")
        L.append(f"| end-to-end latency p50 / p95 (ms) | {a['end_to_end_latency_ms']['p50']} / {a['end_to_end_latency_ms']['p95']} |")
    (EVAL_DIR / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[+] wrote {EVAL_DIR / 'results_summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", nargs="+", default=["lancedb", "qdrant"], choices=["lancedb", "qdrant"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5, help="repetitions per query for latency percentiles")
    ap.add_argument("--skip-answer-eval", action="store_true")
    ap.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.json")
    args = ap.parse_args()
    run(args.stores, args.k, args.repeats, args.skip_answer_eval, args.dataset)


if __name__ == "__main__":
    main()
