import json
import math
import os
import re
import string
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.generator.llm import GroundedRAGService, QueryResponse
from src.vector_stores import get_vector_store

FALLBACK_PHRASE = "I do not have sufficient information in the provided context to answer this question."

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))

def compute_token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)

def is_chunk_relevant(chunk: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
    if not test_case.get("is_answerable", True):
        return False

    target_file = test_case.get("target_file", "")
    chunk_file = chunk.get("filename", "")
    text = chunk.get("text", "").lower()
    keywords = [kw.lower() for kw in test_case.get("expected_keywords", [])]

    # File match + section/keyword overlap
    if target_file and target_file in chunk_file:
        section = test_case.get("relevant_section", "").lower()
        chunk_section = chunk.get("section", "").lower()
        if section and section in chunk_section:
            return True
        if keywords and any(kw in text for kw in keywords):
            return True

    # Fallback keyword match
    if keywords:
        matches = sum(1 for kw in keywords if kw in text)
        if matches >= max(1, len(keywords) // 2):
            return True

    return False

def compute_retrieval_metrics(
    retrieved_chunks: List[Dict[str, Any]],
    test_case: Dict[str, Any],
    k: int
) -> Dict[str, float]:
    """
    Computes standard Information Retrieval (IR) metrics:
    - Hit Rate @ k
    - Recall @ k
    - MRR (Mean Reciprocal Rank)
    - nDCG @ k
    - Context Precision
    """
    is_ans = test_case.get("is_answerable", True)

    if not is_ans:
        # For unanswerable questions, 0 chunks should be relevant.
        # If retrieved chunks have very low similarity or don't fool the system, that's correct.
        return {
            "hit_rate": 1.0,
            "recall": 1.0,
            "mrr": 1.0,
            "ndcg": 1.0,
            "context_precision": 1.0
        }

    relevance_flags = [
        1 if is_chunk_relevant(c, test_case) else 0
        for c in retrieved_chunks[:k]
    ]

    total_relevant = 1  # Standard IR assumption: at least 1 ground truth target chunk exists

    # 1. Hit Rate
    hit_rate = 1.0 if any(r == 1 for r in relevance_flags) else 0.0

    # 2. Recall @ k
    num_relevant_retrieved = sum(relevance_flags)
    recall = min(1.0, num_relevant_retrieved / total_relevant)

    # 3. MRR
    mrr = 0.0
    for rank, rel in enumerate(relevance_flags, start=1):
        if rel == 1:
            mrr = 1.0 / rank
            break

    # 4. nDCG @ k
    dcg = 0.0
    for rank, rel in enumerate(relevance_flags, start=1):
        if rel == 1:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG (relevant chunk at rank 1)
    idcg = 1.0 / math.log2(2)  # log2(1+1) = 1.0 => 1.0
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # 5. Context Precision: sum(P@i * rel_i) / num_relevant_retrieved
    if num_relevant_retrieved > 0:
        precision_sum = 0.0
        cumulative_rel = 0
        for rank, rel in enumerate(relevance_flags, start=1):
            if rel == 1:
                cumulative_rel += 1
                precision_at_i = cumulative_rel / rank
                precision_sum += precision_at_i
        context_precision = precision_sum / num_relevant_retrieved
    else:
        context_precision = 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
        "context_precision": round(context_precision, 4)
    }

class EvaluationHarness:
    def __init__(self, dataset_path: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent
        self.dataset_path = Path(dataset_path or (base_dir / "dataset.json"))
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.test_cases = json.load(f)

    def run_evaluation(self, store_type: str = "lancedb", top_k: int = 4) -> Dict[str, Any]:
        rag_service = GroundedRAGService(store_type=store_type)
        results = []
        retrieval_latencies = []
        total_latencies = []

        print(f"\n{'='*70}")
        print(f"RUNNING RAG EVALUATION ON: {store_type.upper()} (Top-k={top_k})")
        print(f"{'='*70}")

        for tc in self.test_cases:
            qid = tc["id"]
            question = tc["question"]
            gold = tc.get("gold_answer", "")
            is_ans = tc.get("is_answerable", True)
            meta_filter = tc.get("metadata_filter")

            start_t = time.perf_counter()
            response: QueryResponse = rag_service.query(
                query_text=question,
                top_k=top_k,
                store_type=store_type,
                metadata_filter=meta_filter
            )
            elapsed_total = (time.perf_counter() - start_t) * 1000.0

            retrieval_latencies.append(response.retrieval_latency_ms)
            total_latencies.append(elapsed_total)

            # Compute Retrieval Metrics
            raw_chunk_dicts = [c.model_dump() for c in response.raw_chunks]
            ret_metrics = compute_retrieval_metrics(raw_chunk_dicts, tc, k=top_k)

            # Compute Answer Metrics
            prediction = response.answer
            em = compute_exact_match(prediction, gold)
            f1 = compute_token_f1(prediction, gold)

            # Faithfulness & Answer Relevance heuristic
            # Grounded if citations exist or correctly returns fallback
            is_fallback = FALLBACK_PHRASE.lower() in prediction.lower()
            if not is_ans:
                faithfulness = 1.0 if is_fallback else 0.0
                relevance = 1.0 if is_fallback else 0.2
            else:
                has_citations = len(response.citations) > 0
                keyword_overlap = sum(1 for kw in tc.get("expected_keywords", []) if kw.lower() in prediction.lower())
                faithfulness = 1.0 if (has_citations and keyword_overlap > 0) else 0.5
                relevance = min(1.0, 0.4 + (f1 * 0.6))

            case_result = {
                "id": qid,
                "question": question,
                "is_answerable": is_ans,
                "gold_answer": gold,
                "generated_answer": prediction,
                "citations": [c.model_dump() for c in response.citations],
                "retrieval_metrics": ret_metrics,
                "answer_metrics": {
                    "exact_match": round(em, 4),
                    "token_f1": round(f1, 4),
                    "faithfulness": round(faithfulness, 4),
                    "relevance": round(relevance, 4),
                    "correct_fallback": (is_fallback if not is_ans else None)
                },
                "latency_ms": {
                    "retrieval": response.retrieval_latency_ms,
                    "generation": response.generation_latency_ms,
                    "total": round(elapsed_total, 2)
                }
            }
            results.append(case_result)
            print(f"[{qid}] Recall@{top_k}={ret_metrics['recall']} | MRR={ret_metrics['mrr']} | F1={round(f1,2)} | Latency={round(elapsed_total,1)}ms")

        # Aggregate Suite Metrics
        answerable_cases = [r for r in results if r["is_answerable"]]
        unanswerable_cases = [r for r in results if not r["is_answerable"]]

        mean_hit_rate = np.mean([r["retrieval_metrics"]["hit_rate"] for r in answerable_cases])
        mean_recall = np.mean([r["retrieval_metrics"]["recall"] for r in answerable_cases])
        mean_mrr = np.mean([r["retrieval_metrics"]["mrr"] for r in answerable_cases])
        mean_ndcg = np.mean([r["retrieval_metrics"]["ndcg"] for r in answerable_cases])
        mean_precision = np.mean([r["retrieval_metrics"]["context_precision"] for r in answerable_cases])

        mean_f1 = np.mean([r["answer_metrics"]["token_f1"] for r in answerable_cases])
        mean_faithfulness = np.mean([r["answer_metrics"]["faithfulness"] for r in results])
        mean_relevance = np.mean([r["answer_metrics"]["relevance"] for r in results])
        fallback_accuracy = np.mean([1.0 if r["answer_metrics"]["correct_fallback"] else 0.0 for r in unanswerable_cases])

        p50_retrieval = np.percentile(retrieval_latencies, 50)
        p95_retrieval = np.percentile(retrieval_latencies, 95)
        p50_total = np.percentile(total_latencies, 50)
        p95_total = np.percentile(total_latencies, 95)

        summary = {
            "store_type": store_type,
            "total_questions": len(self.test_cases),
            "answerable_count": len(answerable_cases),
            "unanswerable_count": len(unanswerable_cases),
            "retrieval_evaluation": {
                "hit_rate_at_k": round(float(mean_hit_rate), 4),
                "recall_at_k": round(float(mean_recall), 4),
                "mrr": round(float(mean_mrr), 4),
                "ndcg_at_k": round(float(mean_ndcg), 4),
                "context_precision": round(float(mean_precision), 4)
            },
            "answer_evaluation": {
                "mean_token_f1": round(float(mean_f1), 4),
                "mean_faithfulness": round(float(mean_faithfulness), 4),
                "mean_answer_relevance": round(float(mean_relevance), 4),
                "fallback_accuracy_on_unanswerable": round(float(fallback_accuracy), 4)
            },
            "latency_evaluation_ms": {
                "p50_retrieval_latency": round(float(p50_retrieval), 2),
                "p95_retrieval_latency": round(float(p95_retrieval), 2),
                "p50_total_latency": round(float(p50_total), 2),
                "p95_total_latency": round(float(p95_total), 2)
            },
            "per_question_results": results
        }

        # Save results to file
        output_file = Path(__file__).resolve().parent / f"results_{store_type}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[+] Results saved to: {output_file}")
        print("\n" + "=" * 60)
        print(f"EVALUATION SUMMARY ({store_type.upper()})")
        print("=" * 60)
        print(f"Retrieval Recall@{top_k}  : {summary['retrieval_evaluation']['recall_at_k'] * 100:.1f}%")
        print(f"Retrieval Hit Rate@{top_k}: {summary['retrieval_evaluation']['hit_rate_at_k'] * 100:.1f}%")
        print(f"Retrieval MRR         : {summary['retrieval_evaluation']['mrr']:.4f}")
        print(f"Retrieval nDCG@{top_k}     : {summary['retrieval_evaluation']['ndcg_at_k']:.4f}")
        print(f"Context Precision     : {summary['retrieval_evaluation']['context_precision']:.4f}")
        print("-" * 60)
        print(f"Faithfulness Score    : {summary['answer_evaluation']['mean_faithfulness']:.4f}")
        print(f"Answer Relevance      : {summary['answer_evaluation']['mean_answer_relevance']:.4f}")
        print(f"Fallback Accuracy     : {summary['answer_evaluation']['fallback_accuracy_on_unanswerable'] * 100:.1f}%")
        print("-" * 60)
        print(f"p50 Retrieval Latency : {summary['latency_evaluation_ms']['p50_retrieval_latency']} ms")
        print(f"p95 Retrieval Latency : {summary['latency_evaluation_ms']['p95_retrieval_latency']} ms")
        print("=" * 60 + "\n")

        return summary

def main():
    harness = EvaluationHarness()
    # Evaluate primary store: LanceDB
    harness.run_evaluation(store_type="lancedb", top_k=4)
    # Evaluate secondary benchmark store: Qdrant
    harness.run_evaluation(store_type="qdrant", top_k=4)

if __name__ == "__main__":
    main()
