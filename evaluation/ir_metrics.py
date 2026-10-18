"""Information-retrieval metrics with binary relevance.

All functions take the ranked list of retrieved chunk ids (best first), the SET of relevant chunk ids
for the question (ground truth over the *whole* corpus, not just what was retrieved) and k.
"""
import math
from typing import Dict, List, Sequence, Set


def _flags(retrieved: Sequence[str], relevant: Set[str], k: int) -> List[int]:
    return [1 if cid in relevant else 0 for cid in list(retrieved)[:k]]


def hit_rate(retrieved, relevant, k) -> float:
    return 1.0 if any(_flags(retrieved, relevant, k)) else 0.0


def recall_at_k(retrieved, relevant, k) -> float:
    """|relevant retrieved in top-k| / |all relevant chunks in the corpus|."""
    if not relevant:
        raise ValueError("recall is undefined without relevant chunks")
    return sum(_flags(retrieved, relevant, k)) / len(relevant)


def precision_at_k(retrieved, relevant, k) -> float:
    return sum(_flags(retrieved, relevant, k)) / k


def reciprocal_rank(retrieved, relevant, k) -> float:
    for rank, rel in enumerate(_flags(retrieved, relevant, k), start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, relevant, k) -> float:
    """Binary-gain nDCG. IDCG assumes ALL relevant chunks (up to k) are ranked first, so the value is in [0, 1]."""
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(_flags(retrieved, relevant, k), start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision_at_k(retrieved, relevant, k) -> float:
    """Rank-aware context precision (RAGAS definition): mean of precision@i over the ranks i holding a relevant chunk.
    Rewards putting relevant chunks first; 0 if none retrieved."""
    flags = _flags(retrieved, relevant, k)
    n_rel = sum(flags)
    if n_rel == 0:
        return 0.0
    hits, total = 0, 0.0
    for rank, rel in enumerate(flags, start=1):
        if rel:
            hits += 1
            total += hits / rank
    return total / n_rel


def all_metrics(retrieved: Sequence[str], relevant: Set[str], k: int) -> Dict[str, float]:
    return {
        "hit_rate": hit_rate(retrieved, relevant, k),
        "recall": recall_at_k(retrieved, relevant, k),
        "precision": precision_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant, k),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
        "context_precision": context_precision_at_k(retrieved, relevant, k),
    }
