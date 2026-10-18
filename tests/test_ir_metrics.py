import math
import pytest
from evaluation.ir_metrics import all_metrics, ndcg_at_k, recall_at_k, reciprocal_rank, context_precision_at_k

def test_perfect_ranking():
    m = all_metrics(["a", "b", "x", "y"], {"a", "b"}, 4)
    assert m["hit_rate"] == 1 and m["recall"] == 1 and m["mrr"] == 1 and m["ndcg"] == pytest.approx(1.0)
    assert m["context_precision"] == 1.0

def test_ndcg_never_exceeds_one_with_multiple_relevant():
    # the bug in the original harness: 2 relevant chunks at ranks 1,2 gave nDCG 1.63
    assert ndcg_at_k(["a", "b", "c", "d"], {"a", "b", "c"}, 4) == pytest.approx(1.0)

def test_ndcg_known_value():
    # single relevant chunk at rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1
    assert ndcg_at_k(["x", "y", "a", "z"], {"a"}, 4) == pytest.approx(0.5)

def test_mrr_and_miss():
    assert reciprocal_rank(["x", "a"], {"a"}, 4) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}, 4) == 0.0
    assert all_metrics(["x", "y"], {"a"}, 4)["hit_rate"] == 0.0

def test_recall_uses_all_relevant_in_corpus_as_denominator():
    assert recall_at_k(["a", "x", "y", "z"], {"a", "b"}, 4) == 0.5

def test_k_truncates_ranking():
    assert reciprocal_rank(["x", "y", "a"], {"a"}, 2) == 0.0

def test_context_precision_rewards_early_relevant():
    early = context_precision_at_k(["a", "x", "y", "z"], {"a"}, 4)
    late = context_precision_at_k(["x", "y", "z", "a"], {"a"}, 4)
    assert early == 1.0 and late == 0.25

def test_recall_requires_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 4)
