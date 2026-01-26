from ai_core.rag.metrics import evaluate_ranking


def test_evaluate_ranking_basic_hit() -> None:
    metrics = evaluate_ranking({"a", "b"}, ["b", "c", "a"], k=2)
    assert metrics.recall_at_k == 0.5
    assert metrics.mrr_at_k == 1.0
    assert metrics.ndcg_at_k > 0.0


def test_evaluate_ranking_no_relevant() -> None:
    metrics = evaluate_ranking(set(), ["a", "b"], k=5)
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr_at_k == 0.0
    assert metrics.ndcg_at_k == 0.0


def test_evaluate_ranking_no_hits() -> None:
    metrics = evaluate_ranking({"x"}, ["a", "b"], k=2)
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr_at_k == 0.0
    assert metrics.ndcg_at_k == 0.0
