from __future__ import annotations

from ai_core.agent.harness.golden import load_golden_queries


def test_golden_queries_load_and_validate():
    queries = load_golden_queries()
    assert len(queries) == 5
    assert all(q.flow_name == "rag_query" for q in queries)


def test_golden_queries_are_deterministically_ordered_by_id():
    queries = load_golden_queries()
    ids = [q.id for q in queries]
    assert ids == sorted(ids)
