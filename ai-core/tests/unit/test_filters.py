"""Unit tests for the RAG filter helpers."""

from apps.rag.filters import strict_filters


def test_strict_filters_matches_only_specific_tenant_and_case():
    meta = {"tenant": "t1", "case": "c1"}
    assert strict_filters(meta, "t1", "c1") is True
    assert strict_filters(meta, "t2", "c1") is False
    assert strict_filters(meta, "t1", "c2") is False
