"""Regression tests for RAG limit normalization edge cases."""

from __future__ import annotations

import pytest

from ai_core.rag.limits import clamp_fraction, normalize_max_candidates, normalize_top_k
from ai_core.rag.vector_client import PgVectorClient


@pytest.fixture()
def pg_vector_client(monkeypatch: pytest.MonkeyPatch) -> PgVectorClient:
    """Provide a PgVectorClient using a stubbed connection pool for token tests."""

    class DummyPool:
        def __init__(self, minconn: int, maxconn: int, dsn: str):
            self.minconn = minconn
            self.maxconn = maxconn
            self.dsn = dsn

        def closeall(self) -> None:  # pragma: no cover - no side effects to test
            pass

    monkeypatch.setattr("ai_core.rag.vector_client.SimpleConnectionPool", DummyPool)
    return PgVectorClient("postgresql://user:pass@localhost:5432/testdb", schema="test")


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        pytest.param(None, 0.5, 0.5, id="none-uses-default"),
        pytest.param("", 0.25, 0.25, id="blank-string-uses-default"),
        pytest.param("-0.3", 0.75, 0.75, id="negative-falls-back-to-default"),
        pytest.param(2, 0.6, 0.6, id="greater-than-one-clamps-to-default"),
        pytest.param("0.8", 0.4, 0.8, id="valid-fraction-preserved"),
    ],
)
def test_clamp_fraction_keeps_values_in_unit_interval(value, default, expected) -> None:
    """clamp_fraction confines results to [0, 1] and falls back to defaults."""

    assert clamp_fraction(value, default=default) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        pytest.param(None, 5, id="none-returns-default"),
        pytest.param("", 5, id="blank-string-returns-default"),
        pytest.param(0, 1, id="zero-clamped-to-minimum"),
        pytest.param(-5, 1, id="negative-clamped-to-minimum"),
        pytest.param(1, 1, id="single-result-allowed"),
        pytest.param(15, 10, id="large-values-clamped-to-maximum"),
    ],
)
def test_normalize_top_k_enforces_bounds(requested, expected) -> None:
    """normalize_top_k honours explicit bounds even for pathological inputs."""

    assert normalize_top_k(requested, default=5, minimum=1, maximum=10) == expected


@pytest.mark.parametrize(
    ("top_k", "requested", "cap", "expected"),
    [
        pytest.param(1, None, None, 1, id="none-falls-back-to-top-k"),
        pytest.param(3, "", None, 3, id="empty-string-falls-back-to-top-k"),
        pytest.param(4, -1, None, 4, id="negative-values-promoted-to-top-k"),
        pytest.param(5, 2, None, 5, id="less-than-top-k-promoted"),
        pytest.param(5, 500, 200, 200, id="large-request-respects-cap"),
        pytest.param(8, None, 100, 100, id="default-takes-configured-cap"),
        pytest.param(3, 9999, None, 9999, id="uncapped-large-request-preserved"),
    ],
)
def test_normalize_max_candidates_obeys_top_k_and_cap(
    top_k, requested, cap, expected
) -> None:
    """normalize_max_candidates never returns fewer than top_k or more than the cap."""

    assert normalize_max_candidates(top_k, requested, cap) == expected


@pytest.mark.parametrize(
    ("content", "top_k", "cap", "expected_tokens", "expected_candidates"),
    [
        pytest.param("", 1, 50, 1, 1, id="empty-content-reserves-single-token"),
        pytest.param("hello", 1, 50, 1, 1, id="single-word-reserves-single-token"),
        pytest.param(
            " ".join(["chunk"] * 2048),
            5,
            500,
            2048,
            500,
            id="oversized-content-respects-cap",
        ),
    ],
)
def test_estimated_tokens_feed_into_candidate_cap(
    pg_vector_client: PgVectorClient,
    content: str,
    top_k: int,
    cap: int,
    expected_tokens: int,
    expected_candidates: int,
) -> None:
    """Hybrid search combines token estimates with candidate caps without starving empty docs."""

    estimated_tokens = pg_vector_client._estimate_tokens(content)
    assert estimated_tokens == expected_tokens

    normalized_candidates = normalize_max_candidates(top_k, estimated_tokens, cap)
    assert normalized_candidates == expected_candidates
