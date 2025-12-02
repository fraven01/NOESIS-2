from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pytest
from django.core.cache import cache

from ai_core.tests.utils import make_test_meta
from llm_worker.domain_policies import DomainPolicy, DomainPolicyAction
from llm_worker.graphs import build_hybrid_graph
from llm_worker.graphs.hybrid_search_and_score import (
    HybridSearchAndScoreGraph,
    _build_domain_policy,
)
from llm_worker.schemas import (
    CoverageDimension,
    FreshnessMode,
    LLMScoredItem,
    RAGCoverageSummary,
    ScoringContext,
    SearchCandidate,
)


def _base_state() -> dict[str, Any]:
    now = datetime(2024, 6, 1, tzinfo=timezone.utc)
    return {
        "query": "employee data residency controls",
        "candidates": [
            {
                "id": "doc-1",
                "url": "https://example.com/policies/data-residency",
                "title": "Residency policy",
                "snippet": "Document describes residency law and audit duties.",
                "detected_date": now,
                "score": 95,
            },
            {
                "id": "doc-2",
                "url": "https://example.com/security/access-controls",
                "title": "Access controls",
                "snippet": "Explains access controls and integration APIs.",
                "detected_date": now,
                "score": 90,
            },
            {
                "id": "doc-3",
                "url": "https://external.gov/residency/overview",
                "title": "Government residency guidance",
                "snippet": "Official residency guidance with monitoring obligations.",
                "detected_date": now,
                "score": 88,
            },
        ],
    }


def _base_meta() -> dict[str, Any]:
    return make_test_meta(
        tenant_id="11111111-1111-4111-8111-111111111111",
        case_id="CASE-123",
        trace_id="TRACE-123",
        run_id="run-1",
        workflow_id="workflow-1",
        extra={
            "scoring_context": {
                "question": "Which policies govern residency?",
                "purpose": "research",
                "jurisdiction": "DE",
                "output_target": "briefing",
                "preferred_sources": ["https://example.com"],
                "disallowed_sources": ["https://old.example.com"],
                "collection_scope": "compliance",
                "min_diversity_buckets": 3,
            },
        },
    )


def _fake_rag() -> list[RAGCoverageSummary]:
    now = datetime(2024, 5, 20, tzinfo=timezone.utc)
    return [
        RAGCoverageSummary(
            document_id="00000000-0000-4000-8000-000000000001",
            title="Residency existing coverage",
            key_points=[
                "Covers legal obligations",
                "Includes audit overview",
                "Touches monitoring",
            ],
            coverage_facets={CoverageDimension.LEGAL: 0.7},
            custom_facets={"monitoring": 0.4},
            last_ingested_at=now,
        )
    ]


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    cache.clear()


@pytest.fixture(autouse=True)
def _policy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = DomainPolicy()
    policy.add_host(
        "example.com",
        DomainPolicyAction.BOOST,
        priority=85,
        source="test",
    )
    policy.add_host(
        "blocked.example.com",
        DomainPolicyAction.REJECT,
        priority=95,
        source="test",
    )

    monkeypatch.setattr(
        "llm_worker.domain_policies.get_domain_policy",
        lambda _tenant_id: policy,
    )
    monkeypatch.setattr(
        "llm_worker.graphs.hybrid_search_and_score.get_domain_policy",
        lambda _tenant_id: policy,
    )


def _graph(monkeypatch: pytest.MonkeyPatch) -> HybridSearchAndScoreGraph:
    graph = build_hybrid_graph()
    monkeypatch.setattr(graph, "_summarise_matches", lambda _matches: _fake_rag())
    monkeypatch.setattr(
        graph,
        "_retrieve_rag_context",
        lambda **_kwargs: (
            _fake_rag(),
            {"rag_unavailable": False, "rag_cache_hit": False},
        ),
    )

    def _mock_llm(
        *_args: Any, **_kwargs: Any
    ) -> tuple[list[LLMScoredItem], dict[str, bool], dict[str, Any]]:
        items = [
            LLMScoredItem(
                candidate_id="doc-1",
                score=96,
                reason="Matches residency question",
                gap_tags=[CoverageDimension.LEGAL.value],
                risk_flags=[],
                facet_coverage={CoverageDimension.LEGAL: 0.9},
            ),
            LLMScoredItem(
                candidate_id="doc-2",
                score=82,
                reason="Explains access controls",
                gap_tags=[CoverageDimension.ACCESS_PRIVACY_SECURITY.value],
                risk_flags=[],
                facet_coverage={CoverageDimension.ACCESS_PRIVACY_SECURITY: 0.6},
            ),
            LLMScoredItem(
                candidate_id="doc-3",
                score=78,
                reason="Government guidance",
                gap_tags=[CoverageDimension.MONITORING_SURVEILLANCE.value],
                risk_flags=[],
                facet_coverage={CoverageDimension.MONITORING_SURVEILLANCE: 0.5},
            ),
        ]
        return (
            items,
            {"llm_timeout": False, "llm_cache_hit": False},
            {
                "cache_hit": False,
                "fallback": None,
                "llm_items": len(items),
            },
        )

    monkeypatch.setattr(graph, "_run_llm_rerank", _mock_llm)
    return graph


def test_hybrid_graph_produces_ranked_result(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph(monkeypatch)
    state, result = graph.run(_base_state(), _base_meta())

    ranked = result["result"]["ranked"]
    top_ids = [item["candidate_id"] for item in ranked[:2]]

    assert top_ids[0] == "doc-2"
    assert "doc-2" in top_ids
    assert "doc-3" in top_ids
    assert "coverage_delta" in result["result"]
    assert state["flags"]["rag_unavailable"] is False
    fusion_debug = state["flags"]["debug"]["fusion"]
    assert fusion_debug["fused_scores"]["doc-2"] > fusion_debug["fused_scores"]["doc-1"]
    assert fusion_debug["rrf_components"]["doc-1"]["policy_bonus"] > 0
    decision = fusion_debug["rrf_components"]["doc-1"].get("policy_decision")
    assert decision and decision["action"] == "boost"
    assert state["flags"]["debug"]["llm"]["llm_items"] == 3
    normalise_debug = state["flags"]["debug"]["normalise"]["urls"]
    assert any(
        entry["url_canonical"].startswith("https://example.com")
        for entry in normalise_debug
    )
    heuristics = state["flags"]["debug"]["pre_filter"]["heuristics"]
    boost_entries = [
        entry
        for entry in heuristics
        if entry.get("policy", {}).get("action") == "boost"
    ]
    assert boost_entries


def test_rag_failure_sets_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = build_hybrid_graph()
    monkeypatch.setattr(graph, "_summarise_matches", lambda *_args: [])
    monkeypatch.setattr(
        graph,
        "_retrieve_rag_context",
        lambda **_kwargs: ([], {"rag_unavailable": True, "rag_cache_hit": False}),
    )
    monkeypatch.setattr(
        graph,
        "_run_llm_rerank",
        lambda **_kwargs: (
            [
                LLMScoredItem(
                    candidate_id="doc-1",
                    score=80,
                    reason="fallback",
                    gap_tags=[],
                    risk_flags=[],
                    facet_coverage={CoverageDimension.LEGAL: 0.5},
                )
            ],
            {"llm_timeout": False},
            {"cache_hit": False, "fallback": None, "llm_items": 1},
        ),
    )

    state, result = graph.run(_base_state(), _base_meta())

    assert result["flags"]["rag_unavailable"] is True
    assert "result" in state["hybrid_result"]


def test_llm_timeout_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = build_hybrid_graph()
    monkeypatch.setattr(graph, "_summarise_matches", lambda *_args: _fake_rag())
    monkeypatch.setattr(
        graph,
        "_retrieve_rag_context",
        lambda **_kwargs: (
            _fake_rag(),
            {"rag_unavailable": False, "rag_cache_hit": False},
        ),
    )
    monkeypatch.setattr(
        "llm_worker.graphs.hybrid_search_and_score.run_score_results",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )

    state, result = graph.run(_base_state(), _base_meta())

    assert result["flags"]["llm_timeout"] is True
    assert result["result"]["ranked"]
    assert "doc-1" in {item["candidate_id"] for item in result["result"]["ranked"]}
    assert state["flags"]["debug"]["llm"]["fallback"] == "timeout"


def test_llm_cache_avoids_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter = {"count": 0}

    graph = build_hybrid_graph()
    monkeypatch.setattr(graph, "_summarise_matches", lambda *_args: _fake_rag())
    monkeypatch.setattr(
        graph,
        "_retrieve_rag_context",
        lambda **_kwargs: (
            _fake_rag(),
            {"rag_unavailable": False, "rag_cache_hit": False},
        ),
    )

    def _fake_score(
        control: Mapping[str, Any],
        data: Mapping[str, Any],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        call_counter["count"] += 1
        return {
            "evaluations": [
                {
                    "candidate_id": "doc-1",
                    "score": 90,
                    "reason": "First",
                    "gap_tags": [],
                    "risk_flags": [],
                    "facet_coverage": {},
                },
                {
                    "candidate_id": "doc-2",
                    "score": 60,
                    "reason": "Second",
                    "gap_tags": [],
                    "risk_flags": [],
                    "facet_coverage": {},
                },
            ],
            "top_k": [],
        }

    monkeypatch.setattr(
        "llm_worker.graphs.hybrid_search_and_score.run_score_results",
        _fake_score,
    )

    graph.run(_base_state(), _base_meta())
    assert call_counter["count"] == 1

    graph.run(_base_state(), _base_meta())
    assert call_counter["count"] == 1


def test_pre_filter_records_reasons_and_limits_candidates() -> None:
    graph = build_hybrid_graph()
    graph.rerank_top_k = 3
    now = datetime.now(timezone.utc)
    stale = now - timedelta(days=1200)
    candidates: list[dict[str, Any]] = []
    for index in range(25):
        candidates.append(
            {
                "id": f"cand-{index}",
                "snippet": "Policy overview with integration hooks",
                "base_score": 100 - index,
                "host": f"site{index}.example.com",
                "detected_date": now,
            }
        )
    candidates[0]["is_duplicate"] = True
    candidates[1]["host"] = "blocked.example.com"
    candidates[1]["url"] = "https://blocked.example.com/policy"
    candidates[2]["snippet"] = ""
    candidates[3]["detected_date"] = stale

    scoring_context = ScoringContext(
        question="What policies apply?",
        purpose="audit",
        jurisdiction="DE",
        output_target="summary",
        preferred_sources=["https://site5.example.com"],
        disallowed_sources=["https://blocked.example.com"],
        collection_scope="compliance",
    )
    selected, debug = graph._pre_filter_candidates(
        candidates,
        scoring_context=scoring_context,
        domain_policy=_build_domain_policy(scoring_context, tenant_id="tenant"),
    )

    assert len(selected) <= graph.rerank_top_k * 4
    dropped_reasons = {entry["reason"] for entry in debug["dropped"]}
    assert {"duplicate", "policy_block", "empty_snippet", "stale"}.issubset(
        dropped_reasons
    )
    assert len(debug["mmr"]["selected"]) == len(selected)
    assert any(entry.get("freshness_penalty") for entry in debug["heuristics"])


def test_law_evergreen_keeps_stale_entries() -> None:
    graph = build_hybrid_graph()
    graph.rerank_top_k = 3
    current = datetime.now(timezone.utc)
    stale = current - timedelta(days=2000)
    fresh = current - timedelta(days=5)
    candidates = [
        {
            "id": f"law-{index}",
            "snippet": "Statutory guidance article",
            "base_score": 100 - index,
            "host": "laws.example.gov",
            "detected_date": stale if index == 0 else fresh,
        }
        for index in range(21)
    ]

    evergreen_context = ScoringContext(
        question="Which statutes are relevant?",
        purpose="legal",
        jurisdiction="DE",
        output_target="memo",
        preferred_sources=["https://laws.example.gov"],
        disallowed_sources=[],
        collection_scope="law",
        freshness_mode=FreshnessMode.LAW_EVERGREEN,
    )
    selected_evergreen, debug_evergreen = graph._pre_filter_candidates(
        candidates,
        scoring_context=evergreen_context,
        domain_policy=_build_domain_policy(evergreen_context, tenant_id="tenant"),
    )
    assert any(item["id"] == "law-0" for item in selected_evergreen)
    assert not any(entry["reason"] == "stale" for entry in debug_evergreen["dropped"])

    standard_context = evergreen_context.model_copy(
        update={"freshness_mode": FreshnessMode.STANDARD}
    )
    selected_standard, debug_standard = graph._pre_filter_candidates(
        candidates,
        scoring_context=standard_context,
        domain_policy=_build_domain_policy(standard_context, tenant_id="tenant"),
    )
    assert all(item["id"] != "law-0" for item in selected_standard)
    assert any(entry["reason"] == "stale" for entry in debug_standard["dropped"])


def test_build_llm_items_uses_rag_gaps_for_tags() -> None:
    graph = build_hybrid_graph()
    candidates = [
        {
            "id": "doc-gap",
            "snippet": "Monitoring procedures and alerting workflows",
        }
    ]
    ranked_payload = [
        {
            "id": "doc-gap",
            "score": 88,
            "reasons": ["Highlights monitoring requirements and oversight."],
        }
    ]
    rag_facets = {CoverageDimension.MONITORING_SURVEILLANCE: 0.1}

    items = graph._build_llm_items(
        ranked_payload,
        candidates,
        None,
        rag_facets=rag_facets,
    )

    assert CoverageDimension.MONITORING_SURVEILLANCE.value in items[0].gap_tags


def test_build_llm_items_truncates_reasons() -> None:
    graph = build_hybrid_graph()
    long_reason = "a" * 400
    candidates = [{"id": "doc-1", "snippet": "Details on monitoring"}]
    ranked_payload = [
        {
            "id": "doc-1",
            "score": 90,
            "reasons": [long_reason],
        }
    ]
    items = graph._build_llm_items(
        ranked_payload,
        candidates,
        None,
        rag_facets={},
    )

    assert items[0].reason.endswith("…")
    assert len(items[0].reason) <= 280


def test_invalid_candidate_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    graph = build_hybrid_graph()
    candidates = [
        {
            "id": "doc-invalid",
            "url": "https://example.com/invalid",
            "snippet": "Policy text",
        },
        {
            "id": "doc-valid",
            "url": "https://example.com/valid",
            "snippet": "Valid policy",
        },
    ]

    original_validate = SearchCandidate.model_validate.__func__  # type: ignore[attr-defined]

    def _raising_validate(cls, payload: Mapping[str, Any], *args: Any, **kwargs: Any):
        if payload.get("id") == "doc-invalid":
            raise ValueError("invalid candidate")
        return original_validate(cls, payload, *args, **kwargs)

    monkeypatch.setattr(
        SearchCandidate,
        "model_validate",
        classmethod(_raising_validate),
    )

    monkeypatch.setattr(
        "llm_worker.graphs.hybrid_search_and_score.run_score_results",
        lambda *_args, **_kwargs: {"evaluations": [], "top_k": []},
    )

    with caplog.at_level("DEBUG", logger="llm_worker.graphs.hybrid_search_and_score"):
        graph._run_llm_rerank(
            query="policy",
            candidates=candidates,
            meta=_base_meta(),
            scoring_context=None,
            rag_facets={},
            rag_summaries=[],
        )

    assert any(
        "hybrid.candidate_invalid" in record.getMessage() for record in caplog.records
    )
