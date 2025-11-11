from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from ai_core.graphs.software_documentation_collection import (
    HitlDecision,
    SearchStrategy,
    SearchStrategyRequest,
    SoftwareDocumentationCollectionGraph,
)
from ai_core.tools.web_search import SearchProviderError, SearchResult, ToolOutcome, WebSearchResponse
from llm_worker.schemas import CoverageDimension, HybridResult, LLMScoredItem, RecommendedIngestItem


class StubStrategyGenerator:
    def __init__(self) -> None:
        self.requests: list[SearchStrategyRequest] = []

    def __call__(self, request: SearchStrategyRequest) -> SearchStrategy:
        self.requests.append(request)
        return SearchStrategy(
            queries=[
                f"{request.query} admin guide",
                f"{request.query} telemetry",
                f"{request.query} reporting",
            ],
            policies_applied=("tenant-default",),
            preferred_sources=("docs.acme.test",),
            disallowed_sources=("forum.acme.test",),
        )


class StubWebSearchWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def run(self, *, query: str, context: Mapping[str, Any]) -> WebSearchResponse:
        self.calls.append((query, context))
        results = [
            SearchResult(
                url="https://docs.acme.test/admin",
                title="Admin Guide",
                snippet="How to administer the platform",
                source="acme",
                score=0.9,
                is_pdf=False,
            ),
            SearchResult(
                url="https://docs.acme.test/telemetry",
                title="Telemetry",
                snippet="Telemetry configuration",
                source="acme",
                score=0.8,
                is_pdf=False,
            ),
        ]
        outcome = ToolOutcome(
            decision="ok",
            rationale="search_completed",
            meta={"provider": "stub", "latency_ms": 120},
        )
        return WebSearchResponse(results=results, outcome=outcome)


class StubHybridExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Sequence[Any], Mapping[str, Any]]] = []

    def run(
        self,
        *,
        scoring_context,
        candidates,
        tenant_context,
    ) -> HybridResult:
        self.calls.append((scoring_context, candidates, tenant_context))
        ranked = [
            LLMScoredItem(
                candidate_id="q0-0",
                score=92.4,
                reason="Comprehensive admin coverage",
                gap_tags=[CoverageDimension.TECHNICAL.value],
                risk_flags=[],
                facet_coverage={CoverageDimension.TECHNICAL: 0.85},
            ),
            LLMScoredItem(
                candidate_id="q0-1",
                score=81.0,
                reason="Telemetry details",
                gap_tags=[CoverageDimension.MONITORING_SURVEILLANCE.value],
                risk_flags=[],
                facet_coverage={CoverageDimension.MONITORING_SURVEILLANCE: 0.75},
            ),
        ]
        return HybridResult(
            ranked=ranked,
            top_k=ranked[:1],
            coverage_delta="TECHNICAL +20%",
            recommended_ingest=[
                RecommendedIngestItem(
                    candidate_id="q0-0",
                    reason="Fills admin hardening gap",
                )
            ],
        )


class StubHitlGateway:
    def __init__(self, decision: HitlDecision | None) -> None:
        self.decision = decision
        self.payloads: list[Mapping[str, Any]] = []

    def present(self, payload: Mapping[str, Any]) -> HitlDecision | None:
        self.payloads.append(payload)
        return self.decision


class StubIngestionTrigger:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Mapping[str, Any]]] = []

    def trigger(
        self,
        *,
        approved_urls: Sequence[str],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        urls = list(approved_urls)
        self.calls.append((urls, dict(context)))
        return {"status": "queued", "count": len(urls)}


class StubCoverageVerifier:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def verify(
        self,
        *,
        tenant_id: str,
        collection_scope: str,
        candidate_urls: Sequence[str],
        timeout_s: int,
        interval_s: int,
    ) -> Mapping[str, Any]:
        payload = {
            "tenant_id": tenant_id,
            "collection_scope": collection_scope,
            "candidate_urls": list(candidate_urls),
            "timeout_s": timeout_s,
            "interval_s": interval_s,
            "status": "complete",
            "coverage_delta": {"TECHNICAL": 0.2},
            "results": [
                {"url": url, "status": "success" if index == 0 else "pending"}
                for index, url in enumerate(candidate_urls)
            ],
        }
        self.calls.append(payload)
        return payload


def _initial_state() -> dict[str, Any]:
    return {
        "input": {
            "question": "How do I configure telemetry?",
            "collection_scope": "software_docs",
            "quality_mode": "software_docs_strict",
        },
        "meta": {
            "context": {
                "tenant_id": "tenant-1",
                "workflow_id": "wf-1",
                "case_id": "case-1",
                "trace_id": "trace-1",
                "run_id": "run-1",
            }
        },
    }


def test_full_run_completes_with_approval() -> None:
    strategy = StubStrategyGenerator()
    search_worker = StubWebSearchWorker()
    hybrid_executor = StubHybridExecutor()
    hitl_decision = HitlDecision(
        status="approved",
        approved_candidate_ids=("q0-0",),
        added_urls=("https://docs.acme.test/custom",),
    )
    hitl_gateway = StubHitlGateway(decision=hitl_decision)
    ingestion_trigger = StubIngestionTrigger()
    coverage_verifier = StubCoverageVerifier()
    graph = SoftwareDocumentationCollectionGraph(
        strategy_generator=strategy,
        search_worker=search_worker,
        hybrid_executor=hybrid_executor,
        hitl_gateway=hitl_gateway,
        ingestion_trigger=ingestion_trigger,
        coverage_verifier=coverage_verifier,
    )

    state, result = graph.run(_initial_state())

    assert result["outcome"] == "coverage_verified"
    assert "k_generate_strategy" in state["telemetry"]["nodes"]
    assert strategy.requests[0].tenant_id == "tenant-1"
    assert len(search_worker.calls) == 3
    scoring_context = hybrid_executor.calls[0][0]
    assert scoring_context.freshness_mode.name == "SOFTWARE_DOCS_STRICT"
    assert scoring_context.min_diversity_buckets == 3
    payload = hitl_gateway.payloads[0]
    deadline = datetime.fromisoformat(payload["deadline_at"])
    assert deadline.tzinfo is not None and deadline > datetime.now(timezone.utc)
    assert ingestion_trigger.calls[0][0] == [
        "https://docs.acme.test/admin",
        "https://docs.acme.test/custom",
    ]
    coverage_call = coverage_verifier.calls[0]
    assert coverage_call["timeout_s"] == 600
    assert coverage_call["interval_s"] == 30
    assert result["hitl"]["decision"]["status"] == "approved"


def test_run_waits_for_pending_hitl() -> None:
    strategy = StubStrategyGenerator()
    search_worker = StubWebSearchWorker()
    hybrid_executor = StubHybridExecutor()
    hitl_gateway = StubHitlGateway(decision=HitlDecision(status="pending"))
    ingestion_trigger = StubIngestionTrigger()
    coverage_verifier = StubCoverageVerifier()
    graph = SoftwareDocumentationCollectionGraph(
        strategy_generator=strategy,
        search_worker=search_worker,
        hybrid_executor=hybrid_executor,
        hitl_gateway=hitl_gateway,
        ingestion_trigger=ingestion_trigger,
        coverage_verifier=coverage_verifier,
    )

    _, result = graph.run(_initial_state())

    assert result["outcome"] == "awaiting_hitl"
    assert ingestion_trigger.calls == []
    assert coverage_verifier.calls == []


def test_search_failure_aborts_flow() -> None:
    strategy = StubStrategyGenerator()

    class FailingSearchWorker(StubWebSearchWorker):
        def run(self, *, query: str, context: Mapping[str, Any]) -> WebSearchResponse:  # type: ignore[override]
            raise SearchProviderError("boom")

    search_worker = FailingSearchWorker()
    hybrid_executor = StubHybridExecutor()
    hitl_gateway = StubHitlGateway(decision=None)
    ingestion_trigger = StubIngestionTrigger()
    coverage_verifier = StubCoverageVerifier()
    graph = SoftwareDocumentationCollectionGraph(
        strategy_generator=strategy,
        search_worker=search_worker,
        hybrid_executor=hybrid_executor,
        hitl_gateway=hitl_gateway,
        ingestion_trigger=ingestion_trigger,
        coverage_verifier=coverage_verifier,
    )

    _, result = graph.run(_initial_state())

    assert result["outcome"] == "search_failed"
    assert hybrid_executor.calls == []
    assert ingestion_trigger.calls == []
    assert coverage_verifier.calls == []


def test_auto_approve_after_deadline() -> None:
    strategy = StubStrategyGenerator()
    search_worker = StubWebSearchWorker()
    hybrid_executor = StubHybridExecutor()
    hitl_gateway = StubHitlGateway(decision=None)
    ingestion_trigger = StubIngestionTrigger()
    coverage_verifier = StubCoverageVerifier()
    graph = SoftwareDocumentationCollectionGraph(
        strategy_generator=strategy,
        search_worker=search_worker,
        hybrid_executor=hybrid_executor,
        hitl_gateway=hitl_gateway,
        ingestion_trigger=ingestion_trigger,
        coverage_verifier=coverage_verifier,
    )

    state, result = graph.run(_initial_state())
    assert result["outcome"] == "awaiting_hitl"
    assert len(hitl_gateway.payloads) == 1

    state["hitl"]["payload"]["deadline_at"] = (
        datetime.now(timezone.utc) - timedelta(hours=1)
    ).isoformat()

    state, result = graph.run(state)

    assert result["outcome"] == "coverage_verified"
    assert result["hitl"]["auto_approved"] is True
    assert result["hitl"]["decision"]["status"] == "approved"
    assert len(ingestion_trigger.calls) == 1
    coverage_summary = result["coverage"]["summary"]
    assert coverage_summary["ingested_count"] == 1
    assert coverage_summary["total_candidates"] >= 1
