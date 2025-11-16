from __future__ import annotations

from typing import Any, Mapping, Sequence

from ai_core.graphs.collection_search import (
    CollectionSearchGraph,
    HitlDecision,
    SearchStrategy,
    SearchStrategyRequest,
)
from ai_core.tools.web_search import (
    SearchProviderError,
    SearchResult,
    ToolOutcome,
    WebSearchResponse,
)


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
    ) -> Any:
        self.calls.append((scoring_context, candidates, tenant_context))
        raise AssertionError(
            "hybrid executor should not be invoked in search-only flow"
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
            "purpose": "docs-gap-analysis",
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


def test_run_returns_search_results() -> None:
    strategy = StubStrategyGenerator()
    search_worker = StubWebSearchWorker()
    hybrid_executor = StubHybridExecutor()
    hitl_gateway = StubHitlGateway(decision=None)
    ingestion_trigger = StubIngestionTrigger()
    coverage_verifier = StubCoverageVerifier()
    graph = CollectionSearchGraph(
        strategy_generator=strategy,
        search_worker=search_worker,
        hybrid_executor=hybrid_executor,
        hitl_gateway=hitl_gateway,
        ingestion_trigger=ingestion_trigger,
        coverage_verifier=coverage_verifier,
    )

    state, result = graph.run(_initial_state())

    assert result["outcome"] == "search_completed"
    assert "k_generate_strategy" in state["telemetry"]["nodes"]
    assert len(search_worker.calls) == 3
    assert hybrid_executor.calls == []
    assert ingestion_trigger.calls == []
    assert coverage_verifier.calls == []
    assert hitl_gateway.payloads == []

    request = strategy.requests[0]
    assert request.tenant_id == "tenant-1"
    assert request.purpose == "docs-gap-analysis"

    search_payload = result["search"]
    assert search_payload is not None
    assert len(search_payload["results"]) == 6
    expected_queries = [
        f"{request.query} admin guide",
        f"{request.query} telemetry",
        f"{request.query} reporting",
    ]
    assert search_payload["strategy"]["queries"] == expected_queries


def test_search_failure_aborts_flow() -> None:
    strategy = StubStrategyGenerator()

    class FailingSearchWorker(StubWebSearchWorker):
        def run(self, *, query: str, context: Mapping[str, Any]) -> WebSearchResponse:  # type: ignore[override]
            raise SearchProviderError("boom")

    search_worker = FailingSearchWorker()
    hybrid_executor = StubHybridExecutor()
    graph = CollectionSearchGraph(
        strategy_generator=strategy,
        search_worker=search_worker,
        hybrid_executor=hybrid_executor,
        hitl_gateway=StubHitlGateway(decision=None),
        ingestion_trigger=StubIngestionTrigger(),
        coverage_verifier=StubCoverageVerifier(),
    )

    _, result = graph.run(_initial_state())

    assert result["outcome"] == "search_failed"
    assert result["search"]["errors"]
    assert hybrid_executor.calls == []


def test_search_without_results_returns_no_candidates() -> None:
    strategy = StubStrategyGenerator()

    class EmptySearchWorker(StubWebSearchWorker):
        def run(self, *, query: str, context: Mapping[str, Any]) -> WebSearchResponse:  # type: ignore[override]
            self.calls.append((query, context))
            outcome = ToolOutcome(
                decision="ok",
                rationale="no_results",
                meta={"provider": "stub", "latency_ms": 10},
            )
            return WebSearchResponse(results=[], outcome=outcome)

    search_worker = EmptySearchWorker()
    graph = CollectionSearchGraph(
        strategy_generator=strategy,
        search_worker=search_worker,
        hybrid_executor=StubHybridExecutor(),
        hitl_gateway=StubHitlGateway(decision=None),
        ingestion_trigger=StubIngestionTrigger(),
        coverage_verifier=StubCoverageVerifier(),
    )

    state, result = graph.run(_initial_state())

    assert result["outcome"] == "no_candidates"
    assert result["search"]["results"] == []
    assert len(search_worker.calls) == 3
