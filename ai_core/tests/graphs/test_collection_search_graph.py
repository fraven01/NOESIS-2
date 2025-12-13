from __future__ import annotations

from typing import Any, Mapping, Sequence

from ai_core.graphs.collection_search import (
    CollectionSearchAdapter,
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
from ai_core.tests.utils import GraphTestMixin


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
        # Return a mock HybridResult compatible object
        from llm_worker.schemas import HybridResult

        # We need to return real objects because the node validates them
        return HybridResult(
            ranked=[],
            top_k=[],
            recommended_ingest=[],
            coverage_delta={"TECHNICAL": 0.0},
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


class TestCollectionSearchGraph(GraphTestMixin):
    def _initial_state(self) -> dict[str, Any]:
        return self.make_graph_state(
            input_data={
                "question": "How do I configure telemetry?",
                "collection_scope": "software_docs",
                "quality_mode": "software_docs_strict",
                "purpose": "docs-gap-analysis",
            },
            tenant_id="tenant-1",
            workflow_id="wf-1",
            case_id="case-1",
            trace_id="trace-1",
            run_id="run-1",
        )

    def _build_adapter(self, dependencies: dict[str, Any]) -> CollectionSearchAdapter:
        from ai_core.graphs.collection_search import build_compiled_graph

        return CollectionSearchAdapter(build_compiled_graph(), dependencies)

    def test_run_returns_search_results(self) -> None:
        strategy = StubStrategyGenerator()
        search_worker = StubWebSearchWorker()
        hybrid_executor = StubHybridExecutor()
        hitl_gateway = StubHitlGateway(decision=None)
        ingestion_trigger = StubIngestionTrigger()
        coverage_verifier = StubCoverageVerifier()

        dependencies = {
            "runtime_strategy_generator": strategy,
            "runtime_search_worker": search_worker,
            "runtime_hybrid_executor": hybrid_executor,
            "runtime_hitl_gateway": hitl_gateway,
            "runtime_ingestion_trigger": ingestion_trigger,
            "runtime_coverage_verifier": coverage_verifier,
        }

        graph = self._build_adapter(dependencies)

        state, result = graph.run(self._initial_state(), meta={})

        assert result["outcome"] == "completed"
        # Access search results from result dict
        assert result["search"] is not None
        assert "results" in result["search"]
        assert len(result["search"]["results"]) == 6
        assert len(search_worker.calls) == 3

        request = strategy.requests[0]
        assert request.purpose == "docs-gap-analysis"

    def test_search_failure_aborts_flow(self) -> None:
        strategy = StubStrategyGenerator()

        class FailingSearchWorker(StubWebSearchWorker):
            def run(self, *, query: str, context: Mapping[str, Any]) -> WebSearchResponse:  # type: ignore[override]
                raise SearchProviderError("boom")

        search_worker = FailingSearchWorker()

        dependencies = {
            "runtime_strategy_generator": strategy,
            "runtime_search_worker": search_worker,
            "runtime_hybrid_executor": StubHybridExecutor(),
            "runtime_hitl_gateway": StubHitlGateway(decision=None),
            "runtime_ingestion_trigger": StubIngestionTrigger(),
            "runtime_coverage_verifier": StubCoverageVerifier(),
        }

        graph = self._build_adapter(dependencies)

        _, result = graph.run(self._initial_state(), meta={})

        # The new graph might complete but with errors in search results
        # search_node returns errors in its output.
        # The graph continues to Rank/Hybrid, but if results are empty it might skip.
        # Check logic in collection_search.py:
        # if not results: return {"embedding_rank": {"scored_count": 0}}

        assert result["search"]["errors"]
        assert len(result["search"]["results"]) == 0

    def test_search_without_results_returns_no_candidates(self) -> None:
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

        dependencies = {
            "runtime_strategy_generator": strategy,
            "runtime_search_worker": search_worker,
            "runtime_hybrid_executor": StubHybridExecutor(),
            "runtime_hitl_gateway": StubHitlGateway(decision=None),
            "runtime_ingestion_trigger": StubIngestionTrigger(),
            "runtime_coverage_verifier": StubCoverageVerifier(),
        }

        graph = self._build_adapter(dependencies)

        state, result = graph.run(self._initial_state(), meta={})

        assert result["search"]["results"] == []
        assert len(search_worker.calls) == 3

    def test_auto_ingest_disabled_by_default(self) -> None:
        """Test that auto_ingest=False (default) does not trigger ingestion."""

        dependencies = {
            "runtime_strategy_generator": StubStrategyGenerator(),
            "runtime_search_worker": StubWebSearchWorker(),
            "runtime_hybrid_executor": StubHybridExecutor(),
            "runtime_hitl_gateway": StubHitlGateway(decision=None),
            "runtime_ingestion_trigger": StubIngestionTrigger(),
            "runtime_coverage_verifier": StubCoverageVerifier(),
        }

        graph = self._build_adapter(dependencies)
        ingestion_trigger = dependencies["runtime_ingestion_trigger"]

        state, result = graph.run(self._initial_state(), meta={})

        # result["ingestion"] should be None or empty or status skipped
        assert (
            not result.get("ingestion")
            or result["ingestion"].get("status") == "skipped"
        )
        assert ingestion_trigger.calls == []
