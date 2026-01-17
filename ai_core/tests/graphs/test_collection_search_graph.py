from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Mapping, Sequence

# -----------------------------------------------------------------------------
# Fixture Overrides (Blocking global conftest side-effects)
# -----------------------------------------------------------------------------


@pytest.fixture
def documents_repository_stub():
    """Override to prevent importing ai_core.services."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _auto_documents_repository(documents_repository_stub):
    """Override to prevent importing ai_core.services."""
    yield


@pytest.fixture(autouse=True)
def ingestion_status_store():
    """Override to prevent importing ai_core.services/ingestion_status."""
    return MagicMock()


@pytest.fixture(autouse=True)
def disable_async_graphs():
    """Override to prevent importing ai_core.services."""
    yield


@pytest.fixture(autouse=True)
def _prevent_db_access():
    """Ensure no DB access during these tests."""
    with patch(
        "django.db.backends.base.base.BaseDatabaseWrapper.connect",
        side_effect=RuntimeError("DB_ACCESS_FORBIDDEN"),
    ):
        yield


# -----------------------------------------------------------------------------
# Test Logic (Same as before)
# -----------------------------------------------------------------------------


@pytest.fixture
def cs_module():
    """Import collection_search module with isolation."""
    mock_modules = {
        "ai_core.rag.embeddings": MagicMock(),
        "ai_core.llm.client": MagicMock(),
        "django.urls": MagicMock(),
        "django.conf": MagicMock(),
        # Partially mock services but allow strategy (Pydantic models) to load
        "ai_core.services.collection_search.auto_ingest": MagicMock(),
        # "ai_core.services.collection_search.hitl": MagicMock(), # Safe to load real (simple helper)
        "ai_core.services.collection_search.scoring": MagicMock(),
        # "ai_core.services.collection_search.strategy": MagicMock(), # MUST be real for SearchStrategy model
        "crawler": MagicMock(),
        "crawler.manager": MagicMock(),
    }

    with patch(
        "ai_core.infra.observability.observe_span",
        side_effect=lambda name=None, **kwargs: lambda func: func,
    ), patch.dict(sys.modules, mock_modules):

        module_name = "ai_core.graphs.technical.collection_search"
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            import ai_core.graphs.technical.collection_search as m

            yield m
        finally:
            if module_name in sys.modules:
                del sys.modules[module_name]


class TestCollectionSearchGraph:
    """Tests for CollectionSearch using isolated module."""

    def _plan_slot_value(self, plan: Mapping[str, Any], key: str) -> Any:
        for slot in plan.get("slots", []):
            if slot.get("key") == key:
                return slot.get("value")
        return None

    def _plan_metadata(self, plan: Mapping[str, Any]) -> Mapping[str, Any]:
        return plan.get("metadata") or {}

    def _initial_state(self) -> dict[str, Any]:
        """Build GraphIOSpec-compliant boundary request."""
        from ai_core.contracts import BusinessContext, ScopeContext

        scope = ScopeContext(
            tenant_id="tenant-1",
            trace_id="trace-1",
            invocation_id="invoke-1",
            run_id="run-1",
        )
        business = BusinessContext(
            workflow_id="wf-1",
            case_id="case-1",
        )
        tool_context = scope.to_tool_context(business=business)

        return {
            "schema_id": "noesis.graphs.collection_search",
            "schema_version": "1.0.0",
            "input": {
                "question": "How do I configure telemetry?",
                "collection_scope": "550e8400-e29b-41d4-a716-446655440000",
                "quality_mode": "software_docs_strict",
                "purpose": "docs-gap-analysis",
            },
            "tool_context": tool_context,
        }

    def _tool_context_meta(self) -> dict[str, Any]:
        from ai_core.contracts import BusinessContext, ScopeContext

        scope = ScopeContext(
            tenant_id="tenant-1",
            trace_id="trace-1",
            invocation_id="invoke-1",
            run_id="run-1",
        )
        business = BusinessContext(
            workflow_id="wf-1",
            case_id="case-1",
        )
        tool_context = scope.to_tool_context(business=business)
        return {"tool_context": tool_context.model_dump(mode="json", exclude_none=True)}

    def test_run_returns_search_results(self, cs_module) -> None:
        from ai_core.tools.web_search import (
            WebSearchResponse,
            SearchResult,
            ToolOutcome,
        )

        class LocalStubStrategyGenerator:
            def __init__(self) -> None:
                self.requests: list[Any] = []

            def __call__(self, request: Any) -> Any:
                # Ensure scoring returns floats to avoid Mock sorting errors
                cs_module.cosine_similarity.return_value = 0.8
                cs_module.calculate_generic_heuristics.return_value = 0.5

                self.requests.append(request)
                return cs_module.SearchStrategy(
                    queries=[
                        f"{request.query} admin guide",
                        f"{request.query} telemetry",
                        f"{request.query} reporting",
                    ],
                    policies_applied=("tenant-default",),
                    preferred_sources=("docs.acme.test",),
                    disallowed_sources=("forum.acme.test",),
                )

        class LocalStubWebSearchWorker:
            def __init__(self) -> None:
                self.calls: list[tuple[str, Mapping[str, Any]]] = []

            def run(
                self, *, query: str, context: Mapping[str, Any]
            ) -> WebSearchResponse:
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

        class LocalStubHybridExecutor:
            def __init__(self) -> None:
                self.calls: list[tuple[Any, Sequence[Any], Mapping[str, Any]]] = []

            def run(self, *, scoring_context, candidates, tenant_context) -> Any:
                self.calls.append((scoring_context, candidates, tenant_context))
                result = MagicMock()
                # Mock result structure that translates to expected dict
                result.candidates = {}
                result.result = MagicMock()
                result.result.ranked = []

                # Mock model_dump behavior expected by hybrid_score_node
                result.model_dump.return_value = {
                    "ranked": [],
                    "candidates": {},
                    "coverage_delta": {"TECHNICAL": 0.0},
                }
                return result

        class LocalStubHitlGateway:
            def __init__(self, decision: Any | None) -> None:
                self.decision = decision
                self.payloads: list[Mapping[str, Any]] = []

            def present(self, payload: Mapping[str, Any]) -> Any | None:
                self.payloads.append(payload)
                return self.decision

        class LocalStubCoverageVerifier:
            def __init__(self) -> None:
                self.calls: list[Mapping[str, Any]] = []

            def verify(self, **kwargs) -> Mapping[str, Any]:
                self.calls.append(kwargs)
                return {"status": "complete", "coverage_delta": {"TECHNICAL": 0.2}}

        strategy = LocalStubStrategyGenerator()
        search_worker = LocalStubWebSearchWorker()
        hybrid_executor = LocalStubHybridExecutor()
        hitl_gateway = LocalStubHitlGateway(decision=None)
        coverage_verifier = LocalStubCoverageVerifier()

        dependencies = {
            "runtime_strategy_generator": strategy,
            "runtime_search_worker": search_worker,
            "runtime_hybrid_executor": hybrid_executor,
            "runtime_hitl_gateway": hitl_gateway,
            "runtime_coverage_verifier": coverage_verifier,
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph = CollectionSearchAdapter(dependencies)

        initial_state = self._initial_state()
        meta = self._tool_context_meta()
        state, result = graph.run(initial_state, meta=meta)

        if result["outcome"] != "completed":
            print(f"DEBUG: Graph failed with result: {result}")
        assert result["outcome"] == "completed"
        assert result["search"] is not None
        assert "results" in result["search"]
        assert len(result["search"]["results"]) == 6
        assert len(search_worker.calls) == 3

        request = strategy.requests[0]
        assert request.purpose == "docs-gap-analysis"

        assert result.get("plan") is not None
        plan_metadata = self._plan_metadata(result["plan"])
        assert plan_metadata.get("hitl", {}).get("required") is True
        assert result.get("ingestion", {}).get("status") == "planned_only"

    def test_search_failure_aborts_flow(self, cs_module) -> None:
        from ai_core.tools.web_search import (
            WebSearchResponse,
            SearchProviderError,
        )

        class LocalStubStrategyGenerator:
            def __call__(self, request: Any) -> Any:
                return cs_module.SearchStrategy(
                    queries=[request.query],
                    policies_applied=(),
                    preferred_sources=(),
                    disallowed_sources=(),
                )

        class FailingSearchWorker:
            def __init__(self):
                self.calls = []

            def run(
                self, *, query: str, context: Mapping[str, Any]
            ) -> WebSearchResponse:
                raise SearchProviderError("boom")

        class MockObj:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def present(self, p):
                return None

            def run(self, **k):
                return MagicMock()

            def verify(self, **k):
                return {}

        dependencies = {
            "runtime_strategy_generator": LocalStubStrategyGenerator(),
            "runtime_search_worker": FailingSearchWorker(),
            "runtime_hybrid_executor": MockObj(),
            "runtime_hitl_gateway": MockObj(),
            "runtime_coverage_verifier": MockObj(),
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph = CollectionSearchAdapter(dependencies)
        initial_state = self._initial_state()
        meta = self._tool_context_meta()
        _, result = graph.run(initial_state, meta=meta)

        if not result.get("search"):
            print(f"DEBUG: Search failure test aborted graph: {result}")
        assert result["search"]["errors"]
        assert len(result["search"]["results"]) == 0

    def test_search_without_results_returns_no_candidates(self, cs_module) -> None:
        from ai_core.tools.web_search import (
            WebSearchResponse,
            ToolOutcome,
        )

        class LocalStubStrategyGenerator:
            def __call__(self, request: Any) -> Any:
                return cs_module.SearchStrategy(
                    queries=[
                        f"{request.query} 1",
                        f"{request.query} 2",
                        f"{request.query} 3",
                    ],
                    policies_applied=(),
                    preferred_sources=(),
                    disallowed_sources=(),
                )

        class EmptySearchWorker:
            def __init__(self):
                self.calls = []

            def run(
                self, *, query: str, context: Mapping[str, Any]
            ) -> WebSearchResponse:
                self.calls.append((query, context))
                outcome = ToolOutcome(
                    decision="ok",
                    rationale="no_results",
                    meta={"provider": "stub", "latency_ms": 10},
                )
                return WebSearchResponse(results=[], outcome=outcome)

        class MockObj:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def present(self, p):
                return None

            def run(self, **k):
                return MagicMock()

            def verify(self, **k):
                return {}

        strategy = LocalStubStrategyGenerator()
        search_worker = EmptySearchWorker()

        dependencies = {
            "runtime_strategy_generator": strategy,
            "runtime_search_worker": search_worker,
            "runtime_hybrid_executor": MockObj(),
            "runtime_hitl_gateway": MockObj(),
            "runtime_coverage_verifier": MockObj(),
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph = CollectionSearchAdapter(dependencies)

        initial_state = self._initial_state()
        meta = self._tool_context_meta()
        state, result = graph.run(initial_state, meta=meta)

        assert result["search"]["results"] == []
        assert len(search_worker.calls) == 3

    def test_auto_ingest_disabled_by_default(self, cs_module) -> None:
        from ai_core.tools.web_search import (
            WebSearchResponse,
            ToolOutcome,
        )

        class LocalStubStrategyGenerator:
            def __call__(self, request: Any) -> Any:
                return cs_module.SearchStrategy(
                    queries=[request.query],
                    policies_applied=(),
                    preferred_sources=(),
                    disallowed_sources=(),
                )

        class StubWebSearchWorker:
            def run(self, *, query, context) -> WebSearchResponse:
                return WebSearchResponse(
                    results=[],
                    outcome=ToolOutcome(decision="ok", rationale="none", meta={}),
                )

        class MockObj:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def present(self, p):
                return None

            def run(self, **k):
                m = MagicMock()
                m.model_dump.return_value = {}
                m.ranked = []
                m.candidates = {}
                return m

            def verify(self, **k):
                return {}

        dependencies = {
            "runtime_strategy_generator": LocalStubStrategyGenerator(),
            "runtime_search_worker": StubWebSearchWorker(),
            "runtime_hybrid_executor": MockObj(),
            "runtime_hitl_gateway": MockObj(),
            "runtime_coverage_verifier": MockObj(),
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph = CollectionSearchAdapter(dependencies)

        initial_state = self._initial_state()
        meta = self._tool_context_meta()
        state, result = graph.run(initial_state, meta=meta)

        assert result.get("ingestion", {}).get("status") == "planned_only"

    def test_trigger_ingestion_flow(self, cs_module) -> None:
        """Test that approved plan triggers ingestion via CrawlerManager."""
        # Ensure scoring returns floats
        cs_module.cosine_similarity.return_value = 0.8
        cs_module.calculate_generic_heuristics.return_value = 0.5

        from ai_core.tools.web_search import (
            WebSearchResponse,
            ToolOutcome,
            SearchResult,
        )

        decision = cs_module.HitlDecision(
            status="approved",
            approved_candidate_ids=("q0-0",),
            added_urls=("https://added.com",),
            rationale="LGTM",
        )

        class LocalStubHitlGateway:
            def present(self, payload):
                return decision

        class LocalStubStrategyGenerator:
            def __call__(self, request):
                return cs_module.SearchStrategy(
                    queries=["q"],
                    policies_applied=(),
                    preferred_sources=(),
                    disallowed_sources=(),
                )

        class StubWebSearchWorker:
            def run(self, *, query, context) -> WebSearchResponse:
                results = [
                    SearchResult(
                        url="https://docs.acme.test/admin",
                        title="Admin Guide",
                        snippet="How to administer the platform",
                        source="acme",
                        score=0.9,
                        is_pdf=False,
                    ),
                ]
                return WebSearchResponse(
                    results=results,
                    outcome=ToolOutcome(decision="ok", rationale="none", meta={}),
                )

        class ValidCandidatesHybridExecutor:
            def run(self, *, scoring_context, candidates, tenant_context) -> Any:
                result = MagicMock()
                c1 = {
                    "url": "https://docs.acme.test/admin",
                    "title": "Admin",
                    "score": 0.9,
                }

                result.ranked = []
                result.candidates = {"cand1": c1}
                result.coverage_delta = "TECHNICAL"

                def model_dump(**kwargs):
                    return {"candidates": {"cand1": c1}, "result": {"ranked": []}}

                result.model_dump = model_dump
                return result

        mock_crawler_manager = MagicMock()
        mock_crawler_manager.dispatch_crawl_request.return_value = {
            "count": 2,
            "run_id": "test-crawl-run",
        }

        # Inject factory to bypass import
        dependencies = {
            "runtime_strategy_generator": LocalStubStrategyGenerator(),
            "runtime_search_worker": StubWebSearchWorker(),
            "runtime_hybrid_executor": ValidCandidatesHybridExecutor(),
            "runtime_hitl_gateway": LocalStubHitlGateway(),
            "runtime_coverage_verifier": MagicMock(),
            "runtime_crawler_manager": mock_crawler_manager,
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph_adapter = CollectionSearchAdapter(dependencies)

        initial_state = self._initial_state()
        initial_state["input"]["execute_plan"] = True

        meta = self._tool_context_meta()
        state, result = graph_adapter.run(initial_state, meta=meta)

        if result.get("outcome") != "completed":
            print(f"DEBUG: Graph failed with result: {result}")

        assert result.get("outcome") == "completed", f"Graph failed: {result}"
        assert result.get("hitl") is not None, "HITL state missing"
        assert (
            result["hitl"].get("decision") is not None
        ), f"HITL decision missing: {result['hitl']}"
        plan_metadata = self._plan_metadata(result["plan"])
        assert (
            self._plan_slot_value(result["plan"], "collection_id")
            == "550e8400-e29b-41d4-a716-446655440000"
        )
        assert plan_metadata.get("execution_mode") == "acquire_and_ingest"
        assert plan_metadata.get("hitl", {}).get("required") is False
        selected_urls = self._plan_slot_value(result["plan"], "selected_urls") or []
        assert "https://added.com" in selected_urls
        assert "https://docs.acme.test/admin" in selected_urls

        assert result["ingestion"]["status"] == "triggered"
        assert result["ingestion"]["task_info"]["run_id"] == "test-crawl-run"

        # Verify CrawlerManager was called
        mock_crawler_manager.dispatch_crawl_request.assert_called_once()
        call_args = mock_crawler_manager.dispatch_crawl_request.call_args
        crawl_req = call_args[0][0]
        assert len(crawl_req.origins) == 2
        urls = [o.url for o in crawl_req.origins]
        assert "https://added.com" in urls
        assert "https://docs.acme.test/admin" in urls

    def test_auto_ingest_triggers_crawler(self, cs_module) -> None:
        """Test that auto_ingest triggers crawler without HITL approval."""
        cs_module.cosine_similarity.return_value = 0.8
        cs_module.calculate_generic_heuristics.return_value = 0.5

        from ai_core.tools.web_search import (
            WebSearchResponse,
            ToolOutcome,
            SearchResult,
        )

        class LocalStubStrategyGenerator:
            def __call__(self, request):
                return cs_module.SearchStrategy(
                    queries=["q"],
                    policies_applied=(),
                    preferred_sources=(),
                    disallowed_sources=(),
                )

        class StubWebSearchWorker:
            def run(self, *, query, context) -> WebSearchResponse:
                results = [
                    SearchResult(
                        url="https://docs.acme.test/admin",
                        title="Admin Guide",
                        snippet="How to administer the platform",
                        source="acme",
                        score=0.9,
                        is_pdf=False,
                    ),
                ]
                return WebSearchResponse(
                    results=results,
                    outcome=ToolOutcome(decision="ok", rationale="none", meta={}),
                )

        class AutoIngestHybridExecutor:
            def run(self, *, scoring_context, candidates, tenant_context):
                result = MagicMock()
                ranked = [
                    {"url": "https://docs.acme.test/admin", "score": 95.0},
                    {"url": "https://docs.acme.test/ignore", "score": 40.0},
                ]
                result.model_dump.return_value = {"ranked": ranked}
                return result

        class LocalStubHitlGateway:
            def present(self, payload):
                return None

        cs_module.select_auto_ingest_urls.return_value = [
            "https://docs.acme.test/admin"
        ]

        mock_crawler_manager = MagicMock()
        mock_crawler_manager.dispatch_crawl_request.return_value = {
            "count": 1,
            "run_id": "auto-ingest-run",
        }

        dependencies = {
            "runtime_strategy_generator": LocalStubStrategyGenerator(),
            "runtime_search_worker": StubWebSearchWorker(),
            "runtime_hybrid_executor": AutoIngestHybridExecutor(),
            "runtime_hitl_gateway": LocalStubHitlGateway(),
            "runtime_coverage_verifier": MagicMock(),
            "runtime_crawler_manager": mock_crawler_manager,
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph_adapter = CollectionSearchAdapter(dependencies)

        initial_state = self._initial_state()
        initial_state["input"]["auto_ingest"] = True
        initial_state["input"]["auto_ingest_top_k"] = 1
        initial_state["input"]["auto_ingest_min_score"] = 60.0

        meta = self._tool_context_meta()
        _, result = graph_adapter.run(initial_state, meta=meta)

        assert result["ingestion"]["status"] == "triggered"
        assert result["ingestion"]["task_info"]["run_id"] == "auto-ingest-run"
        mock_crawler_manager.dispatch_crawl_request.assert_called_once()

    def test_boundary_validation_rejects_missing_schema_id(self, cs_module) -> None:
        """Test that missing schema_id raises InvalidGraphInput."""
        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        InvalidGraphInput = cs_module.InvalidGraphInput

        dependencies = {
            "runtime_strategy_generator": MagicMock(),
            "runtime_search_worker": MagicMock(),
            "runtime_hybrid_executor": MagicMock(),
            "runtime_hitl_gateway": MagicMock(),
            "runtime_coverage_verifier": MagicMock(),
        }

        graph_adapter = CollectionSearchAdapter(dependencies)

        # Legacy state without schema_id/schema_version
        legacy_state = {
            "input": {
                "question": "test",
                "collection_scope": "docs",
                "purpose": "test",
            },
        }

        meta = self._tool_context_meta()

        with pytest.raises(InvalidGraphInput) as exc_info:
            graph_adapter.run(legacy_state, meta=meta)

        assert "schema_id" in str(exc_info.value).lower()
        assert "mandatory" in str(exc_info.value).lower()

    def test_boundary_validation_rejects_invalid_schema_version(
        self, cs_module
    ) -> None:
        """Test that invalid schema_version raises InvalidGraphInput."""
        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        InvalidGraphInput = cs_module.InvalidGraphInput

        dependencies = {
            "runtime_strategy_generator": MagicMock(),
            "runtime_search_worker": MagicMock(),
            "runtime_hybrid_executor": MagicMock(),
            "runtime_hitl_gateway": MagicMock(),
            "runtime_coverage_verifier": MagicMock(),
        }

        graph_adapter = CollectionSearchAdapter(dependencies)

        invalid_state = self._initial_state()
        invalid_state["schema_version"] = "99.99.99"  # Wrong version

        meta = self._tool_context_meta()

        with pytest.raises(InvalidGraphInput) as exc_info:
            graph_adapter.run(invalid_state, meta=meta)

        assert "schema_version" in str(exc_info.value).lower()

    def test_boundary_validation_with_tool_context_in_meta(self, cs_module) -> None:
        """Test that tool_context can be provided via meta fallback."""
        from ai_core.tools.web_search import (
            WebSearchResponse,
            ToolOutcome,
        )

        class LocalStubStrategyGenerator:
            def __call__(self, request):
                return cs_module.SearchStrategy(
                    queries=["q"],
                    policies_applied=(),
                    preferred_sources=(),
                    disallowed_sources=(),
                )

        class StubWebSearchWorker:
            def run(self, *, query, context):
                return WebSearchResponse(
                    results=[],
                    outcome=ToolOutcome(decision="ok", rationale="none", meta={}),
                )

        class MockObj:
            def present(self, p):
                return None

            def run(self, **k):
                m = MagicMock()
                m.model_dump.return_value = {}
                m.ranked = []
                m.candidates = {}
                return m

            def verify(self, **k):
                return {}

        dependencies = {
            "runtime_strategy_generator": LocalStubStrategyGenerator(),
            "runtime_search_worker": StubWebSearchWorker(),
            "runtime_hybrid_executor": MockObj(),
            "runtime_hitl_gateway": MockObj(),
            "runtime_coverage_verifier": MockObj(),
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph_adapter = CollectionSearchAdapter(dependencies)

        # State without tool_context (should fallback to meta)
        state = {
            "schema_id": "noesis.graphs.collection_search",
            "schema_version": "1.0.0",
            "input": {
                "question": "test",
                "collection_scope": "docs",
                "purpose": "test",
            },
            # tool_context is missing here
        }

        meta = self._tool_context_meta()
        _, result = graph_adapter.run(state, meta=meta)

        assert result["outcome"] == "completed"

    def test_search_worker_receives_tool_context_not_dict(self, cs_module) -> None:
        """Regression test: search worker must receive ToolContext, not a dict.

        This test ensures that _execute_single_search passes a proper ToolContext
        object to the worker, not a plain dict. Previously, a dict was passed which
        caused 'AttributeError: dict has no attribute metadata' in production.
        """
        from ai_core.tools.web_search import (
            WebSearchResponse,
            ToolOutcome,
            SearchResult,
        )
        from ai_core.tool_contracts import ToolContext

        received_contexts: list[Any] = []

        class ContextCapturingWorker:
            """Worker that captures the context type for verification."""

            def run(self, *, query: str, context: Any) -> WebSearchResponse:
                received_contexts.append(context)
                results = [
                    SearchResult(
                        url="https://test.example/doc",
                        title="Test Doc",
                        snippet="Test snippet",
                        source="test",
                        score=0.9,
                        is_pdf=False,
                    ),
                ]
                outcome = ToolOutcome(
                    decision="ok",
                    rationale="search_completed",
                    meta={"provider": "stub", "latency_ms": 10},
                )
                return WebSearchResponse(results=results, outcome=outcome)

        class LocalStubStrategyGenerator:
            def __call__(self, request: Any) -> Any:
                cs_module.cosine_similarity.return_value = 0.8
                cs_module.calculate_generic_heuristics.return_value = 0.5
                return cs_module.SearchStrategy(
                    queries=["query1", "query2"],
                    policies_applied=(),
                    preferred_sources=(),
                    disallowed_sources=(),
                )

        class MockObj:
            def present(self, p):
                return None

            def run(self, **k):
                m = MagicMock()
                m.model_dump.return_value = {"ranked": []}
                return m

            def verify(self, **k):
                return {}

        dependencies = {
            "runtime_strategy_generator": LocalStubStrategyGenerator(),
            "runtime_search_worker": ContextCapturingWorker(),
            "runtime_hybrid_executor": MockObj(),
            "runtime_hitl_gateway": MockObj(),
            "runtime_coverage_verifier": MockObj(),
        }

        CollectionSearchAdapter = cs_module.CollectionSearchAdapter
        graph = CollectionSearchAdapter(dependencies)

        initial_state = self._initial_state()
        meta = self._tool_context_meta()
        _, result = graph.run(initial_state, meta=meta)

        assert result["outcome"] == "completed"
        assert len(received_contexts) == 2, "Expected 2 search calls (one per query)"

        for i, ctx in enumerate(received_contexts):
            assert isinstance(
                ctx, ToolContext
            ), f"Search call {i}: Expected ToolContext, got {type(ctx).__name__}"
            assert hasattr(
                ctx, "metadata"
            ), f"Search call {i}: ToolContext missing metadata attribute"
            assert (
                "worker_call_id" in ctx.metadata
            ), f"Search call {i}: worker_call_id not set in metadata"
