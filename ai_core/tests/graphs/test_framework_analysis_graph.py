"""Integration tests for framework analysis graph."""

from __future__ import annotations

from typing import Any
import asyncio
from uuid import uuid4

import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.graphs.business.framework_analysis import (
    FrameworkAnalysisGraphInput,
    FrameworkAnalysisGraphOutput,
)
from ai_core.graphs.business.framework_analysis.graph import FrameworkAnalysisStateGraph
from ai_core.graphs.business.framework_analysis.nodes import assemble_profile_node
from ai_core.graphs.business.framework_analysis_graph import (
    build_graph,
    FrameworkAnalysisGraph,
)
from ai_core.tools.framework_contracts import FrameworkAnalysisInput
from pydantic import ValidationError


def _test_context() -> Any:
    return ScopeContext(
        tenant_id="test_tenant",
        tenant_schema="test_schema",
        trace_id="test_trace",
        invocation_id=str(uuid4()),
        run_id="run-test",
        service_id="test-worker",
    ).to_tool_context(
        business=BusinessContext(
            collection_id=str(uuid4()),
            document_id=str(uuid4()),
        )
    )


class TestFrameworkAnalysisGraphBuilder:
    """Tests for graph builder and initialization."""

    def test_build_graph(self) -> None:
        """Test that graph can be built."""
        graph = build_graph()
        assert isinstance(graph, FrameworkAnalysisGraph)

    def test_boundary_requires_schema_version(self) -> None:
        """Test that schema_version is required at the boundary."""
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )

        state = {
            "schema_id": "noesis.graphs.framework_analysis",
            "input": FrameworkAnalysisInput(),
            "tool_context": context,
        }

        with pytest.raises(ValidationError):
            FrameworkAnalysisGraphInput.model_validate(state)

    def test_boundary_accepts_runtime_overrides(self) -> None:
        """Runtime overrides are accepted at the boundary."""
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )

        state = {
            "schema_id": "noesis.graphs.framework_analysis",
            "schema_version": "1.0.0",
            "input": FrameworkAnalysisInput(),
            "tool_context": context,
            "runtime": {"retrieval_service": object(), "llm_service": object()},
        }

        parsed = FrameworkAnalysisGraphInput.model_validate(state)
        assert parsed.runtime is not None


class TestAssembleProfileNode:
    """Tests for the assemble_profile node logic."""

    def test_assemble_with_all_components_found(self) -> None:
        """Test assembly when all components are found."""
        state = {
            "context": _test_context(),
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "confidence_threshold": 0.70,
            "located_components": {
                "systembeschreibung": {
                    "location": "main",
                    "outline_path": "2",
                    "heading": "§ 2 Systembeschreibung",
                    "chunk_ids": ["chunk1"],
                    "page_numbers": [2],
                    "confidence": 0.92,
                },
                "funktionsbeschreibung": {
                    "location": "annex",
                    "outline_path": "Anlage 1",
                    "chunk_ids": ["chunk2"],
                    "page_numbers": [15],
                    "confidence": 0.88,
                },
                "auswertungen": {
                    "location": "annex_group",
                    "outline_path": "Anlage 3",
                    "annex_root": "Anlage 3",
                    "subannexes": ["3.1", "3.2"],
                    "chunk_ids": ["chunk3"],
                    "page_numbers": [25],
                    "confidence": 0.85,
                },
                "zugriffsrechte": {
                    "location": "main",
                    "outline_path": "4",
                    "heading": "§ 4 Zugriffsrechte",
                    "chunk_ids": ["chunk4"],
                    "page_numbers": [10],
                    "confidence": 0.90,
                },
            },
            "validations": {
                "systembeschreibung": {"plausible": True, "confidence": 0.92},
                "funktionsbeschreibung": {"plausible": True, "confidence": 0.88},
                "auswertungen": {"plausible": True, "confidence": 0.85},
                "zugriffsrechte": {"plausible": True, "confidence": 0.90},
            },
        }

        updates = assemble_profile_node(state)
        state.update(updates)

        assert state["completeness_score"] == 1.0  # All 4 found
        assert state["missing_components"] == []
        assert state["hitl_required"] is False

    def test_assemble_with_missing_component(self) -> None:
        """Test assembly when one component is missing."""
        state = {
            "context": _test_context(),
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "confidence_threshold": 0.70,
            "located_components": {
                "systembeschreibung": {
                    "location": "main",
                    "confidence": 0.92,
                },
                "funktionsbeschreibung": {
                    "location": "annex",
                    "confidence": 0.88,
                },
                "auswertungen": {
                    "location": "annex_group",
                    "confidence": 0.85,
                },
                "zugriffsrechte": {
                    "location": "not_found",
                    "confidence": 0.0,
                },
            },
            "validations": {
                "systembeschreibung": {"plausible": True},
                "funktionsbeschreibung": {"plausible": True},
                "auswertungen": {"plausible": True},
                "zugriffsrechte": {"plausible": False},
            },
        }

        updates = assemble_profile_node(state)
        state.update(updates)

        assert state["completeness_score"] == 0.75  # 3 out of 4
        assert "zugriffsrechte" in state["missing_components"]
        assert state["hitl_required"] is False  # Not required just for missing

    def test_assemble_triggers_hitl_on_low_confidence(self) -> None:
        """Test that HITL is triggered when confidence is below threshold."""
        state = {
            "context": _test_context(),
            "confidence_threshold": 0.70,
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "located_components": {
                "systembeschreibung": {
                    "location": "main",
                    "confidence": 0.65,  # Below threshold
                },
                "funktionsbeschreibung": {
                    "location": "annex",
                    "confidence": 0.88,
                },
                "auswertungen": {
                    "location": "annex_group",
                    "confidence": 0.85,
                },
                "zugriffsrechte": {
                    "location": "not_found",
                    "confidence": 0.0,
                },
            },
            "validations": {
                "systembeschreibung": {"plausible": True},
                "funktionsbeschreibung": {"plausible": True},
                "auswertungen": {"plausible": True},
                "zugriffsrechte": {"plausible": False},
            },
            "gremium_identifier": "TEST_GREMIUM",
        }

        updates = assemble_profile_node(state)
        state.update(updates)

        assert state["hitl_required"] is True
        assert any("Low confidence" in reason for reason in state["hitl_reasons"])

    def test_assemble_triggers_hitl_on_failed_validation(self) -> None:
        """Test that HITL is triggered when validation fails."""
        state = {
            "context": _test_context(),
            "confidence_threshold": 0.70,
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "located_components": {
                "systembeschreibung": {
                    "location": "main",
                    "confidence": 0.92,
                },
                "funktionsbeschreibung": {
                    "location": "annex",
                    "confidence": 0.88,
                },
                "auswertungen": {
                    "location": "annex_group",
                    "confidence": 0.85,
                },
                "zugriffsrechte": {
                    "location": "main",
                    "confidence": 0.75,
                },
            },
            "validations": {
                "systembeschreibung": {"plausible": True},
                "funktionsbeschreibung": {
                    "plausible": False,  # Validation failed
                    "reason": "Content doesn't match expected pattern",
                },
                "auswertungen": {"plausible": True},
                "zugriffsrechte": {"plausible": True},
            },
            "gremium_identifier": "TEST_GREMIUM",
        }

        updates = assemble_profile_node(state)
        state.update(updates)

        assert state["hitl_required"] is True
        assert any("Validation failed" in reason for reason in state["hitl_reasons"])

    def test_assemble_sets_validation_notes(self) -> None:
        """Test that validation notes are correctly set."""
        state = {
            "context": _test_context(),
            "confidence_threshold": 0.70,
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "located_components": {
                "systembeschreibung": {
                    "location": "main",
                    "confidence": 0.92,
                },
                "funktionsbeschreibung": {
                    "location": "annex",
                    "confidence": 0.88,
                },
                "auswertungen": {
                    "location": "annex_group",
                    "confidence": 0.85,
                },
                "zugriffsrechte": {
                    "location": "not_found",
                    "confidence": 0.0,
                },
            },
            "validations": {
                "systembeschreibung": {
                    "plausible": True,
                    "reason": "Contains technical specs",
                },
                "funktionsbeschreibung": {
                    "plausible": False,
                    "reason": "Describes legal framework instead",
                },
                "auswertungen": {"plausible": True, "reason": "Lists reports"},
                "zugriffsrechte": {"plausible": False},
            },
            "gremium_identifier": "TEST_GREMIUM",
        }

        updates = assemble_profile_node(state)
        state.update(updates)

        assembled = state["assembled_structure"]
        assert assembled["systembeschreibung"]["validated"] is True
        assert (
            assembled["systembeschreibung"]["validation_notes"]
            == "Contains technical specs"
        )
        assert assembled["funktionsbeschreibung"]["validated"] is False
        assert (
            "legal framework" in assembled["funktionsbeschreibung"]["validation_notes"]
        )


class TestGraphEndToEnd:
    """End-to-end tests for the complete graph."""

    def test_graph_executes_all_nodes(
        self,
    ) -> None:
        """Test that graph executes all nodes successfully."""

        # Mock retrieval graph responses
        class StubRetrievalGraph:
            def invoke(self, state):
                queries = state.get("queries") or []
                if queries == ["document"]:
                    return {
                        "matches": [
                            {
                                "text": "Konzernbetriebsvereinbarung IT-Systeme",
                                "meta": {
                                    "parents": [
                                        {
                                            "id": "h1",
                                            "type": "heading",
                                            "title": "? 1 Praeambel",
                                            "level": 1,
                                            "order": 1,
                                        }
                                    ]
                                },
                            }
                        ]
                    }
                return {
                    "snippets": [
                        {
                            "id": "chunk-1",
                            "text": "Systembeschreibung Architektur Module",
                            "score": 0.9,
                        },
                        {
                            "id": "chunk-2",
                            "text": "Funktionsbeschreibung Features Use Cases",
                            "score": 0.8,
                        },
                    ]
                }

        # Mock LLM responses
        type_detection_response = {
            "agreement_type": "kbv",
            "type_confidence": 0.95,
            "gremium_name_raw": "Konzernbetriebsrat",
            "gremium_identifier_suggestion": "KBR",
            "evidence": [],
            "scope_indicators": {},
        }

        location_response = {
            "systembeschreibung": {
                "location": "main",
                "confidence": 0.9,
            },
            "funktionsbeschreibung": {
                "location": "annex",
                "confidence": 0.85,
            },
            "auswertungen": {
                "location": "annex_group",
                "confidence": 0.8,
            },
            "zugriffsrechte": {
                "location": "not_found",
                "confidence": 0.0,
            },
        }

        llm_calls: list[dict[str, Any]] = []

        def stub_llm(*, prompt_key: str, prompt_input: str, meta: dict[str, Any]):
            llm_calls.append({"prompt_key": prompt_key, "meta": meta})
            if len(llm_calls) == 1:
                return type_detection_response
            return location_response

        # Execute graph
        graph = FrameworkAnalysisGraph(
            retrieval_service=StubRetrievalGraph(),
            llm_service=stub_llm,
        )
        input_params = FrameworkAnalysisInput()
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )

        output = graph.run(context=context, input_params=input_params)

        # Verify output
        assert output.gremium_identifier == "KBR"
        assert output.completeness_score == 0.75
        assert "zugriffsrechte" in output.missing_components
        assert output.analysis_metadata.model_version == "framework_analysis_v1"

    def test_graph_stops_on_error(self) -> None:
        """Test that graph degrades gracefully on error."""

        # Make retrieval graph raise an error
        class StubRetrievalGraph:
            def invoke(self, _state):
                raise Exception("Retrieve failed")

        graph = FrameworkAnalysisGraph(retrieval_service=StubRetrievalGraph())
        input_params = FrameworkAnalysisInput()
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )

        output = graph.run(context=context, input_params=input_params)
        assert output.errors
        assert output.hitl_required is True

    def test_graph_records_errors_on_failure(self) -> None:
        """Test that structured errors are returned on failure."""

        class FailingRetrieval:
            def invoke(self, _state):
                raise Exception("boom")

        def stub_llm(*, prompt_key: str, prompt_input: str, meta: dict[str, Any]):
            return {}

        graph = FrameworkAnalysisGraph(
            retrieval_service=FailingRetrieval(),
            llm_service=stub_llm,
        )
        input_params = FrameworkAnalysisInput()
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )

        output = graph.run(context=context, input_params=input_params)

        assert output.errors
        assert output.hitl_required is True
        assert output.partial_results is not None
        assert output.completeness_score == 0.0

    def test_emit_event_called(self, monkeypatch) -> None:
        """emit_event is called during graph execution."""
        events: list[dict[str, Any]] = []

        def capture_event(payload: dict[str, Any]) -> None:
            events.append(payload)

        monkeypatch.setattr(
            "ai_core.graphs.business.framework_analysis.nodes.emit_event",
            capture_event,
        )
        monkeypatch.setattr(
            "ai_core.graphs.business.framework_analysis.graph.emit_event",
            capture_event,
        )

        class StubRetrievalGraph:
            def invoke(self, state):
                queries = state.get("queries") or []
                if queries == ["document"]:
                    return {"matches": []}
                return {"snippets": []}

        def stub_llm(*, prompt_key: str, prompt_input: str, meta: dict[str, Any]):
            if prompt_key == "framework/detect_type_gremium.v1":
                return {
                    "agreement_type": "kbv",
                    "type_confidence": 0.95,
                    "gremium_name_raw": "Konzernbetriebsrat",
                    "gremium_identifier_suggestion": "KBR",
                    "evidence": [],
                    "scope_indicators": {},
                }
            return {
                "systembeschreibung": {"location": "not_found", "confidence": 0.0},
                "funktionsbeschreibung": {"location": "not_found", "confidence": 0.0},
                "auswertungen": {"location": "not_found", "confidence": 0.0},
                "zugriffsrechte": {"location": "not_found", "confidence": 0.0},
            }

        graph = FrameworkAnalysisStateGraph(
            retrieval_service=StubRetrievalGraph(),
            llm_service=stub_llm,
        )
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )
        state = {
            "schema_id": "noesis.graphs.framework_analysis",
            "schema_version": "1.0.0",
            "input": FrameworkAnalysisInput(),
            "tool_context": context,
        }

        graph.run(state, {})

        assert any(event.get("event") == "framework.graph_started" for event in events)
        assert any(
            event.get("event") == "framework.graph_completed" for event in events
        )

    def test_async_graph_timeout_records_error(self) -> None:
        """Async path records graph timeout errors."""

        class SlowRetrieval:
            def invoke(self, _state):
                import time

                time.sleep(0.05)
                return {"matches": []}

        def stub_llm(*, prompt_key: str, prompt_input: str, meta: dict[str, Any]):
            return {}

        graph = FrameworkAnalysisStateGraph(
            retrieval_service=SlowRetrieval(),
            llm_service=stub_llm,
        )
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )
        state = {
            "schema_id": "noesis.graphs.framework_analysis",
            "schema_version": "1.0.0",
            "input": FrameworkAnalysisInput(),
            "tool_context": context,
            "runtime": {"graph_timeout_s": 0.01},
        }
        _, result = asyncio.run(graph.arun(state, {}))
        output = FrameworkAnalysisGraphOutput.model_validate(result)
        assert output.errors
        assert output.hitl_required is True

    def test_node_timeout_records_error(self) -> None:
        """Test that node timeouts record structured errors."""

        class SlowRetrieval:
            def invoke(self, _state):
                import time

                time.sleep(0.05)
                return {"matches": []}

        def stub_llm(*, prompt_key: str, prompt_input: str, meta: dict[str, Any]):
            return {}

        graph = FrameworkAnalysisStateGraph(
            retrieval_service=SlowRetrieval(),
            llm_service=stub_llm,
        )
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )
        state = {
            "schema_id": "noesis.graphs.framework_analysis",
            "schema_version": "1.0.0",
            "input": FrameworkAnalysisInput(),
            "tool_context": context,
            "runtime": {"node_timeout_s": 0.01},
        }
        _, result = graph.run(state, {})
        output = FrameworkAnalysisGraphOutput.model_validate(result)
        assert output.errors
        assert output.hitl_required is True

    def test_graph_timeout_records_error(self) -> None:
        """Test that graph-level timeout records structured errors."""

        class SlowRetrieval:
            def invoke(self, _state):
                import time

                time.sleep(0.05)
                return {"matches": []}

        def stub_llm(*, prompt_key: str, prompt_input: str, meta: dict[str, Any]):
            return {}

        graph = FrameworkAnalysisStateGraph(
            retrieval_service=SlowRetrieval(),
            llm_service=stub_llm,
        )
        context = ScopeContext(
            tenant_id="test_tenant",
            tenant_schema="test_schema",
            trace_id="test_trace",
            invocation_id=str(uuid4()),
            run_id="run-test",
            service_id="test-worker",
        ).to_tool_context(
            business=BusinessContext(
                collection_id=str(uuid4()),
                document_id=str(uuid4()),
            )
        )
        state = {
            "schema_id": "noesis.graphs.framework_analysis",
            "schema_version": "1.0.0",
            "input": FrameworkAnalysisInput(),
            "tool_context": context,
            "runtime": {"graph_timeout_s": 0.01},
        }
        _, result = graph.run(state, {})
        output = FrameworkAnalysisGraphOutput.model_validate(result)
        assert output.errors
        assert output.hitl_required is True
