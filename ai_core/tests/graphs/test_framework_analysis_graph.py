"""Integration tests for framework analysis graph."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.graphs.business.framework_analysis_graph import (
    build_graph,
    FrameworkAnalysisGraph,
)
from ai_core.tools.framework_contracts import FrameworkAnalysisInput


class TestFrameworkAnalysisGraphBuilder:
    """Tests for graph builder and initialization."""

    def test_build_graph(self) -> None:
        """Test that graph can be built."""
        graph = build_graph()
        assert isinstance(graph, FrameworkAnalysisGraph)

    def test_graph_has_correct_nodes(self) -> None:
        """Test that graph has all required nodes."""
        graph = build_graph()
        nodes = graph.build_nodes()

        node_names = [node.name for node in nodes]
        assert "detect_type_and_gremium" in node_names
        assert "extract_toc" in node_names
        assert "locate_components" in node_names
        assert "validate_components" in node_names
        assert "assemble_profile" in node_names
        assert "finish" in node_names

    def test_nodes_are_ordered_correctly(self) -> None:
        """Test that nodes are in correct execution order."""
        graph = build_graph()
        nodes = graph.build_nodes()

        expected_order = [
            "detect_type_and_gremium",
            "extract_toc",
            "locate_components",
            "validate_components",
            "assemble_profile",
            "finish",
        ]

        actual_order = [node.name for node in nodes]
        assert actual_order == expected_order


class TestAssembleProfileNode:
    """Tests for the assemble_profile node logic."""

    def test_assemble_with_all_components_found(self) -> None:
        """Test assembly when all components are found."""
        graph = FrameworkAnalysisGraph()

        state = {
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

        transition, should_continue = graph._assemble_profile(state)

        assert should_continue is True
        assert transition.decision == "profile_assembled"
        assert state["completeness_score"] == 1.0  # All 4 found
        assert state["missing_components"] == []
        assert state["hitl_required"] is False

    def test_assemble_with_missing_component(self) -> None:
        """Test assembly when one component is missing."""
        graph = FrameworkAnalysisGraph()

        state = {
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

        transition, should_continue = graph._assemble_profile(state)

        assert state["completeness_score"] == 0.75  # 3 out of 4
        assert "zugriffsrechte" in state["missing_components"]
        assert state["hitl_required"] is False  # Not required just for missing

    def test_assemble_triggers_hitl_on_low_confidence(self) -> None:
        """Test that HITL is triggered when confidence is below threshold."""
        graph = FrameworkAnalysisGraph()

        state = {
            "confidence_threshold": 0.70,
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
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "gremium_identifier": "TEST_GREMIUM",
        }

        transition, should_continue = graph._assemble_profile(state)

        assert state["hitl_required"] is True
        assert any("Low confidence" in reason for reason in state["hitl_reasons"])
        assert transition.severity == "warning"

    def test_assemble_triggers_hitl_on_failed_validation(self) -> None:
        """Test that HITL is triggered when validation fails."""
        graph = FrameworkAnalysisGraph()

        state = {
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
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "gremium_identifier": "TEST_GREMIUM",
        }

        transition, should_continue = graph._assemble_profile(state)

        assert state["hitl_required"] is True
        assert any("Validation failed" in reason for reason in state["hitl_reasons"])

    def test_assemble_sets_validation_notes(self) -> None:
        """Test that validation notes are correctly set."""
        graph = FrameworkAnalysisGraph()

        state = {
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
            "tenant_id": "test_tenant",
            "trace_id": "test_trace",
            "gremium_identifier": "TEST_GREMIUM",
        }

        transition, should_continue = graph._assemble_profile(state)

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

    @patch("ai_core.graphs.business.framework_analysis_graph.llm_client")
    @patch("ai_core.graphs.business.framework_analysis_graph.retrieve")
    @patch("ai_core.graphs.business.framework_analysis_graph.load_prompt")
    def test_graph_executes_all_nodes(
        self,
        mock_load_prompt: MagicMock,
        mock_retrieve: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Test that graph executes all nodes successfully."""
        # Mock prompt
        mock_load_prompt.return_value = {"text": "Prompt text", "version": "v1"}

        # Mock retrieve responses
        mock_retrieve_output = MagicMock()
        mock_retrieve_output.matches = [
            {
                "text": "Konzernbetriebsvereinbarung IT-Systeme",
                "meta": {
                    "parents": [
                        {
                            "id": "h1",
                            "type": "heading",
                            "title": "§ 1 Präambel",
                            "level": 1,
                            "order": 1,
                        }
                    ]
                },
            }
        ]
        mock_retrieve.run.return_value = mock_retrieve_output

        # Mock LLM responses
        type_detection_response = {
            "text": json.dumps(
                {
                    "agreement_type": "kbv",
                    "type_confidence": 0.95,
                    "gremium_name_raw": "Konzernbetriebsrat",
                    "gremium_identifier_suggestion": "KBR",
                    "evidence": [],
                    "scope_indicators": {},
                }
            )
        }

        location_response = {
            "text": json.dumps(
                {
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
            )
        }

        mock_llm.call.side_effect = [type_detection_response, location_response]

        # Execute graph
        graph = build_graph()
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

    @patch("ai_core.graphs.business.framework_analysis_graph.llm_client")
    @patch("ai_core.graphs.business.framework_analysis_graph.retrieve")
    def test_graph_stops_on_error(
        self, mock_retrieve: MagicMock, mock_llm: MagicMock
    ) -> None:
        """Test that graph stops execution on error."""
        # Make retrieve raise an error
        mock_retrieve.run.side_effect = Exception("Retrieve failed")

        graph = build_graph()
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

        # Should handle error gracefully
        with pytest.raises(Exception):
            graph.run(context=context, input_params=input_params)
