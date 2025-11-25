"""Framework agreement analysis graph for AI-first structure detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Tuple
from uuid import UUID, uuid4

from django.db import transaction

from ai_core.graphs.transition_contracts import GraphTransition
from ai_core.infra.observability import observe_span
from ai_core.infra.prompts import load as load_prompt
from ai_core.llm import client as llm_client
from ai_core.nodes import retrieve
from ai_core.tool_contracts import ToolContext
from ai_core.tools.framework_contracts import (
    FrameworkAnalysisInput,
    FrameworkAnalysisMetadata,
    FrameworkAnalysisOutput,
    FrameworkStructure,
)
from common.logging import get_logger
from customers.models import Tenant
from documents.framework_models import FrameworkProfile, FrameworkDocument
from documents.models import DocumentCollection

logger = get_logger(__name__)

StateMapping = Dict[str, Any]


@dataclass(frozen=True)
class GraphNode:
    """Tie a node name to an execution callable."""

    name: str
    runner: Callable[[Dict[str, Any]], Tuple[GraphTransition, bool]]

    def execute(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        return self.runner(state)


def _transition(
    *,
    decision: str,
    reason: str,
    severity: str = "info",
    attributes: Dict[str, Any] | None = None,
) -> GraphTransition:
    """Helper to create a standard transition."""
    return GraphTransition.from_dict(
        {
            "decision": decision,
            "reason": reason,
            "severity": severity,
            "attributes": attributes or {},
        }
    )


def normalize_gremium_identifier(suggestion: str, raw_name: str) -> str:
    """
    Normalize gremium identifier for database storage.

    Examples:
    - "Konzernbetriebsrat" → "KBR"
    - "Gesamtbetriebsrat München" → "GBR_MUENCHEN"
    - "Betriebsrat Berlin" → "BR_BERLIN"
    """
    normalized = suggestion.upper()
    # Replace umlauts
    normalized = normalized.replace("Ü", "UE").replace("Ä", "AE").replace("Ö", "OE")
    normalized = normalized.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe")
    # Replace special chars with underscore
    normalized = re.sub(r"[^A-Z0-9_]", "_", normalized)
    # Remove consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized


def extract_toc_from_chunks(chunks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Extract table of contents from parent node metadata."""
    parent_map = {}

    for chunk in chunks:
        parents = chunk.get("meta", {}).get("parents", [])
        for parent in parents:
            parent_id = parent.get("id")
            if parent_id and parent_id not in parent_map:
                parent_type = parent.get("type", "")
                # Only include structural elements
                if parent_type in {"heading", "section", "article", "document"}:
                    parent_map[parent_id] = {
                        "id": parent_id,
                        "title": parent.get("title", ""),
                        "type": parent_type,
                        "level": parent.get("level", 0),
                        "order": parent.get("order", 0),
                    }

    # Sort by level and order
    toc_entries = sorted(
        parent_map.values(), key=lambda p: (p.get("level", 0), p.get("order", 0))
    )

    return toc_entries


class FrameworkAnalysisGraph:
    """
    Graph orchestrating framework agreement analysis.

    Node sequence:
    detect_type_and_gremium → extract_toc → locate_components →
    validate_components → assemble_profile → persist_profile → finish
    """

    def __init__(self) -> None:
        """Initialize graph with default dependencies."""
        pass

    def build_nodes(self) -> list[GraphNode]:
        """Build the node sequence for framework analysis."""
        return [
            GraphNode("detect_type_and_gremium", self._detect_type_and_gremium),
            GraphNode("extract_toc", self._extract_toc),
            GraphNode("locate_components", self._locate_components),
            GraphNode("validate_components", self._validate_components),
            GraphNode("assemble_profile", self._assemble_profile),
            GraphNode("persist_profile", self._persist_profile),
            GraphNode("finish", self._finish),
        ]

    def run(
        self,
        input_params: FrameworkAnalysisInput,
        tenant_id: str,
        tenant_schema: str,
        trace_id: str,
    ) -> FrameworkAnalysisOutput:
        """Execute the framework analysis graph."""
        logger.info(
            "framework_graph_starting",
            extra={
                "tenant_id": tenant_id,
                "tenant_schema": tenant_schema,
                "trace_id": trace_id,
                "document_collection_id": str(input_params.document_collection_id),
                "document_id": (
                    str(input_params.document_id) if input_params.document_id else None
                ),
                "force_reanalysis": input_params.force_reanalysis,
            },
        )

        # Initialize state
        state: StateMapping = {
            "input": input_params.model_dump(),
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "trace_id": trace_id,
            "document_collection_id": str(input_params.document_collection_id),
            "document_id": (
                str(input_params.document_id) if input_params.document_id else None
            ),
            "force_reanalysis": input_params.force_reanalysis,
            "confidence_threshold": input_params.confidence_threshold,
            "transitions": [],
        }

        # Execute nodes sequentially
        nodes = self.build_nodes()
        for node in nodes:
            transition, should_continue = node.execute(state)
            state["transitions"].append(transition.model_dump())

            if not should_continue:
                break

        # Build output from state
        output = FrameworkAnalysisOutput(
            profile_id=state["profile_id"],
            version=state["version"],
            gremium_identifier=state["gremium_identifier"],
            structure=FrameworkStructure(**state["assembled_structure"]),
            completeness_score=state["completeness_score"],
            missing_components=state["missing_components"],
            hitl_required=state["hitl_required"],
            hitl_reasons=state.get("hitl_reasons", []),
            analysis_metadata=FrameworkAnalysisMetadata(**state["analysis_metadata"]),
        )

        logger.info(
            "framework_graph_completed",
            extra={
                "tenant_id": tenant_id,
                "trace_id": trace_id,
                "profile_id": str(output.profile_id),
                "gremium_identifier": output.gremium_identifier,
                "version": output.version,
                "completeness_score": output.completeness_score,
                "hitl_required": output.hitl_required,
                "nodes_executed": len(state["transitions"]),
            },
        )

        return output

    @observe_span(name="framework.detect_type_and_gremium")
    def _detect_type_and_gremium(
        self, state: StateMapping
    ) -> Tuple[GraphTransition, bool]:
        """Detect framework type (KBV/GBV/BV) and extract gremium."""
        try:
            context = ToolContext(
                tenant_id=state["tenant_id"],
                trace_id=state["trace_id"],
                invocation_id=uuid4(),
                run_id=uuid4(),
            )

            # Fetch first chunks
            retrieve_params = retrieve.RetrieveInput(
                query="",
                filters=(
                    {"id": state["document_id"]} if state.get("document_id") else {}
                ),
                collection_id=state.get("document_collection_id"),
                hybrid={"alpha": 0.0, "top_k": 3},
            )

            retrieve_output = retrieve.run(context, retrieve_params)

            document_text = "\n".join(
                chunk.get("text", "")[:1000] for chunk in retrieve_output.matches[:2]
            )[:2000]

            # Load prompt and call LLM
            prompt = load_prompt("framework/detect_type_gremium.v1")
            full_prompt = f"{prompt['text']}\n\nDokument (Anfang):\n{document_text}"

            meta = {
                "tenant_id": state["tenant_id"],
                "trace_id": state["trace_id"],
                "prompt_version": prompt["version"],
            }

            llm_result = llm_client.call("analyze", full_prompt, meta)

            # Parse JSON
            try:
                detection_result = json.loads(llm_result["text"])
            except json.JSONDecodeError:
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", llm_result["text"], re.DOTALL
                )
                if json_match:
                    detection_result = json.loads(json_match.group(1))
                else:
                    raise ValueError("LLM response is not valid JSON")

            # Normalize gremium
            gremium_identifier = normalize_gremium_identifier(
                detection_result.get("gremium_identifier_suggestion", ""),
                detection_result.get("gremium_name_raw", ""),
            )

            state["agreement_type"] = detection_result.get("agreement_type", "other")
            state["type_confidence"] = detection_result.get("type_confidence", 0.0)
            state["gremium_identifier"] = gremium_identifier
            state["gremium_name_raw"] = detection_result.get("gremium_name_raw", "")
            state["evidence"] = detection_result.get("evidence", [])

            return (
                _transition(
                    decision="type_detected",
                    reason=f"Type: {state['agreement_type']}, Gremium: {gremium_identifier}",
                    attributes={
                        "agreement_type": state["agreement_type"],
                        "gremium_identifier": gremium_identifier,
                    },
                ),
                True,
            )

        except Exception as e:
            return (
                _transition(
                    decision="type_detection_failed",
                    reason=f"Failed: {str(e)}",
                    severity="error",
                ),
                False,
            )

    @observe_span(name="framework.extract_toc")
    def _extract_toc(self, state: StateMapping) -> Tuple[GraphTransition, bool]:
        """Extract table of contents from parent nodes."""
        try:
            context = ToolContext(
                tenant_id=state["tenant_id"],
                trace_id=state["trace_id"],
                invocation_id=uuid4(),
                run_id=uuid4(),
            )

            retrieve_params = retrieve.RetrieveInput(
                query="",
                filters=(
                    {"id": state["document_id"]} if state.get("document_id") else {}
                ),
                collection_id=state.get("document_collection_id"),
                hybrid={"alpha": 0.0, "top_k": 100},
            )

            retrieve_output = retrieve.run(context, retrieve_params)
            toc = extract_toc_from_chunks(retrieve_output.matches)

            state["toc"] = toc
            state["all_chunks"] = retrieve_output.matches

            return (
                _transition(
                    decision="toc_extracted",
                    reason=f"ToC extracted: {len(toc)} entries",
                    attributes={"toc_entries": len(toc)},
                ),
                True,
            )

        except Exception as e:
            return (
                _transition(
                    decision="toc_extraction_failed",
                    reason=f"Failed: {str(e)}",
                    severity="error",
                ),
                False,
            )

    @observe_span(name="framework.locate_components")
    def _locate_components(self, state: StateMapping) -> Tuple[GraphTransition, bool]:
        """Locate the four components using hybrid search."""
        try:
            context = ToolContext(
                tenant_id=state["tenant_id"],
                trace_id=state["trace_id"],
                invocation_id=uuid4(),
                run_id=uuid4(),
            )

            # Semantic search per component
            component_queries = {
                "systembeschreibung": "Systembeschreibung technische Beschreibung",
                "funktionsbeschreibung": "Funktionsbeschreibung Funktionen",
                "auswertungen": "Auswertungen Berichte Reports",
                "zugriffsrechte": "Zugriffsrechte Berechtigungen Rollen",
            }

            component_chunks = {}
            for component, query in component_queries.items():
                retrieve_params = retrieve.RetrieveInput(
                    query=query,
                    filters=(
                        {"id": state["document_id"]} if state.get("document_id") else {}
                    ),
                    collection_id=state.get("document_collection_id"),
                    hybrid={"alpha": 0.7, "top_k": 10},
                )
                retrieve_output = retrieve.run(context, retrieve_params)
                component_chunks[component] = retrieve_output.matches[:10]

            # Call LLM
            prompt = load_prompt("framework/locate_components.v1")

            toc_text = "\n".join(
                f"{'  ' * entry.get('level', 0)}{entry.get('title', '')}"
                for entry in state.get("toc", [])[:50]
            )

            chunks_text = ""
            for component, chunks in component_chunks.items():
                chunks_text += f"\n\n## {component}:\n"
                for i, chunk in enumerate(chunks[:5]):
                    chunks_text += f"[{i+1}] {chunk.get('text', '')[:300]}...\n"

            full_prompt = f"{prompt['text']}\n\n## ToC:\n{toc_text}\n{chunks_text}"

            meta = {
                "tenant_id": state["tenant_id"],
                "trace_id": state["trace_id"],
                "prompt_version": prompt["version"],
            }

            llm_result = llm_client.call("analyze", full_prompt, meta)

            # Parse JSON
            try:
                locations_result = json.loads(llm_result["text"])
            except json.JSONDecodeError:
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", llm_result["text"], re.DOTALL
                )
                if json_match:
                    locations_result = json.loads(json_match.group(1))
                else:
                    raise ValueError("LLM response is not valid JSON")

            state["located_components"] = locations_result

            found_count = sum(
                1
                for comp in locations_result.values()
                if comp.get("location") != "not_found"
            )

            return (
                _transition(
                    decision="components_located",
                    reason=f"Located: {found_count}/4 found",
                    attributes={"found_components": found_count},
                ),
                True,
            )

        except Exception as e:
            return (
                _transition(
                    decision="location_failed",
                    reason=f"Failed: {str(e)}",
                    severity="error",
                ),
                False,
            )

    @observe_span(name="framework.validate_components")
    def _validate_components(self, state: StateMapping) -> Tuple[GraphTransition, bool]:
        """Validate component locations."""
        # Simplified validation based on confidence
        validations = {}
        for component, location in state.get("located_components", {}).items():
            confidence = location.get("confidence", 0.0)
            found = location.get("location") != "not_found"

            if not found:
                validations[component] = {
                    "plausible": False,
                    "confidence": 0.0,
                    "reason": "Not found",
                }
            elif confidence >= 0.8:
                validations[component] = {
                    "plausible": True,
                    "confidence": confidence,
                    "reason": "High confidence",
                }
            else:
                validations[component] = {
                    "plausible": True,
                    "confidence": confidence,
                    "reason": "Moderate confidence",
                }

        state["validations"] = validations

        plausible_count = sum(1 for v in validations.values() if v.get("plausible"))

        return (
            _transition(
                decision="components_validated",
                reason=f"Validated: {plausible_count}/4 plausible",
                attributes={"plausible_components": plausible_count},
            ),
            True,
        )

    @observe_span(name="framework.assemble_profile")
    def _assemble_profile(self, state: StateMapping) -> Tuple[GraphTransition, bool]:
        """Assemble final profile."""
        CONFIDENCE_THRESHOLD = state["confidence_threshold"]

        assembled = {}
        missing = []
        hitl_required = False
        hitl_reasons = []

        components = [
            "systembeschreibung",
            "funktionsbeschreibung",
            "auswertungen",
            "zugriffsrechte",
        ]

        for component in components:
            located = state["located_components"].get(component, {})
            validated = state["validations"].get(component, {})

            if located.get("location") == "not_found":
                missing.append(component)
                assembled[component] = {
                    "location": "not_found",
                    "outline_path": None,
                    "heading": None,
                    "chunk_ids": [],
                    "page_numbers": [],
                    "confidence": 0.0,
                    "validated": False,
                    "validation_notes": "Not found",
                }
                continue

            merged = {
                "location": located.get("location"),
                "outline_path": located.get("outline_path"),
                "heading": located.get("heading"),
                "chunk_ids": located.get("chunk_ids", []),
                "page_numbers": located.get("page_numbers", []),
                "confidence": located.get("confidence", 0.0),
                "annex_root": located.get("annex_root"),
                "subannexes": located.get("subannexes", []),
            }

            if not validated.get("plausible", True):
                hitl_required = True
                hitl_reasons.append(f"{component}: Validation failed")
                merged["validated"] = False
                merged["validation_notes"] = validated.get("reason", "")
            else:
                merged["validated"] = True
                merged["validation_notes"] = validated.get("reason", "Plausible")

            if merged["confidence"] < CONFIDENCE_THRESHOLD:
                hitl_required = True
                hitl_reasons.append(f"{component}: Low confidence")

            assembled[component] = merged

        completeness_score = (
            len([c for c in assembled.values() if c["location"] != "not_found"]) / 4.0
        )

        state["assembled_structure"] = assembled
        state["completeness_score"] = completeness_score
        state["missing_components"] = missing
        state["hitl_required"] = hitl_required
        state["hitl_reasons"] = hitl_reasons

        if hitl_required:
            logger.warning(
                "framework_hitl_required",
                extra={
                    "tenant_id": state["tenant_id"],
                    "trace_id": state["trace_id"],
                    "gremium_identifier": state["gremium_identifier"],
                    "completeness_score": completeness_score,
                    "hitl_reasons": hitl_reasons,
                    "missing_components": missing,
                },
            )

        return (
            _transition(
                decision="profile_assembled",
                reason=f"{completeness_score:.0%} complete",
                severity="warning" if hitl_required else "info",
            ),
            True,
        )

    @observe_span(name="framework.persist_profile")
    def _persist_profile(self, state: StateMapping) -> Tuple[GraphTransition, bool]:
        """Persist FrameworkProfile to database."""
        tenant_schema = state["tenant_schema"]
        gremium_identifier = state["gremium_identifier"]
        force_reanalysis = state["force_reanalysis"]

        # Get tenant object
        try:
            tenant = Tenant.objects.get(schema_name=tenant_schema)
        except Tenant.DoesNotExist:
            raise ValueError(f"Tenant not found: {tenant_schema}")

        # Build analysis metadata
        analysis_metadata = {
            "detected_type": state["agreement_type"],
            "type_confidence": state["type_confidence"],
            "gremium_name_raw": state["gremium_name_raw"],
            "gremium_identifier": gremium_identifier,
            "completeness_score": state["completeness_score"],
            "missing_components": state["missing_components"],
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "framework_analysis_v1",
        }

        # Check for existing profile
        with transaction.atomic():
            existing_profile = FrameworkProfile.objects.filter(
                tenant=tenant,
                gremium_identifier=gremium_identifier,
                is_current=True,
            ).first()

            if existing_profile and not force_reanalysis:
                # Profile already exists and not forcing reanalysis
                raise ValueError(
                    f"Framework profile already exists for {gremium_identifier}. "
                    f"Use force_reanalysis=true to create new version."
                )

            # Determine version
            if existing_profile:
                # Set old profile to not current and increment version
                existing_profile.is_current = False
                existing_profile.save(update_fields=["is_current", "updated_at"])
                version = existing_profile.version + 1
            else:
                version = 1

            # Get document collection
            collection_id = state["document_collection_id"]
            try:
                doc_collection = DocumentCollection.objects.get(
                    id=UUID(collection_id),
                    tenant=tenant,
                )
            except DocumentCollection.DoesNotExist:
                raise ValueError(f"DocumentCollection not found: {collection_id}")

            # Generate name from gremium identifier and type
            agreement_type = state["agreement_type"]
            name = f"{agreement_type.upper()} {state['gremium_name_raw']}"

            # Create new profile
            profile = FrameworkProfile.objects.create(
                tenant=tenant,
                name=name,
                agreement_type=agreement_type,
                gremium_identifier=gremium_identifier,
                gremium_name_raw=state["gremium_name_raw"],
                version=version,
                is_current=True,
                external_id=f"{gremium_identifier}_v{version}",
                source_document_collection=doc_collection,
                source_document_id=(
                    UUID(state["document_id"]) if state.get("document_id") else None
                ),
                structure=state["assembled_structure"],
                analysis_metadata=analysis_metadata,
                metadata={
                    "trace_id": state["trace_id"],
                    "confidence_threshold": state["confidence_threshold"],
                    "hitl_required": state["hitl_required"],
                    "hitl_reasons": state["hitl_reasons"],
                },
            )

            # Create document link for main document
            if state.get("document_id"):
                FrameworkDocument.objects.create(
                    tenant=tenant,
                    profile=profile,
                    document_collection=doc_collection,
                    document_id=UUID(state["document_id"]),
                    document_type=FrameworkDocument.DocumentType.MAIN,
                    position=0,
                    analysis={
                        "completeness_score": state["completeness_score"],
                        "missing_components": state["missing_components"],
                    },
                )

            logger.info(
                "framework_profile_persisted",
                extra={
                    "tenant_id": state["tenant_id"],
                    "tenant_schema": tenant_schema,
                    "trace_id": state["trace_id"],
                    "profile_id": str(profile.id),
                    "gremium_identifier": gremium_identifier,
                    "version": version,
                    "is_new_version": existing_profile is not None,
                    "agreement_type": agreement_type,
                },
            )

        # Update state with persisted data
        state["profile_id"] = profile.id
        state["version"] = version
        state["analysis_metadata"] = analysis_metadata

        return (
            _transition(
                decision="profile_persisted",
                reason=f"Persisted {gremium_identifier} v{version}",
            ),
            True,
        )

    def _finish(self, state: StateMapping) -> Tuple[GraphTransition, bool]:
        """Finalize graph execution."""
        return (
            _transition(
                decision="finished",
                reason="Analysis completed",
            ),
            False,
        )


def build_graph() -> FrameworkAnalysisGraph:
    """Build and return a framework analysis graph instance."""
    return FrameworkAnalysisGraph()
