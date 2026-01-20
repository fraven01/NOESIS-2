"""LangGraph node functions for framework analysis."""

from __future__ import annotations

from typing import Any, Mapping

from ai_core.graphs.transition_contracts import GraphTransition
from ai_core.infra.observability import observe_span, emit_event
from ai_core.tool_contracts import ToolContext
from ai_core.tools.framework_contracts import ComponentLocation
from ai_core.services.framework_analysis_capabilities import (
    extract_toc_from_chunks,
    normalize_gremium_identifier,
    validate_component_locations,
)

from .state import FrameworkAnalysisState


def _get_ids(context: ToolContext) -> dict[str, Any]:
    scope = context.scope
    business = context.business
    return {
        "tenant_id": scope.tenant_id,
        "trace_id": scope.trace_id,
        "workflow_id": business.workflow_id,
        "case_id": business.case_id,
        "collection_id": business.collection_id,
        "document_id": business.document_id,
        "run_id": scope.run_id,
        "ingestion_run_id": scope.ingestion_run_id,
        "user_id": scope.user_id,
        "service_id": scope.service_id,
    }


def _transition(
    *,
    decision: str,
    reason: str,
    severity: str = "info",
    attributes: Mapping[str, Any] | None = None,
) -> GraphTransition:
    return GraphTransition.from_dict(
        {
            "decision": decision,
            "reason": reason,
            "severity": severity,
            "attributes": dict(attributes or {}),
        }
    )


def _error_payload(node: str, exc: Exception) -> dict[str, Any]:
    return {
        "node": node,
        "message": str(exc),
        "error_type": type(exc).__name__,
    }


def _empty_framework_structure() -> tuple[dict[str, Any], list[str]]:
    components = [
        "systembeschreibung",
        "funktionsbeschreibung",
        "auswertungen",
        "zugriffsrechte",
    ]
    assembled: dict[str, Any] = {}
    for component in components:
        location = ComponentLocation.from_partial({})
        assembled_location, _ = location.to_assembled()
        assembled[component] = assembled_location.model_dump()
    return assembled, components


def _partition_component_chunks(
    component_queries: Mapping[str, str],
    snippets: list[Mapping[str, Any]],
    *,
    per_component: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    def _score_chunk(query: str, chunk: Mapping[str, Any]) -> float:
        text = str(chunk.get("text") or "").lower()
        query_tokens = [tok for tok in query.lower().split() if tok]
        if not query_tokens:
            return 0.0
        hits = sum(1 for tok in query_tokens if tok in text)
        base = float(chunk.get("score") or 0.0)
        return base + hits * 0.05

    component_chunks: dict[str, list[dict[str, Any]]] = {}
    for component, query in component_queries.items():
        ranked = sorted(
            (dict(chunk) for chunk in snippets if isinstance(chunk, Mapping)),
            key=lambda item: (-_score_chunk(query, item), str(item.get("id") or "")),
        )
        component_chunks[component] = ranked[:per_component]
    return component_chunks


@observe_span(name="framework.init_and_fetch")
def init_and_fetch_node(state: FrameworkAnalysisState) -> dict[str, Any]:
    """Fetch initial chunks once and extract ToC."""
    context = state["context"]
    retrieval_service = state["retrieval_service"]
    ids = _get_ids(context)
    init_query = "document"
    retrieval_state = {
        "schema_id": "noesis.graphs.rag_retrieval",
        "schema_version": "1.0.0",
        "tool_context": context,
        "queries": [init_query],
        "retrieve": {
            "query": init_query,
            "filters": (
                {"id": state["document_id"]} if state.get("document_id") else {}
            ),
            "hybrid": {"alpha": 0.0, "top_k": 100},
        },
        "use_rerank": False,
        "document_id": state.get("document_id"),
    }

    retrieval_result = retrieval_service.invoke(retrieval_state)
    if "error" in retrieval_result:
        raise ValueError(retrieval_result["error"])

    all_chunks = retrieval_result.get("matches", [])
    toc = extract_toc_from_chunks(all_chunks)

    document_text = "\n".join(chunk.get("text", "")[:1000] for chunk in all_chunks[:2])[
        :2000
    ]

    transition = _transition(
        decision="init_complete",
        reason=f"Fetched {len(all_chunks)} chunks; ToC {len(toc)} entries",
        attributes={"chunk_count": len(all_chunks), "toc_entries": len(toc)},
    )

    emit_event(
        {
            "event": "framework.init_and_fetch",
            "tenant_id": ids["tenant_id"],
            "trace_id": ids["trace_id"],
            "chunk_count": len(all_chunks),
            "toc_entries": len(toc),
        }
    )

    return {
        "all_chunks": all_chunks,
        "toc": toc,
        "document_preview": document_text,
        "transitions": [transition.to_dict()],
    }


@observe_span(name="framework.detect_type_and_gremium")
def detect_type_node(state: FrameworkAnalysisState) -> dict[str, Any]:
    """Detect framework type (KBV/GBV/BV) and extract gremium."""
    document_text = state.get("document_preview", "")
    llm_service = state["llm_service"]
    ids = _get_ids(state["context"])

    prompt_input = f"Dokument (Anfang):\n{document_text}"
    meta = {
        "tenant_id": ids["tenant_id"],
        "trace_id": ids["trace_id"],
    }
    detection_result = llm_service(
        prompt_key="framework/detect_type_gremium.v1",
        prompt_input=prompt_input,
        meta=meta,
    )

    gremium_identifier = normalize_gremium_identifier(
        detection_result.get("gremium_identifier_suggestion", ""),
        detection_result.get("gremium_name_raw", ""),
    )

    agreement_type = detection_result.get("agreement_type", "other")
    type_confidence = detection_result.get("type_confidence", 0.0)

    updates: dict[str, Any] = {
        "agreement_type": agreement_type,
        "type_confidence": type_confidence,
        "gremium_identifier": gremium_identifier,
        "gremium_name_raw": detection_result.get("gremium_name_raw", ""),
        "evidence": detection_result.get("evidence", []),
    }

    if type_confidence < 0.5:
        empty_structure, missing_components = _empty_framework_structure()
        updates.update(
            {
                "assembled_structure": empty_structure,
                "completeness_score": 0.0,
                "missing_components": missing_components,
                "hitl_required": True,
                "hitl_reasons": ["type_confidence_below_threshold"],
                "early_exit": True,
            }
        )

        transition = _transition(
            decision="early_exit_low_confidence",
            reason="Type confidence below threshold",
            severity="warning",
            attributes={
                "agreement_type": agreement_type,
                "type_confidence": type_confidence,
            },
        )
        emit_event(
            {
                "event": "framework.detect_type.early_exit",
                "tenant_id": ids["tenant_id"],
                "trace_id": ids["trace_id"],
                "agreement_type": agreement_type,
                "type_confidence": type_confidence,
            }
        )
    else:
        transition = _transition(
            decision="type_detected",
            reason=f"Type: {agreement_type}, Gremium: {gremium_identifier}",
            attributes={
                "agreement_type": agreement_type,
                "gremium_identifier": gremium_identifier,
            },
        )
        emit_event(
            {
                "event": "framework.detect_type",
                "tenant_id": ids["tenant_id"],
                "trace_id": ids["trace_id"],
                "agreement_type": agreement_type,
                "type_confidence": type_confidence,
            }
        )

    updates["transitions"] = [transition.to_dict()]
    return updates


@observe_span(name="framework.locate_components")
def locate_components_node(state: FrameworkAnalysisState) -> dict[str, Any]:
    """Locate the four components using hybrid search."""
    context = state["context"]
    retrieval_service = state["retrieval_service"]
    llm_service = state["llm_service"]
    ids = _get_ids(context)

    agreement_type = state.get("agreement_type", "other")
    component_queries = state["component_queries"].get(
        agreement_type, state["component_queries"]["other"]
    )

    retrieval_state = {
        "schema_id": "noesis.graphs.rag_retrieval",
        "schema_version": "1.0.0",
        "tool_context": context,
        "queries": list(component_queries.values()),
        "retrieve": {
            "query": "",
            "filters": (
                {"id": state["document_id"]} if state.get("document_id") else {}
            ),
            "hybrid": {"alpha": 0.7, "top_k": 40},
        },
        "use_rerank": True,
        "document_id": state.get("document_id"),
    }

    retrieval_result = retrieval_service.invoke(retrieval_state)
    if "error" in retrieval_result:
        raise ValueError(retrieval_result["error"])

    base_snippets = retrieval_result.get("snippets", [])
    component_chunks = _partition_component_chunks(component_queries, base_snippets)

    toc_text = "\n".join(
        f"{'  ' * entry.get('level', 0)}{entry.get('title', '')}"
        for entry in state.get("toc", [])[:50]
    )

    chunks_text = ""
    for component, chunks in component_chunks.items():
        chunks_text += f"\n\n## {component}:\n"
        for i, chunk in enumerate(chunks[:5]):
            chunks_text += f"[{i+1}] {chunk.get('text', '')[:300]}...\n"

    prompt_input = f"## ToC:\n{toc_text}\n{chunks_text}"
    meta = {
        "tenant_id": ids["tenant_id"],
        "trace_id": ids["trace_id"],
    }
    locations_result = llm_service(
        prompt_key="framework/locate_components.v1",
        prompt_input=prompt_input,
        meta=meta,
    )

    validations = validate_component_locations(
        locations_result,
        high_confidence_threshold=0.8,
    )

    found_count = sum(
        1 for comp in locations_result.values() if comp.get("location") != "not_found"
    )

    transition = _transition(
        decision="components_located",
        reason=f"Located: {found_count}/4 found",
        attributes={"found_components": found_count},
    )

    emit_event(
        {
            "event": "framework.locate_components",
            "tenant_id": ids["tenant_id"],
            "trace_id": ids["trace_id"],
            "found_components": found_count,
        }
    )

    return {
        "located_components": locations_result,
        "validations": validations,
        "transitions": [transition.to_dict()],
    }


@observe_span(name="framework.assemble_profile")
def assemble_profile_node(state: FrameworkAnalysisState) -> dict[str, Any]:
    """Assemble final profile."""
    confidence_threshold = state["confidence_threshold"]
    ids = _get_ids(state["context"])

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

        resolved = ComponentLocation.from_partial(located)
        assembled_location, validation_failed = resolved.to_assembled(
            validation=validated
        )

        if not resolved.is_found():
            missing.append(component)
            assembled[component] = assembled_location.model_dump()
            continue

        if validation_failed:
            hitl_required = True
            hitl_reasons.append(f"{component}: Validation failed")

        if resolved.is_low_confidence(confidence_threshold):
            hitl_required = True
            hitl_reasons.append(f"{component}: Low confidence")

        assembled[component] = assembled_location.model_dump()

    completeness_score = (
        len([c for c in assembled.values() if c["location"] != "not_found"]) / 4.0
    )

    transition = _transition(
        decision="profile_assembled",
        reason=f"{completeness_score:.0%} complete",
        severity="warning" if hitl_required else "info",
    )

    emit_event(
        {
            "event": "framework.assemble_profile",
            "tenant_id": ids["tenant_id"],
            "trace_id": ids["trace_id"],
            "completeness_score": completeness_score,
            "hitl_required": hitl_required,
        }
    )

    return {
        "assembled_structure": assembled,
        "completeness_score": completeness_score,
        "missing_components": missing,
        "hitl_required": hitl_required,
        "hitl_reasons": hitl_reasons,
        "transitions": [transition.to_dict()],
    }


__all__ = [
    "init_and_fetch_node",
    "detect_type_node",
    "locate_components_node",
    "assemble_profile_node",
]
