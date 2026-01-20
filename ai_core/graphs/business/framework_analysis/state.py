"""Typed state for framework analysis graph."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, TypedDict, Annotated

from ai_core.tool_contracts import ToolContext


def _merge_dict(
    left: MutableMapping[str, Any] | None, right: Mapping[str, Any] | None
) -> MutableMapping[str, Any]:
    if left is None:
        return dict(right or {})
    if right is None:
        return left
    merged = dict(left)
    merged.update(right)
    return merged


def _append_list(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    if left is None:
        return list(right or [])
    if right is None:
        return left
    return left + list(right)


class FrameworkAnalysisState(TypedDict, total=False):
    """Runtime state for framework analysis."""

    input: Mapping[str, Any]
    context: ToolContext
    retrieval_service: Any
    llm_service: Any
    component_queries: Mapping[str, Mapping[str, str]]
    runtime: Mapping[str, Any]

    tenant_id: str
    tenant_schema: str | None
    trace_id: str
    scope_context: Any
    document_collection_id: str
    document_id: str | None
    force_reanalysis: bool
    confidence_threshold: float

    transitions: Annotated[list[Mapping[str, Any]], _append_list]

    agreement_type: str
    type_confidence: float
    gremium_name_raw: str
    gremium_identifier: str
    evidence: list[Mapping[str, Any]]

    all_chunks: list[Mapping[str, Any]]
    toc: list[Mapping[str, Any]]
    document_preview: str

    located_components: Mapping[str, Any]
    validations: Mapping[str, Any]

    assembled_structure: Mapping[str, Any]
    completeness_score: float
    missing_components: list[str]
    hitl_required: bool
    hitl_reasons: list[str]
    errors: Annotated[list[Mapping[str, Any]], _append_list]


__all__ = ["FrameworkAnalysisState", "_merge_dict", "_append_list"]
