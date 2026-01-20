"""Service boundary for framework analysis persistence."""

from __future__ import annotations

from ai_core.contracts.audit_meta import audit_meta_from_scope
from typing import Any

from ai_core.graph.execution import GraphExecutor, LocalGraphExecutor
from ai_core.tool_contracts import ToolContext
from ai_core.tools.framework_contracts import (
    FrameworkAnalysisDraft,
    FrameworkAnalysisInput,
    FrameworkAnalysisMetadata,
    FrameworkAnalysisOutput,
)
from ai_core.graphs.business.framework_analysis import (
    FRAMEWORK_ANALYSIS_IO_VERSION_STRING,
    FRAMEWORK_ANALYSIS_SCHEMA_ID,
    FrameworkAnalysisGraphInput,
    FrameworkAnalysisGraphOutput,
)
from documents.services.framework_service import persist_profile


def _build_graph_meta(context: ToolContext) -> dict[str, Any]:
    return {
        "tool_context": context.model_dump(mode="json", exclude_none=True),
        "scope_context": context.scope.model_dump(mode="json", exclude_none=True),
        "business_context": context.business.model_dump(mode="json", exclude_none=True),
    }


def run_framework_analysis(
    *,
    context: ToolContext,
    input_params: FrameworkAnalysisInput,
    executor: GraphExecutor | None = None,
) -> FrameworkAnalysisOutput:
    """Execute analysis and persist results via the framework service boundary."""
    graph_request = FrameworkAnalysisGraphInput(
        schema_id=FRAMEWORK_ANALYSIS_SCHEMA_ID,
        schema_version=FRAMEWORK_ANALYSIS_IO_VERSION_STRING,
        input=input_params,
        tool_context=context,
    )
    state = graph_request.model_dump(mode="json")
    meta = _build_graph_meta(context)

    graph_executor = executor or LocalGraphExecutor()
    _, result = graph_executor.run("framework_analysis", state, meta)
    graph_output = FrameworkAnalysisGraphOutput.model_validate(result)
    draft: FrameworkAnalysisDraft = graph_output

    scope = context.scope
    initiated_by_user_id = context.metadata.get("initiated_by_user_id")
    audit_meta = audit_meta_from_scope(
        scope,
        created_by_user_id=scope.user_id,
        initiated_by_user_id=initiated_by_user_id,
    )

    profile = persist_profile(
        tenant_schema=str(scope.tenant_schema or ""),
        gremium_identifier=draft.gremium_identifier,
        gremium_name_raw=draft.gremium_name_raw,
        agreement_type=draft.agreement_type,
        structure=draft.structure.model_dump(mode="json"),
        document_collection_id=draft.document_collection_id,
        document_id=draft.document_id,
        trace_id=scope.trace_id,
        force_reanalysis=draft.force_reanalysis,
        audit_meta=audit_meta.model_dump(mode="json"),
        analysis_metadata=draft.analysis_metadata.model_dump(mode="json"),
        metadata={
            "confidence_threshold": draft.confidence_threshold,
            "hitl_required": draft.hitl_required,
            "hitl_reasons": list(draft.hitl_reasons),
        },
        completeness_score=draft.completeness_score,
        missing_components=draft.missing_components,
    )

    analysis_metadata = FrameworkAnalysisMetadata.model_validate(
        profile.analysis_metadata
    )

    return FrameworkAnalysisOutput(
        profile_id=profile.id,
        version=profile.version,
        gremium_identifier=draft.gremium_identifier,
        structure=draft.structure,
        completeness_score=draft.completeness_score,
        missing_components=draft.missing_components,
        hitl_required=draft.hitl_required,
        hitl_reasons=list(draft.hitl_reasons),
        idempotent=True,
        analysis_metadata=analysis_metadata,
    )
