from __future__ import annotations

import inspect
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Mapping, Optional

from celery import shared_task

from ai_core.graphs.technical.document_service import normalize_raw_reference
from ai_core.ids.http_scope import normalize_task_context
from ai_core.ingestion_orchestration import (
    IngestionContextBuilder,
    ObservabilityWrapper,
)
from ai_core.infra import observability as observability_helpers
from documents.contracts import NormalizedDocument
from common.celery import RetryableTask

from .helpers.task_utils import (
    _jsonify_for_task,
    _resolve_trace_context,
    _task_result,
)
from ai_core.graph import registry
from ai_core.tool_contracts import tool_context_from_meta


def _resolve_event_emitter(meta: Optional[Mapping[str, Any]] = None):
    if isinstance(meta, MappingABC):
        candidate = meta.get("ingestion_event_emitter")
        if callable(candidate):
            return candidate
    return None


def _callable_accepts_kwarg(func: Any, keyword: str) -> bool:
    """Return True when the callable supports the provided keyword argument."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - non introspectable callables
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _build_ingestion_graph(event_emitter: Optional[Any]):
    """Invoke build_universal_ingestion_graph."""
    from ai_core.graphs.technical.universal_ingestion_graph import (
        build_universal_ingestion_graph,
    )

    return build_universal_ingestion_graph()


@shared_task(
    base=RetryableTask,
    queue="ingestion",
    name="ai_core.tasks.run_ingestion_graph",
    soft_time_limit=1740,  # 29 minutes
)
def run_ingestion_graph(
    state: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the crawler ingestion LangGraph orchestration.

    Refactored from 127-line function with defensive glue to use:
    - IngestionContextBuilder: Extract metadata from nested dicts
    - ObservabilityWrapper: Manage tracing lifecycle
    - Cleaner orchestration flow
    """
    # 1. Build infrastructure components
    event_emitter = _resolve_event_emitter(meta)
    graph = _build_ingestion_graph(event_emitter)
    trace_context = _resolve_trace_context(state, meta)

    # 2. Extract ingestion context (defensive metadata extraction)
    context_builder = IngestionContextBuilder()
    ingestion_ctx = context_builder.build_from_state(
        state=state,
        meta=meta,
        trace_context=trace_context,
    )

    # 3. Normalize document input if needed
    # Note: Upload worker provides this, Crawler might not if legacy.
    # But for Universal Graph, we prefer normalized input.
    working_state = _prepare_working_state(
        state=state,
        ingestion_ctx=ingestion_ctx,
        trace_context=trace_context,
    )

    # 4. Setup observability (tracing)
    obs_wrapper = ObservabilityWrapper(observability_helpers)
    task_request = getattr(run_ingestion_graph, "request", None)
    obs_ctx = obs_wrapper.create_context(
        ingestion_ctx=ingestion_ctx,
        trace_context=trace_context,
        task_request=task_request,
    )

    obs_wrapper.start_trace(obs_ctx)

    try:
        # 5. Execute Universal Graph
        # Map legacy/worker state to UniversalIngestionInput

        # Extract normalized document (it might be a dict or NormalizedDocument object)
        normalized_doc = working_state.get("normalized_document_input")
        if hasattr(normalized_doc, "model_dump"):
            normalized_doc = normalized_doc.model_dump(mode="json")

        # Extract collection_id
        collection_id = None
        if isinstance(normalized_doc, dict):
            ref = normalized_doc.get("ref", {})
            collection_id = ref.get("collection_id")

        if not collection_id:
            collection_id = ingestion_ctx.collection_id

        input_payload = {
            "normalized_document": normalized_doc,
        }

        # BREAKING CHANGE (Option A): Business IDs extracted separately
        # They are NO LONGER part of ScopeContext after Phase 3 completion
        case_id = ingestion_ctx.case_id or "general"
        workflow_id = ingestion_ctx.workflow_id

        # Build Context via normalize_task_context (Pre-MVP ID Contract)
        # S2S Hop: service_id REQUIRED, user_id ABSENT
        scope = normalize_task_context(
            tenant_id=ingestion_ctx.tenant_id,
            service_id="celery-ingestion-worker",
            trace_id=ingestion_ctx.trace_id,
            invocation_id=getattr(ingestion_ctx, "invocation_id", None)
            or trace_context.get("invocation_id")
            or (meta.get("invocation_id") if meta else None),
            run_id=ingestion_ctx.run_id,
            ingestion_run_id=ingestion_ctx.ingestion_run_id,
        )

        # BREAKING CHANGE (Option A - Full ToolContext Migration):
        # Universal ingestion graph now expects nested ToolContext structure
        from ai_core.contracts.business import BusinessContext

        business = BusinessContext(
            case_id=case_id,
            workflow_id=workflow_id,
            collection_id=collection_id,
        )

        audit_meta = dict((meta or {}).get("audit_meta") or {})
        initiated_by_user_id = (meta or {}).get("initiated_by_user_id")
        if initiated_by_user_id and "initiated_by_user_id" not in audit_meta:
            audit_meta["initiated_by_user_id"] = initiated_by_user_id
        run_context = {
            "scope": scope.model_dump(mode="json"),
            "business": business.model_dump(mode="json"),
            "metadata": {
                "dry_run": False,
                "audit_meta": dict(audit_meta or {}),
            },
        }

        result = graph.invoke({"input": input_payload, "context": run_context})

        # Output is in result["output"] usually, or result IS output?
        # Universal Graph returns UniversalIngestionOutput in 'output' key?
        # Wait, build_universal_ingestion_graph uses StateGraph.
        # invoke returns the final state.
        # UniversalIngestionState has 'output' key.
        final_output = result.get("output", {})

        # 6. Serialize result for Celery
        serialized_result = _jsonify_for_task(final_output)
        if not isinstance(serialized_result, dict):
            # Fallback if output structure is unexpected
            serialized_result = _jsonify_for_task(result)
        if not isinstance(serialized_result, dict):
            serialized_result = {"result": serialized_result}

        return _task_result(
            status="success",
            data=serialized_result,
            meta=meta,
            task_name=getattr(run_ingestion_graph, "name", "run_ingestion_graph"),
        )

    finally:
        try:
            obs_wrapper.end_trace()
        finally:
            # Cleanup temporary payload file
            if ingestion_ctx.raw_payload_path:
                from ai_core.ingestion import cleanup_raw_payload_artifact

                cleanup_raw_payload_artifact(ingestion_ctx.raw_payload_path)


def _prepare_working_state(
    state: Mapping[str, Any],
    ingestion_ctx,
    trace_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prepare working state with normalized document input.

    Extracted from run_ingestion_graph to isolate normalization logic.

    Args:
        state: Original graph state
        ingestion_ctx: Extracted ingestion context
        trace_context: Trace context

    Returns:
        Working state dict with normalized_document_input if applicable
    """
    working_state: Dict[str, Any] = dict(state)

    # Check if normalized input already present
    try:
        normalized_present = isinstance(
            working_state.get("normalized_document_input"), NormalizedDocument
        )
    except Exception:
        normalized_present = False

    if normalized_present:
        return working_state

    # Normalize from raw_document if available
    raw_reference = working_state.get("raw_document")
    if not isinstance(raw_reference, MappingABC):
        return working_state

    if not ingestion_ctx.tenant_id:
        return working_state

    try:
        normalized_payload = normalize_raw_reference(
            raw_reference=raw_reference,
            tenant_id=ingestion_ctx.tenant_id,
            case_id=ingestion_ctx.case_id,
            workflow_id=ingestion_ctx.workflow_id,
            source=ingestion_ctx.source,
        )

        # Serialize to maintain JSON compatibility for Celery task payloads
        working_state["normalized_document_input"] = (
            normalized_payload.document.model_dump(mode="json")
        )

        try:
            working_state["document_id"] = str(
                normalized_payload.document.ref.document_id
            )
        except Exception:
            pass

    except Exception:
        # Fall back to original state; the graph will surface an error
        pass

    return working_state


@shared_task(
    base=RetryableTask,
    queue="agents",
    name="ai_core.tasks.run_business_graph",
    time_limit=300,  # 5 minutes hard limit
    soft_time_limit=240,  # 4 minutes soft limit
)
def run_business_graph(
    graph_name: str,
    state: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a generic business graph via the registry.

    This task enables asynchronous execution of any registered graph
    (e.g., collection_search, web_acquisition, framework_analysis).

    Args:
        graph_name: The registry name of the graph to run.
        state: The initial graph state.
        meta: Metadata containing scope_context, business_context, etc.

    Returns:
        Dict with status, data (graph output), and metadata.
    """
    # 1. Resolve Graph
    try:
        runner = registry.get(graph_name)
    except KeyError:
        return _task_result(
            status="error",
            data={"error": f"Graph not found: {graph_name}"},
            meta=meta,
            task_name="run_business_graph",
        )

    # 2. Setup Observability
    trace_context = _resolve_trace_context(state, meta)

    # We need to construct a robust ToolContext for the graph
    # tool_context_from_meta handles fallback to scope/business keys
    try:
        tool_context = tool_context_from_meta(meta or {})
    except Exception as e:
        # Fallback for minimal context if meta is malformed
        # Create a dummy scope if needed, or fail? Better to fail safely.
        return _task_result(
            status="error",
            data={"error": f"Invalid context metadata: {str(e)}"},
            meta=meta,
            task_name="run_business_graph",
        )

    # Manual Trace Context start (since ObservabilityWrapper is ingestion-centric currently)
    # Actually, ObservabilityWrapper.start_trace expects IngestionContext.
    # TODO: Refactor ObservabilityWrapper to be generic.
    # For now, we use low-level start_trace if wrapper fails, or adapt.
    # Let's use low-level observability_helpers directly for generic graphs.

    # Start Trace
    span = observability_helpers.start_trace(
        name=f"task.{graph_name}",
        kind="server",
        trace_id=tool_context.scope.trace_id,
        parent_id=trace_context.get("span_id"),
        attributes={
            "graph.name": graph_name,
            "tenant.id": tool_context.scope.tenant_id,
            "case.id": tool_context.business.case_id,
        },
    )

    try:
        # 3. Execute Graph
        # GraphRunner.run() expects (state, meta) and handles tool_context injection if supported
        # But wait, registry.get() returns a GraphRunner or ANY (LazyGraphFactory returns instance).
        # Most graphs via `module_runner` or `build_graph` exposed `run` or `invoke`?
        # GraphRunner contract has `run(input, meta)`.
        # LangGraph implementations have `invoke(input, config)`.

        # We need to handle both provided inputs.
        # Standardize on GraphRunner.run() if possible.
        # If 'runner' has 'run' method, use it.

        if hasattr(runner, "run"):
            # Runner handles lifecycle
            final_state, result = runner.run(state, meta)
            output = result
        elif hasattr(runner, "invoke"):
            # Raw LangGraph or Runnable
            # We must construct config manually
            run_config = {"configurable": {"tool_context": tool_context}}
            # Pass meta as part of state? No.
            # Usually we pass input dict.
            # This path is risky without standardization.
            # AGENTS.md implies we should wrap in GraphRunner or compatible adapter.
            # Assuming widely used graphs have .run via `module_runner`.

            # Fallback for raw invoke
            final_state = runner.invoke(state, config=run_config)
            output = final_state  # Raw state as output
        else:
            raise ValueError(
                f"Graph runner for '{graph_name}' has no run() or invoke() method"
            )

        # 4. Success Response
        serialized_output = _jsonify_for_task(output)
        return _task_result(
            status="success",
            data=serialized_output,
            meta=meta,
            task_name=f"run_business_graph[{graph_name}]",
        )

    except Exception as exc:
        # 5. Error Response
        observability_helpers.record_exception(span, exc)
        return _task_result(
            status="error",
            data={
                "error": str(exc),
                "type": exc.__class__.__name__,
            },
            meta=meta,
            task_name=f"run_business_graph[{graph_name}]",
        )
    finally:
        observability_helpers.end_trace(span)
