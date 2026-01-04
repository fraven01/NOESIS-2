"""Command object for graph execution orchestration."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any
from uuid import uuid4

from celery import current_app, exceptions as celery_exceptions
from django.conf import settings
from pydantic import ValidationError
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response

from ai_core.graph.core import GraphContext, GraphRunner
from ai_core.graph.schemas import merge_state
from ai_core.graphs.technical.cost_tracking import track_ledger_costs
from ai_core.infra.observability import (
    emit_event,
    update_observation,
    end_trace as lf_end_trace,
    start_trace as lf_start_trace,
    tracing_enabled as lf_tracing_enabled,
)
from ai_core.infra.resp import build_tool_error_payload
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.tool_contracts import (
    ContextError as ToolContextError,
    InconsistentMetadataError as ToolInconsistentMetadataError,
    InputError as ToolInputError,
    NotFoundError as ToolNotFoundError,
    RateLimitedError as ToolRateLimitedError,
    TimeoutError as ToolTimeoutError,
    UpstreamServiceError as ToolUpstreamServiceError,
)
from ai_core.tool_contracts.base import tool_context_from_meta
from ai_core.tools import InputError
from common.celery import with_scope_apply_async
from ai_core.services.graph_support import (
    GRAPH_REQUEST_MODELS,
    _apply_collection_header_bridge,
    _dump_jsonable,
    _error_response,
    _extract_initial_cost,
    _extract_ledger_identifier,
    _format_validation_error,
    _get_checkpointer,
    _log_graph_response_payload,
    _normalize_meta,
    _should_enqueue_graph,
)

logger = logging.getLogger("ai_core.services.graph_executor")

GraphRunnerFactory = Callable[[], GraphRunner]


def _resolve_service_callable(name: str, fallback):  # type: ignore[no-untyped-def]
    """Allow tests to monkeypatch service callables via ai_core.services."""
    try:
        services = import_module("ai_core.services")
        candidate = getattr(services, name, None)
        if callable(candidate):
            return candidate
    except (ImportError, AttributeError):
        pass
    return fallback


class GraphExecutionCommand:
    """Encapsulate graph execution orchestration for reuse and testing."""

    def __init__(
        self,
        *,
        graph_request_models: Mapping[str, Any] | None = None,
        service_lookup: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self._graph_request_models = graph_request_models
        self._service_lookup = service_lookup or _resolve_service_callable

    def execute(
        self,
        request: Request,
        *,
        graph_runner_factory: GraphRunnerFactory,
    ) -> Response:
        """
        Orchestrates the execution of a graph, handling context, state, and errors.
        """
        normalize_meta = self._service_lookup("_normalize_meta", _normalize_meta)
        get_checkpointer = self._service_lookup("_get_checkpointer", _get_checkpointer)
        should_enqueue_graph = self._service_lookup(
            "_should_enqueue_graph", _should_enqueue_graph
        )
        with_scope_apply_async_fn = self._service_lookup(
            "with_scope_apply_async", with_scope_apply_async
        )
        update_observation_fn = self._service_lookup(
            "update_observation", update_observation
        )
        emit_event_fn = self._service_lookup("emit_event", emit_event)
        lf_tracing_enabled_fn = self._service_lookup(
            "lf_tracing_enabled", lf_tracing_enabled
        )
        lf_start_trace_fn = self._service_lookup("lf_start_trace", lf_start_trace)
        lf_end_trace_fn = self._service_lookup("lf_end_trace", lf_end_trace)

        graph_request_models = self._graph_request_models
        if graph_request_models is None:
            try:
                services = import_module("ai_core.services")
                graph_request_models = getattr(
                    services, "GRAPH_REQUEST_MODELS", GRAPH_REQUEST_MODELS
                )
            except (ImportError, AttributeError):
                graph_request_models = GRAPH_REQUEST_MODELS

        try:
            normalized_meta = normalize_meta(request)
        except ValueError as exc:
            error_msg = str(exc)
            error_code = (
                "invalid_case_header"
                if "Case header" in error_msg
                else "invalid_request"
            )
            return _error_response(error_msg, error_code, status.HTTP_400_BAD_REQUEST)

        tool_context = tool_context_from_meta(normalized_meta)
        setattr(request, "tool_context", tool_context)
        if hasattr(request, "_request") and request._request is not request:
            setattr(request._request, "tool_context", tool_context)

        run_id = uuid4().hex
        workflow_id = tool_context.business.workflow_id or tool_context.business.case_id
        context = GraphContext(
            tenant_id=tool_context.scope.tenant_id,
            case_id=tool_context.business.case_id,
            trace_id=tool_context.scope.trace_id,
            workflow_id=workflow_id,
            run_id=run_id,
            graph_name=normalized_meta["graph_name"],
            graph_version=normalized_meta["graph_version"],
        )

        ledger_identifier = _extract_ledger_identifier(normalized_meta)
        initial_cost_total = _extract_initial_cost(normalized_meta)
        base_observation_metadata: dict[str, Any] = {
            "trace_id": context.trace_id,
            "tenant.id": context.tenant_id,
            "case.id": context.case_id,
            "graph.version": context.graph_version,
            "workflow.id": context.workflow_id,
            "run.id": context.run_id,
        }
        if ledger_identifier:
            base_observation_metadata["ledger.id"] = ledger_identifier
        if initial_cost_total is not None:
            base_observation_metadata["cost.total_usd"] = initial_cost_total

        observation_kwargs = {
            "tags": [
                "graph",
                f"graph:{context.graph_name}",
                f"version:{context.graph_version}",
            ],
            "user_id": str(context.tenant_id),
            "session_id": str(context.case_id),
            "metadata": dict(base_observation_metadata),
        }

        req_started = time.monotonic()
        try:
            logger.info(
                "graph.request.start",
                extra={
                    "graph": context.graph_name,
                    "tenant_id": context.tenant_id,
                    "case_id": context.case_id,
                    "trace_id": context.trace_id,
                },
            )
        except Exception:
            pass

        trace_started = False
        if lf_tracing_enabled_fn():
            try:
                lf_start_trace_fn(
                    name=f"graph:{context.graph_name}",
                    user_id=str(context.tenant_id),
                    session_id=str(context.case_id),
                    metadata={
                        "trace_id": context.trace_id,
                        "version": context.graph_version,
                        "workflow_id": context.workflow_id,
                        "run_id": context.run_id,
                    },
                )
                trace_started = True
            except Exception:
                pass
            if trace_started:
                try:
                    update_observation_fn(**observation_kwargs)
                except Exception:
                    pass

        cost_summary: dict[str, Any] | None = None
        try:
            try:
                update_observation_fn(**observation_kwargs)
            except Exception:
                pass
            try:
                state = get_checkpointer().load(context)
            except (TypeError, ValueError) as exc:
                return _error_response(
                    str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
                )

            raw_body = getattr(request, "body", b"")
            if not raw_body and hasattr(request, "_request"):
                raw_body = getattr(request._request, "body", b"")
            content_type_header = request.headers.get("Content-Type")
            normalized_content_type = ""
            if content_type_header:
                normalized_content_type = (
                    content_type_header.split(";")[0].strip().lower()
                )

            incoming_state = None
            if raw_body:
                if normalized_content_type and not (
                    normalized_content_type == "application/json"
                    or normalized_content_type.endswith("+json")
                ):
                    return _error_response(
                        "Request payload must be encoded as application/json.",
                        "unsupported_media_type",
                        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    )
                try:
                    payload = json.loads(raw_body)
                    if isinstance(payload, dict):
                        incoming_state = payload
                except json.JSONDecodeError:
                    return _error_response(
                        "Request payload contained invalid JSON.",
                        "invalid_json",
                        status.HTTP_400_BAD_REQUEST,
                    )

            data = _apply_collection_header_bridge(request, incoming_state)
            request_model = graph_request_models.get(context.graph_name)
            if request_model is not None:
                try:
                    validated = request_model.model_validate(data)
                except ValidationError as exc:
                    return _error_response(
                        _format_validation_error(exc),
                        "invalid_request",
                        status.HTTP_400_BAD_REQUEST,
                    )
                incoming_state = validated.model_dump(exclude_none=True)
            else:
                incoming_state = data

            merged_state = merge_state(state, incoming_state)
            runner_meta = dict(normalized_meta)
            if normalized_meta.get("tenant_schema"):
                runner_meta["tenant_schema"] = normalized_meta["tenant_schema"]
            if normalized_meta.get("key_alias"):
                runner_meta["key_alias"] = normalized_meta["key_alias"]

            try:
                t0 = time.monotonic()
                try:
                    logger.info(
                        "graph.run.start",
                        extra={
                            "graph": context.graph_name,
                            "tenant_id": context.tenant_id,
                            "case_id": context.case_id,
                        },
                    )
                except Exception:
                    pass

                if should_enqueue_graph(context.graph_name):
                    signature = current_app.signature(
                        "llm_worker.tasks.run_graph",
                        kwargs={
                            "graph_name": context.graph_name,
                            "state": merged_state,
                            "meta": runner_meta,
                            "ledger_identifier": ledger_identifier,
                            "initial_cost_total": initial_cost_total,
                        },
                        queue="agents-high",
                    )
                    scope = {
                        "tenant_id": context.tenant_id,
                        "case_id": context.case_id,
                        "trace_id": context.trace_id,
                    }
                    async_result = with_scope_apply_async_fn(signature, scope)
                    try:
                        timeout_s = getattr(settings, "GRAPH_WORKER_TIMEOUT_S", 45)
                        task_payload = async_result.get(
                            timeout=timeout_s, propagate=True
                        )
                        new_state = task_payload["state"]
                        result = task_payload["result"]
                        cost_summary = task_payload.get("cost_summary")
                        if cost_summary and "total_usd" in cost_summary:
                            cost_summary["total_usd"] = round(
                                cost_summary["total_usd"], 4
                            )
                    except celery_exceptions.TimeoutError:
                        logger.warning(
                            "graph.worker_timeout",
                            extra={
                                "graph": context.graph_name,
                                "tenant_id": context.tenant_id,
                                "case_id": context.case_id,
                                "task_id": async_result.id,
                                "timeout_s": timeout_s,
                            },
                        )
                        return Response(
                            {
                                "status": "queued",
                                "task_id": async_result.id,
                                "graph": context.graph_name,
                                "tenant_id": context.tenant_id,
                                "case_id": context.case_id,
                                "trace_id": context.trace_id,
                            },
                            status=status.HTTP_202_ACCEPTED,
                        )
                else:
                    runner = graph_runner_factory()
                    with track_ledger_costs(initial_cost_total) as tracker:
                        runner_meta["ledger_logger"] = tracker.record_ledger_meta
                        try:
                            new_state, result = runner.run(merged_state, runner_meta)
                        finally:
                            runner_meta.pop("ledger_logger", None)
                    cost_summary = tracker.summary(ledger_identifier)
                    if cost_summary and "total_usd" in cost_summary:
                        cost_summary["total_usd"] = round(cost_summary["total_usd"], 4)
                try:
                    dt_ms = int((time.monotonic() - t0) * 1000)
                    logger.info(
                        "graph.run.end",
                        extra={
                            "graph": context.graph_name,
                            "tenant_id": context.tenant_id,
                            "case_id": context.case_id,
                            "duration_ms": dt_ms,
                        },
                    )
                except Exception:
                    pass
            except InputError as exc:
                return _error_response(
                    str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
                )
            except ToolContextError as exc:
                return _error_response(
                    str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
                )
            except ValueError as exc:
                return _error_response(
                    str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
                )
            except ToolNotFoundError as exc:
                logger.info("tool.not_found")
                detail = str(exc) or "No matching documents were found."
                return _error_response(
                    detail, "rag_no_matches", status.HTTP_404_NOT_FOUND
                )
            except ToolInconsistentMetadataError as exc:
                logger.warning("tool.inconsistent_metadata")
                context_data = getattr(exc, "context", None)
                payload = build_tool_error_payload(
                    message=str(exc) or "reindex required",
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    code="retrieval_inconsistent_metadata",
                    details=context_data if isinstance(context_data, Mapping) else None,
                )
                return Response(payload, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
            except ToolInputError as exc:
                return _error_response(
                    str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
                )
            except ToolRateLimitedError:
                logger.warning("tool.rate_limited")
                return _error_response(
                    "Tool rate limited.",
                    "llm_rate_limited",
                    status.HTTP_429_TOO_MANY_REQUESTS,
                )
            except ToolTimeoutError:
                logger.warning("tool.timeout")
                return _error_response(
                    "Upstream tool timeout.",
                    "llm_timeout",
                    status.HTTP_504_GATEWAY_TIMEOUT,
                )
            except ToolUpstreamServiceError:
                logger.warning("tool.upstream_error")
                return _error_response(
                    "Upstream tool error.",
                    "llm_error",
                    status.HTTP_502_BAD_GATEWAY,
                )
            except RateLimitError as exc:
                try:
                    extra = {
                        "status": getattr(exc, "status", None),
                        "code": getattr(exc, "code", None),
                    }
                    logger.warning("llm.rate_limited", extra=extra)
                except Exception:
                    pass
                status_code = (
                    int(getattr(exc, "status", 429))
                    if str(getattr(exc, "status", "")).isdigit()
                    else status.HTTP_429_TOO_MANY_REQUESTS
                )
                detail = getattr(exc, "detail", None) or "LLM rate limited."
                return _error_response(detail, "llm_rate_limited", status_code)
            except LlmClientError as exc:
                try:
                    extra = {
                        "status": getattr(exc, "status", None),
                        "code": getattr(exc, "code", None),
                    }
                    logger.warning("llm.client_error", extra=extra)
                except Exception:
                    pass
                raw_status = getattr(exc, "status", None)
                status_code = status.HTTP_502_BAD_GATEWAY
                try:
                    if isinstance(raw_status, int) and raw_status == 429:
                        status_code = status.HTTP_429_TOO_MANY_REQUESTS
                except Exception:
                    pass
                detail = getattr(exc, "detail", None) or "Upstream LLM error."
                return _error_response(detail, "llm_error", status_code)
            except Exception:
                try:
                    logger.exception(
                        "graph.execution_failed",
                        extra={
                            "graph": context.graph_name,
                            "tenant_id": context.tenant_id,
                            "case_id": context.case_id,
                        },
                    )
                except Exception:
                    pass
                return _error_response(
                    "Service temporarily unavailable.",
                    "service_unavailable",
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            try:
                get_checkpointer().save(context, _dump_jsonable(new_state))
            except (TypeError, ValueError) as exc:
                return _error_response(
                    str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
                )
            try:
                _log_graph_response_payload(result, context)
            except TypeError:
                raise
            except Exception:
                logger.exception(
                    "graph.response_payload_logging_failed",
                    extra={
                        "graph": context.graph_name,
                        "tenant_id": context.tenant_id,
                        "case_id": context.case_id,
                    },
                )

            try:
                response = Response(_dump_jsonable(result))
            except TypeError:
                logger.exception(
                    "graph.response_serialization_error",
                    extra={
                        "graph": context.graph_name,
                        "tenant_id": context.tenant_id,
                        "case_id": context.case_id,
                        "payload_type": type(result).__name__,
                    },
                )
                raise

            if cost_summary:
                final_metadata = dict(base_observation_metadata)
                final_metadata["cost.total_usd"] = cost_summary["total_usd"]
                try:
                    update_observation_fn(metadata=final_metadata)
                except Exception:
                    pass
                event_payload = {
                    "total_usd": cost_summary["total_usd"],
                    "components": cost_summary["components"],
                    "tenant_id": context.tenant_id,
                    "case_id": context.case_id,
                    "graph_name": context.graph_name,
                    "graph_version": context.graph_version,
                }
                if "reconciliation" in cost_summary:
                    event_payload["reconciliation"] = cost_summary["reconciliation"]
                try:
                    emit_event_fn("cost.summary", event_payload)
                except Exception:
                    pass
            return response
        finally:
            try:
                lf_end_trace_fn()
            except Exception:
                pass
            try:
                dt_total_ms = int((time.monotonic() - req_started) * 1000)
                logger.info(
                    "graph.request.end",
                    extra={
                        "graph": context.graph_name,
                        "tenant_id": context.tenant_id,
                        "case_id": context.case_id,
                        "duration_ms": dt_total_ms,
                    },
                )
            except Exception:
                pass
