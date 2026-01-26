from __future__ import annotations

import logging
import os
import random
import re
import time
import uuid
from datetime import datetime
from collections.abc import Mapping
from typing import Any, Mapping as TypingMapping

from celery import Task
from celery.canvas import Signature
from celery.exceptions import SoftTimeLimitExceeded
from celery.result import AsyncResult
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .constants import HEADER_CANDIDATE_MAP, X_TRACE_ID_HEADER
from .logging import bind_log_context, clear_log_context
from structlog.contextvars import bind_contextvars, clear_contextvars
from ai_core.infra.pii_flags import (
    clear_pii_config,
    load_tenant_pii_config,
    set_pii_config,
)
from ai_core.infra.observability import emit_event, record_span, update_observation
from ai_core.infra import rate_limit as tenant_rate_limit
from ai_core.infra.policy import (
    clear_session_scope,
    set_session_scope,
    get_session_scope,
)
from ai_core.metrics.task_metrics import record_task_retry
from ai_core.tool_contracts.base import ToolContext
from ai_core.tool_contracts.task_context import TaskContext, task_context_from_meta
from ai_core.tools.errors import (
    InputError,
    PermanentError,
    RateLimitedError,
    TransientError,
    UpstreamError,
)

try:  # pragma: no cover - defensive import when Django isn't available
    from django.db import DatabaseError, OperationalError
except Exception:  # pragma: no cover - optional for tests
    DatabaseError = None  # type: ignore[assignment]
    OperationalError = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from ai_core.rag.embeddings import (
        EmbeddingClientError,
        EmbeddingProviderUnavailable,
        EmbeddingTimeoutError,
    )
except Exception:  # pragma: no cover - optional dependency
    EmbeddingClientError = None  # type: ignore[assignment]
    EmbeddingProviderUnavailable = None  # type: ignore[assignment]
    EmbeddingTimeoutError = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from ai_core.llm.client import LlmUpstreamError, RateLimitError as LlmRateLimitError
except Exception:  # pragma: no cover - optional dependency
    LlmUpstreamError = None  # type: ignore[assignment]
    LlmRateLimitError = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


try:
    from opentelemetry import context as otel_context
    from opentelemetry.trace import (
        SpanContext,
        TraceFlags,
        NonRecordingSpan,
        set_span_in_context,
    )
    from opentelemetry.trace.propagation.trace_context import (
        TraceContextTextMapPropagator,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def _normalize_trace_id_for_otel(trace_id: object) -> str | None:
    if trace_id is None:
        return None
    text = str(trace_id).strip()
    if not text:
        return None
    try:
        return uuid.UUID(text).hex
    except (TypeError, ValueError):
        pass
    cleaned = re.sub(r"[^0-9a-fA-F]", "", text).lower()
    if len(cleaned) == 32:
        return cleaned
    if len(cleaned) > 32:
        return cleaned[:32]
    return None


def _otel_context_from_trace_id(trace_id: str) -> object | None:
    if not _OTEL_AVAILABLE:
        return None
    normalized = _normalize_trace_id_for_otel(trace_id)
    if not normalized:
        return None
    try:
        trace_int = int(normalized, 16)
    except (TypeError, ValueError):
        return None
    if not trace_int:
        return None
    try:
        span_id = random.getrandbits(64)
        trace_flags = (
            TraceFlags.SAMPLED
            if isinstance(TraceFlags.SAMPLED, TraceFlags)
            else TraceFlags(TraceFlags.SAMPLED)
        )
        span_context = SpanContext(
            trace_id=trace_int,
            span_id=span_id,
            is_remote=True,
            trace_flags=trace_flags,
        )
        return set_span_in_context(NonRecordingSpan(span_context))
    except Exception:
        return None


def _current_otel_trace_id() -> str | None:
    if not _OTEL_AVAILABLE:
        return None
    try:
        from opentelemetry import trace as otel_trace

        span = otel_trace.get_current_span()
        span_context = span.get_span_context() if span else None
        if not span_context or not span_context.is_valid:
            return None
        return f"{span_context.trace_id:032x}"
    except Exception:
        return None


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    try:
        return str(value).strip() or None
    except Exception:
        return None


class TaskContextMeta(BaseModel):
    tenant_id: str | None = None
    case_id: str | None = None
    trace_id: str | None = None

    model_config = ConfigDict(extra="ignore", frozen=True)

    @field_validator("tenant_id", "case_id", "trace_id", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        return _coerce_optional_text(value)


class PiiSessionContext(BaseModel):
    session_salt: str | None = None
    session_scope: tuple[str, str, str] | None = None

    model_config = ConfigDict(extra="ignore", frozen=True)

    @field_validator("session_salt", mode="before")
    @classmethod
    def _normalize_session_salt(cls, value: Any) -> str | None:
        return _coerce_optional_text(value)

    @field_validator("session_scope", mode="before")
    @classmethod
    def _normalize_session_scope(cls, value: Any) -> tuple[str, str, str] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        tenant_val, case_val, salt_val = value
        return (
            _coerce_optional_text(tenant_val) or "",
            _coerce_optional_text(case_val) or "",
            _coerce_optional_text(salt_val) or "",
        )


class ContextTask(Task):
    """Celery task base class that binds log context from headers/kwargs."""

    abstract = True

    _HEADER_CANDIDATES = HEADER_CANDIDATE_MAP
    _REQUEST_OVERRIDE_SENTINEL = object()

    @property
    def request(self):  # type: ignore[override]
        override = getattr(self, "_request_override", self._REQUEST_OVERRIDE_SENTINEL)
        if override is not self._REQUEST_OVERRIDE_SENTINEL:
            return override
        return Task.request.__get__(self, type(self))

    @request.setter
    def request(self, value):
        self._request_override = value

    @request.deleter
    def request(self):  # noqa: F811
        self._request_override = self._REQUEST_OVERRIDE_SENTINEL

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
        clear_log_context()
        clear_contextvars()
        context = self._gather_context(args, kwargs)
        request = getattr(self, "request", None)
        task_id = getattr(request, "id", None) if request is not None else None
        if task_id:
            task_id = self._normalize(task_id)
            context["task_id"] = task_id
        if context:
            bind_contextvars(**context)
            bind_log_context(**context)

        token = None
        if _OTEL_AVAILABLE:
            trace_id = context.get("trace_id")
            ctx = _otel_context_from_trace_id(trace_id) if trace_id else None
            if ctx is None:
                # Extract W3C trace context from headers and activate it
                request_headers = getattr(self.request, "headers", None) or {}
                if not isinstance(request_headers, dict):
                    request_headers = {}
                ctx = TraceContextTextMapPropagator().extract(carrier=request_headers)
            token = otel_context.attach(ctx)
            logger.debug(
                "celery.task.context",
                extra={
                    "task_id": task_id,
                    "task_name": getattr(self, "name", None),
                    "trace_id": context.get("trace_id"),
                    "otel_trace_id": _current_otel_trace_id(),
                    "otel_from_context": bool(trace_id),
                },
            )

        task_name = getattr(self, "name", None) or type(self).__name__
        queue = _resolve_task_queue(self, kwargs)
        queue_time_ms = _resolve_queue_time_ms(request)
        start_perf = time.perf_counter()
        retry_count = int(getattr(request, "retries", 0)) if request is not None else 0
        start_payload: dict[str, Any] = {
            "task_id": task_id,
            "task_name": task_name,
            "queue": queue,
            "task.queue_time_ms": queue_time_ms,
            "task.retry_count": retry_count,
        }

        record_span(
            f"task.{task_name}.start",
            attributes={**context, **start_payload},
        )
        logger.info("celery.task.start", extra=start_payload)

        exc: BaseException | None = None
        try:
            pre_execute = getattr(self, "_pre_execute", None)
            if callable(pre_execute):
                pre_execute(args, kwargs)
            return super().__call__(*args, **kwargs)
        except BaseException as caught:
            exc = caught
            raise
        finally:
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            retry_count = (
                int(getattr(request, "retries", 0)) if request is not None else 0
            )
            end_payload = {
                **start_payload,
                "task.duration_ms": duration_ms,
                "task.retry_count": retry_count,
                "task.status": "error" if exc is not None else "ok",
            }
            if exc is not None:
                end_payload["error.type"] = exc.__class__.__name__
                end_payload["error.message"] = str(exc)

            record_span(
                f"task.{task_name}.end",
                attributes={**context, **end_payload},
            )

            if exc is not None:
                logger.error("celery.task.failure", extra=end_payload, exc_info=True)
            else:
                logger.info("celery.task.end", extra=end_payload)

            if _OTEL_AVAILABLE and token is not None:
                otel_context.detach(token)
            clear_log_context()
            clear_contextvars()

    def _gather_context(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, str]:
        context: dict[str, str] = {}

        request = getattr(self, "request", None)
        if request:
            context.update(self._from_headers(getattr(request, "headers", None)))
            request_kwargs = getattr(request, "kwargs", None)
            if not isinstance(request_kwargs, Mapping):
                request_kwargs = {}
            context.update(self._from_meta(request_kwargs.get("meta")))
            context.update(self._from_tool_context(request_kwargs.get("tool_context")))

        context.update(self._from_meta(kwargs.get("meta")))
        context.update(self._from_tool_context(kwargs.get("tool_context")))

        try:
            normalized = TaskContextMeta.model_validate(context).model_dump(
                exclude_none=True
            )
            context.update(normalized)
        except ValidationError:
            pass

        return {key: value for key, value in context.items() if value}

    def _from_headers(self, headers: Any) -> dict[str, str]:
        if not isinstance(headers, Mapping):
            return {}

        lowered = {str(key).lower(): value for key, value in headers.items()}
        context: dict[str, str] = {}

        for field, candidates in self._HEADER_CANDIDATES.items():
            for candidate in candidates:
                value = lowered.get(candidate.lower())
                if value:
                    context[field] = self._normalize(value)
                    break

        return context

    def _from_meta(self, meta: Any) -> dict[str, str]:
        """Extract scope context from meta dict.

        BREAKING CHANGE (Option A - Strict Separation):
        case_id is a business identifier, now extracted from business_context.
        """
        if not isinstance(meta, Mapping):
            return {}

        try:
            task_context = task_context_from_meta(meta)
        except (TypeError, ValueError, ValidationError):
            return {}

        return self._from_task_context(task_context)

    def _from_task_context(self, task_context: TaskContext) -> dict[str, str]:
        context: dict[str, str] = {}

        scope = task_context.scope
        business = task_context.business

        # Infrastructure IDs from scope_context
        if scope.trace_id:
            context["trace_id"] = self._normalize(scope.trace_id)
        if scope.tenant_id:
            context["tenant_id"] = self._normalize(scope.tenant_id)
        if scope.run_id:
            context["run_id"] = self._normalize(scope.run_id)
        if scope.ingestion_run_id:
            context["ingestion_run_id"] = self._normalize(scope.ingestion_run_id)

        # Business IDs from business_context (BREAKING CHANGE)
        if business.case_id:
            context["case_id"] = self._normalize(business.case_id)
        if business.collection_id:
            context["collection_id"] = self._normalize(business.collection_id)
        if business.workflow_id:
            context["workflow_id"] = self._normalize(business.workflow_id)
        if business.document_id:
            context["document_id"] = self._normalize(business.document_id)
        if business.document_version_id:
            context["document_version_id"] = self._normalize(
                business.document_version_id
            )

        key_alias = task_context.metadata.key_alias
        if key_alias:
            context["key_alias"] = self._normalize(key_alias)

        return context

    def _from_tool_context(self, tool_context: Any) -> dict[str, str]:
        if tool_context is None:
            return {}

        tenant_id = None
        case_id = None
        trace_id = None
        idempotency_key = None
        collection_id = None
        workflow_id = None
        document_id = None
        document_version_id = None
        run_id = None
        ingestion_run_id = None
        key_alias = None

        if isinstance(tool_context, TaskContext):
            scope = tool_context.scope
            business = tool_context.business
            tenant_id = scope.tenant_id
            trace_id = scope.trace_id
            idempotency_key = scope.idempotency_key
            case_id = business.case_id
            collection_id = business.collection_id
            workflow_id = business.workflow_id
            document_id = business.document_id
            document_version_id = business.document_version_id
            run_id = scope.run_id
            ingestion_run_id = scope.ingestion_run_id
            key_alias = tool_context.metadata.key_alias
        elif isinstance(tool_context, ToolContext):
            scope = tool_context.scope
            business = tool_context.business
            tenant_id = scope.tenant_id
            trace_id = scope.trace_id
            idempotency_key = scope.idempotency_key
            case_id = business.case_id
            collection_id = business.collection_id
            workflow_id = business.workflow_id
            document_id = business.document_id
            document_version_id = business.document_version_id
            run_id = scope.run_id
            ingestion_run_id = scope.ingestion_run_id
        elif isinstance(tool_context, Mapping):
            scope_data = tool_context.get("scope")
            business_data = tool_context.get("business")
            if isinstance(scope_data, Mapping):
                tenant_id = scope_data.get("tenant_id")
                trace_id = scope_data.get("trace_id")
                idempotency_key = scope_data.get("idempotency_key")
                run_id = scope_data.get("run_id")
                ingestion_run_id = scope_data.get("ingestion_run_id")
            if isinstance(business_data, Mapping):
                case_id = business_data.get("case_id")
                collection_id = business_data.get("collection_id")
                workflow_id = business_data.get("workflow_id")
                document_id = business_data.get("document_id")
                document_version_id = business_data.get("document_version_id")
        else:
            return {}

        context: dict[str, str] = {}

        if tenant_id:
            context["tenant_id"] = self._normalize(tenant_id)

        if case_id:
            context["case_id"] = self._normalize(case_id)

        if trace_id:
            context["trace_id"] = self._normalize(trace_id)

        if run_id:
            context["run_id"] = self._normalize(run_id)

        if ingestion_run_id:
            context["ingestion_run_id"] = self._normalize(ingestion_run_id)

        if idempotency_key:
            context["idempotency_key"] = self._normalize(idempotency_key)

        if collection_id:
            context["collection_id"] = self._normalize(collection_id)

        if workflow_id:
            context["workflow_id"] = self._normalize(workflow_id)

        if document_id:
            context["document_id"] = self._normalize(document_id)

        if document_version_id:
            context["document_version_id"] = self._normalize(document_version_id)

        if key_alias:
            context["key_alias"] = self._normalize(key_alias)

        return context

    @staticmethod
    def _normalize(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return str(value)


class ScopedTask(ContextTask):
    """Celery task base that propagates the PII session scope.

    The scope is a three-part tuple ``(tenant_id, case_id, session_salt)``
    that drives deterministic masking and logging behaviour for PII-aware
    helpers. ``session_salt`` acts as the entropy source that keeps masking
    tokens stable for a single trace/case combination, while ``session_scope``
    lets producers override the tuple entirely when they already derived a
    canonical scope upstream. Scope values are derived from meta/tool_context
    (or explicit scope kwargs) without being forwarded into task signatures.
    """

    abstract = True

    # PII-session scope fields (not ScopeContext fields).
    _PII_SESSION_FIELDS = ("tenant_id", "case_id", "trace_id", "session_salt")

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
        call_kwargs = dict(kwargs)
        scope_kwargs: dict[str, Any] = {}
        for field in self._PII_SESSION_FIELDS:
            if field in call_kwargs:
                scope_kwargs[field] = call_kwargs.pop(field)

        explicit_scope = call_kwargs.pop("session_scope", None)

        context = self._gather_context(args, call_kwargs)

        pii_context = PiiSessionContext.model_validate(
            {
                "session_salt": scope_kwargs.get("session_salt"),
                "session_scope": explicit_scope,
            }
        )

        tenant_id = _coerce_optional_text(
            scope_kwargs.get("tenant_id")
        ) or _coerce_optional_text(context.get("tenant_id"))
        case_id = _coerce_optional_text(
            scope_kwargs.get("case_id")
        ) or _coerce_optional_text(context.get("case_id"))
        trace_id = _coerce_optional_text(
            scope_kwargs.get("trace_id")
        ) or _coerce_optional_text(context.get("trace_id"))
        session_salt = pii_context.session_salt
        scope_override = pii_context.session_scope

        if scope_override is not None:
            tenant_scope_val, case_scope_val, salt_scope_val = scope_override
            if not case_id and case_scope_val:
                case_id = case_scope_val or None
            if not session_salt and salt_scope_val:
                session_salt = salt_scope_val or None

        if not session_salt:
            session_salt = _derive_session_salt_from_ids(
                trace_id=trace_id,
                case_id=case_id,
                tenant_id=tenant_id,
            )

        tenant_config = load_tenant_pii_config(tenant_id) if tenant_id else None

        scope = None
        if tenant_id and case_id and session_salt:
            tenant_scope = (
                scope_override[0] if scope_override and scope_override[0] else tenant_id
            )
            case_scope = (
                scope_override[1] if scope_override and scope_override[1] else case_id
            )
            salt_scope = (
                scope_override[2]
                if scope_override and scope_override[2]
                else session_salt
            )
            set_session_scope(
                tenant_id=str(tenant_scope),
                case_id=str(case_scope),
                session_salt=str(salt_scope),
            )
            scope = get_session_scope()
        elif scope_override and all(scope_override):
            set_session_scope(
                tenant_id=str(scope_override[0]),
                case_id=str(scope_override[1]),
                session_salt=str(scope_override[2]),
            )
            scope = get_session_scope()

        if tenant_config:
            if scope:
                scoped_config = dict(tenant_config)
                scoped_config["session_scope"] = scope
            else:
                scoped_config = tenant_config
            set_pii_config(scoped_config)

        try:
            return super().__call__(*args, **call_kwargs)
        finally:
            clear_pii_config()
            clear_session_scope()


def _filter_retry_classes(*classes: Any) -> tuple[type[BaseException], ...]:
    return tuple(candidate for candidate in classes if isinstance(candidate, type))


_RETRYABLE_EXCEPTIONS = _filter_retry_classes(
    TransientError,
    RateLimitedError,
    UpstreamError,
    EmbeddingTimeoutError,
    EmbeddingClientError,
    LlmUpstreamError,
    LlmRateLimitError,
    TimeoutError,
    SoftTimeLimitExceeded,
    ConnectionError,
    OperationalError,
    DatabaseError,
)

_NON_RETRYABLE_EXCEPTIONS = _filter_retry_classes(
    InputError,
    PermanentError,
    ValidationError,
    EmbeddingProviderUnavailable,
)


def _is_instance(
    exc: BaseException, candidates: tuple[type[BaseException], ...]
) -> bool:
    for candidate in candidates:
        if isinstance(exc, candidate):
            return True
    return False


def _retry_reason_category(exc: BaseException) -> str:
    if _is_instance(exc, _filter_retry_classes(RateLimitedError, LlmRateLimitError)):
        return "rate_limit"
    if _is_instance(
        exc,
        _filter_retry_classes(
            EmbeddingTimeoutError,
            TimeoutError,
            SoftTimeLimitExceeded,
        ),
    ):
        return "timeout"
    if _is_instance(exc, _filter_retry_classes(OperationalError, DatabaseError)):
        return "db_error"
    if _is_instance(
        exc,
        _filter_retry_classes(UpstreamError, EmbeddingClientError, LlmUpstreamError),
    ):
        return "api_error"
    if _is_instance(exc, _filter_retry_classes(ConnectionError)):
        return "network"
    if _is_instance(exc, _filter_retry_classes(TransientError)):
        return "transient"
    return "unknown"


def _resolve_agent_id(task: Task, kwargs: dict[str, Any]) -> str:
    graph_name = kwargs.get("graph_name")
    if graph_name:
        return str(graph_name)
    task_name = getattr(task, "name", None)
    return str(task_name) if task_name else "unknown"


def _resolve_priority(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip().lower()
    except Exception:
        return None
    return text or None


def _resolve_priority_from_kwargs(kwargs: dict[str, Any]) -> str | None:
    for key in ("priority", "task_priority"):
        value = _resolve_priority(kwargs.get(key))
        if value:
            return value
    meta = kwargs.get("meta")
    if isinstance(meta, Mapping):
        for key in ("priority", "task_priority"):
            value = _resolve_priority(meta.get(key))
            if value:
                return value
    return None


def _coerce_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0:
        return None
    if candidate > 1e12:
        candidate /= 1000.0
    return candidate


def _resolve_queue_time_ms(request: Any) -> float | None:
    if request is None:
        return None

    sent_at: float | None = None
    headers = getattr(request, "headers", None)
    if isinstance(headers, Mapping):
        for key in ("sent_at", "sentat", "sent-at", "timestamp"):
            sent_at = _coerce_timestamp(headers.get(key))
            if sent_at is not None:
                break

    if sent_at is None:
        sent_at = _coerce_timestamp(getattr(request, "sent_at", None))

    if sent_at is None:
        return None

    start_time = _coerce_timestamp(getattr(request, "time_start", None))
    if start_time is None:
        start_time = time.time()

    queue_time = (start_time - sent_at) * 1000.0
    if queue_time < 0:
        return 0.0
    return queue_time


def _resolve_task_queue(task: Task, kwargs: dict[str, Any]) -> str | None:
    delivery_info = getattr(task.request, "delivery_info", None)
    if isinstance(delivery_info, Mapping):
        queue = (
            delivery_info.get("routing_key")
            or delivery_info.get("queue")
            or delivery_info.get("exchange")
        )
        if queue:
            return str(queue)

    task_name = getattr(task, "name", None) or ""
    return _select_queue_for_task(task_name, _resolve_priority_from_kwargs(kwargs))


def _resolve_rate_limit_scope(queue: str | None) -> str | None:
    if not queue:
        return None
    if queue == "ingestion-bulk":
        return None
    if queue.startswith("agents-"):
        return "agents"
    if queue == "ingestion":
        return "ingestion"
    return None


def _select_queue_for_task(task_name: str, priority: str | None) -> str | None:
    if task_name == "llm_worker.tasks.run_graph":
        if priority in {"low", "background", "bulk"}:
            return "agents-low"
        return "agents-high"
    if task_name.startswith("ai_core.tasks.") or task_name.startswith(
        "ai_core.ingestion."
    ):
        if priority in {"low", "background", "bulk"}:
            return "ingestion-bulk"
        return "ingestion"
    return None


def route_task(  # noqa: D401
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    options: dict[str, Any],
    task: Any = None,
    **_other: Any,
) -> dict[str, Any] | None:
    """Route tasks to queues based on declared priority."""

    if options.get("queue"):
        return None

    queue = _select_queue_for_task(name, _resolve_priority_from_kwargs(kwargs))
    if not queue:
        return None

    return {"queue": queue, "routing_key": queue}


class RetryableTask(ScopedTask):
    """ScopedTask with centralized retry handling and observability hooks."""

    abstract = True
    autoretry_for = _RETRYABLE_EXCEPTIONS
    dont_autoretry_for = _NON_RETRYABLE_EXCEPTIONS
    max_retries = 3
    retry_backoff = True
    retry_backoff_max = 300
    retry_jitter = True

    def _pre_execute(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if not _rate_limit_enabled():
            return
        context = self._gather_context(args, kwargs)
        tenant_id = context.get("tenant_id") or context.get("tenant")
        if not tenant_id:
            return
        queue = _resolve_task_queue(self, kwargs)
        scope = _resolve_rate_limit_scope(queue)
        if not scope:
            return
        if tenant_rate_limit.check_scoped(str(tenant_id), scope):
            return
        retry_after_ms = _compute_retry_after_ms()
        raise RateLimitedError(
            code="rate_limit",
            message="Tenant rate limit exceeded",
            context={
                "tenant_id": str(tenant_id),
                "rate_limit_scope": scope,
                "queue": queue,
            },
            retry_after_ms=retry_after_ms,
        )

    def on_retry(  # noqa: D401
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        reason_category = _retry_reason_category(exc)
        retry_count = int(getattr(self.request, "retries", 0))
        attempt = retry_count + 1
        task_name = getattr(self, "name", None)
        agent_id = _resolve_agent_id(self, kwargs)
        context = self._gather_context(args, kwargs)
        if "tenant" in context and "tenant_id" not in context:
            context["tenant_id"] = context["tenant"]

        payload: dict[str, Any] = {
            "task_id": task_id,
            "task_name": task_name,
            "agent_id": agent_id,
            "retry_count": retry_count,
            "attempt": attempt,
            "max_retries": getattr(self, "max_retries", None),
            "reason_category": reason_category,
            "exc_type": exc.__class__.__name__,
            "exc_message": str(exc),
            "task.retry_count": retry_count,
            **context,
        }

        delivery_info = getattr(self.request, "delivery_info", None)
        if isinstance(delivery_info, Mapping):
            payload["queue"] = delivery_info.get("routing_key") or delivery_info.get(
                "exchange"
            )

        retry_after_ms = getattr(exc, "retry_after_ms", None)
        if retry_after_ms is not None:
            payload["retry_after_ms"] = retry_after_ms

        upstream_status = getattr(exc, "upstream_status", None)
        if upstream_status is not None:
            payload["upstream_status"] = upstream_status

        if einfo is not None:
            traceback = getattr(einfo, "traceback", None)
            if traceback:
                payload["traceback"] = str(traceback)

        try:
            record_task_retry(agent_id=agent_id, reason_category=reason_category)
        except Exception:
            logger.debug("task.retry.metrics_failed", exc_info=True)

        try:
            update_observation(
                tags=["task", "retry"],
                metadata={
                    "task.retry.count": retry_count,
                    "task.retry.attempt": attempt,
                    "task.retry.reason": reason_category,
                    "task.retry.exception": exc.__class__.__name__,
                    "task.name": getattr(self, "name", None),
                },
            )
        except Exception:
            logger.debug("task.retry.span_failed", exc_info=True)

        try:
            record_span(
                f"task.{task_name or agent_id}.retry.{attempt}",
                attributes=payload,
            )
        except Exception:
            logger.debug("task.retry.record_span_failed", exc_info=True)

        try:
            emit_event("task.retry", payload)
        except Exception:
            logger.debug("task.retry.event_failed", exc_info=True)

        logger.warning("celery.task.retry", extra=payload)
        return super().on_retry(exc, task_id, args, kwargs, einfo)


def _rate_limit_enabled() -> bool:
    env_value = os.getenv("CELERY_TENANT_RATE_LIMIT_ENABLED")
    if env_value is not None:
        lowered = env_value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    try:  # pragma: no cover - optional settings
        from django.conf import settings

        return bool(getattr(settings, "CELERY_TENANT_RATE_LIMIT_ENABLED", False))
    except Exception:
        return False


def _compute_retry_after_ms(now: float | None = None) -> int:
    timestamp = now if now is not None else time.time()
    window_start = int(timestamp) - (int(timestamp) % 60)
    ttl = 60 - (timestamp - window_start)
    return max(0, int(ttl * 1000))


def _derive_session_salt_from_ids(
    *,
    trace_id: str | None,
    case_id: str | None,
    tenant_id: str | None,
) -> str | None:
    salt_parts = [trace_id, case_id, tenant_id]
    filtered = [str(part) for part in salt_parts if part]
    if filtered:
        return "||".join(filtered)
    return None


def _clone_with_scope(
    signature: Signature,
    scope_kwargs: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> Signature:
    cloned = signature.clone()

    # Inject headers if provided
    if headers:
        existing_options = dict(getattr(cloned, "options", {}) or {})
        existing_headers = dict(existing_options.get("headers") or {})
        existing_headers.update(headers)
        existing_options["headers"] = existing_headers
        cloned.options = existing_options

    if hasattr(cloned, "tasks"):
        tasks = getattr(cloned, "tasks")
        scoped_tasks = [
            _clone_with_scope(sub_sig, scope_kwargs, headers) for sub_sig in tasks
        ]
        cloned.tasks = type(tasks)(scoped_tasks)
        body = getattr(cloned, "body", None)
        if body is not None:
            cloned.body = _clone_with_scope(body, scope_kwargs, headers)
        return cloned

    merged_kwargs = dict(getattr(cloned, "kwargs", {}) or {})
    merged_kwargs.update(scope_kwargs)
    cloned.kwargs = merged_kwargs
    return cloned


def with_scope_apply_async(
    signature: Signature,
    scope: TypingMapping[str, Any],
    *,
    task_id: str | None = None,
    countdown: float | None = None,
    eta: datetime | None = None,
    expires: datetime | float | None = None,
    retry: bool | None = None,
    retry_policy: dict[str, Any] | None = None,
    queue: str | None = None,
    priority: int | None = None,
) -> AsyncResult:
    """Clone a Celery signature and schedule it (trace headers injected).

    Example
    -------
    >>> from celery import chain
    >>> meta = {
    ...     "scope_context": {
    ...         "tenant_id": "t-1",
    ...         "trace_id": "tr-1",
    ...         "invocation_id": "inv-1",
    ...         "run_id": "run-1",
    ...     },
    ...     "business_context": {"case_id": "c-1"},
    ... }
    >>> signature = chain(task_a.s(state={}, meta=meta), task_b.s(state={}, meta=meta))
    >>> scoped = with_scope_apply_async(signature, {})  # trace headers only

    Scope context must be supplied via meta/tool_context; this helper only
    injects trace headers for observability.
    """

    request_headers: dict[str, str] = {}
    trace_id = scope.get("trace_id")
    if trace_id:
        request_headers[X_TRACE_ID_HEADER] = str(trace_id)
    otel_headers: dict[str, str] = {}
    if _OTEL_AVAILABLE:
        TraceContextTextMapPropagator().inject(otel_headers)
        request_headers.update(otel_headers)

    apply_kwargs: dict[str, Any] = {}
    if task_id is not None:
        apply_kwargs["task_id"] = task_id
    if countdown is not None:
        apply_kwargs["countdown"] = countdown
    if eta is not None:
        apply_kwargs["eta"] = eta
    if expires is not None:
        apply_kwargs["expires"] = expires
    if retry is not None:
        apply_kwargs["retry"] = retry
    if retry_policy is not None:
        apply_kwargs["retry_policy"] = retry_policy
    if queue is not None:
        apply_kwargs["queue"] = queue
    if priority is not None:
        apply_kwargs["priority"] = priority

    if not isinstance(signature, Signature):
        if hasattr(signature, "apply_async"):
            return signature.apply_async(**apply_kwargs)
        raise TypeError("signature must be a celery Signature instance")

    if not request_headers:
        return signature.apply_async(**apply_kwargs)

    scoped_signature = _clone_with_scope(signature, {}, request_headers)
    return scoped_signature.apply_async(**apply_kwargs)
