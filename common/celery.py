from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Mapping as TypingMapping

from celery import Task
from celery.canvas import Signature
from celery.exceptions import SoftTimeLimitExceeded
from pydantic import ValidationError

from .constants import HEADER_CANDIDATE_MAP
from .logging import bind_log_context, clear_log_context
from ai_core.infra.pii_flags import (
    clear_pii_config,
    load_tenant_pii_config,
    set_pii_config,
)
from ai_core.infra.observability import emit_event, update_observation
from ai_core.infra import rate_limit as tenant_rate_limit
from ai_core.infra.policy import (
    clear_session_scope,
    set_session_scope,
    get_session_scope,
)
from ai_core.metrics.task_metrics import record_task_retry
from ai_core.tool_contracts.base import tool_context_from_meta
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
    from opentelemetry.trace.propagation.trace_context import (
        TraceContextTextMapPropagator,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


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
        import sys

        sys.stderr.write(f"DEBUG: ContextTask.__call__ ENTERED for {self.name}\n")
        sys.stderr.flush()
        clear_log_context()
        context = self._gather_context(args, kwargs)
        if context:
            bind_log_context(**context)

        token = None
        if _OTEL_AVAILABLE:
            # Extract W3C trace context from headers and activate it
            request_headers = getattr(self.request, "headers", None) or {}
            # Ensure headers are a dict
            if not isinstance(request_headers, dict):
                request_headers = {}
            ctx = TraceContextTextMapPropagator().extract(carrier=request_headers)
            token = otel_context.attach(ctx)

        try:
            return super().__call__(*args, **kwargs)
        finally:
            if _OTEL_AVAILABLE and token is not None:
                otel_context.detach(token)
            clear_log_context()

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

        if args and not kwargs.get("meta"):
            context.update(self._from_meta(args[0]))

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

        context: dict[str, str] = {}

        try:
            tool_context = tool_context_from_meta(meta)
        except (TypeError, ValueError):
            return {}

        # Infrastructure IDs from scope_context
        trace_id = tool_context.scope.trace_id
        if trace_id:
            context["trace_id"] = self._normalize(trace_id)

        tenant = tool_context.scope.tenant_id
        if tenant:
            context["tenant_id"] = self._normalize(tenant)

        # Business IDs from business_context (BREAKING CHANGE)
        case = tool_context.business.case_id
        if case:
            context["case_id"] = self._normalize(case)

        key_alias = meta.get("key_alias")
        if key_alias:
            context["key_alias"] = self._normalize(key_alias)

        return context

    def _from_tool_context(self, tool_context: Any) -> dict[str, str]:
        if tool_context is None:
            return {}

        context_data: dict[str, Any] | None = None

        if isinstance(tool_context, Mapping):
            context_data = dict(tool_context)
        else:
            attributes = {}
            for field in ("tenant_id", "case_id", "trace_id", "idempotency_key"):
                if hasattr(tool_context, field):
                    attributes[field] = getattr(tool_context, field)
            if attributes:
                context_data = attributes

        if not context_data:
            return {}

        context: dict[str, str] = {}

        tenant_id = context_data.get("tenant_id")
        if tenant_id:
            context["tenant"] = self._normalize(tenant_id)

        case_id = context_data.get("case_id")
        if case_id:
            context["case_id"] = self._normalize(case_id)

        trace_id = context_data.get("trace_id")
        if trace_id:
            context["trace_id"] = self._normalize(trace_id)

        idempotency_key = context_data.get("idempotency_key")
        if idempotency_key:
            context["idempotency_key"] = self._normalize(idempotency_key)

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
    canonical scope upstream. Both parameters are therefore plumbed through to
    worker tasks so the masking layer can associate emitted tokens with the
    right tenant/case context.
    """

    abstract = True

    # Tasks must explicitly opt-in to receiving scope kwargs in ``run``.
    accepts_scope = False

    _SCOPE_FIELDS = ("tenant_id", "case_id", "trace_id", "session_salt")

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
        import sys

        sys.stderr.write(f"DEBUG: ScopedTask.__call__ ENTERED for {self.name}\n")
        sys.stderr.flush()
        call_kwargs = dict(kwargs)
        scope_kwargs: dict[str, Any] = {}
        provided_scope_fields: dict[str, bool] = {}
        for field in self._SCOPE_FIELDS:
            if field in call_kwargs:
                provided_scope_fields[field] = True
                scope_kwargs[field] = call_kwargs.pop(field)
            else:
                provided_scope_fields[field] = False

        explicit_scope = call_kwargs.pop("session_scope", None)

        tenant_id = scope_kwargs.get("tenant_id")
        case_id = scope_kwargs.get("case_id")
        trace_id = scope_kwargs.get("trace_id")
        session_salt = scope_kwargs.get("session_salt")
        scope_override: tuple[str, str, str] | None = None

        if isinstance(explicit_scope, (list, tuple)) and len(explicit_scope) == 3:
            tenant_scope_val, case_scope_val, salt_scope_val = explicit_scope
            scope_override = (
                str(tenant_scope_val) if tenant_scope_val is not None else "",
                str(case_scope_val) if case_scope_val is not None else "",
                str(salt_scope_val) if salt_scope_val is not None else "",
            )
            if not case_id and case_scope_val:
                case_id = case_scope_val
            if not session_salt and salt_scope_val:
                session_salt = salt_scope_val

        if not session_salt:
            salt_parts = [
                str(value) for value in (trace_id, case_id, tenant_id) if value
            ]
            session_salt = "||".join(salt_parts) if salt_parts else None

        if session_salt:
            scope_kwargs["session_salt"] = session_salt

        for field, value in (
            ("tenant_id", tenant_id),
            ("case_id", case_id),
            ("trace_id", trace_id),
        ):
            if value is not None:
                scope_kwargs.setdefault(field, value)

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

        # Forward scope keyword arguments only to tasks that explicitly opted in
        # via ``accepts_scope``. This keeps the masking/logging context intact
        # while avoiding unexpected keyword arguments for tasks that do not
        # handle scope-aware parameters.
        accepts_scope = bool(getattr(self, "accepts_scope", False))
        if accepts_scope:
            for field in self._SCOPE_FIELDS:
                value = scope_kwargs.get(field)
                if value is not None:
                    call_kwargs.setdefault(field, value)

            if explicit_scope is not None:
                call_kwargs.setdefault("session_scope", explicit_scope)

        try:
            pre_execute = getattr(self, "_pre_execute", None)
            if callable(pre_execute):
                pre_execute(args, call_kwargs)
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
        agent_id = _resolve_agent_id(self, kwargs)
        context = self._gather_context(args, kwargs)
        if "tenant" in context and "tenant_id" not in context:
            context["tenant_id"] = context["tenant"]

        payload: dict[str, Any] = {
            "task_id": task_id,
            "task_name": getattr(self, "name", None),
            "agent_id": agent_id,
            "retry_count": retry_count,
            "attempt": attempt,
            "max_retries": getattr(self, "max_retries", None),
            "reason_category": reason_category,
            "exc_type": exc.__class__.__name__,
            "exc_message": str(exc),
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


_SCOPE_KWARG_KEYS = ("tenant_id", "case_id", "trace_id", "session_salt")


def _derive_session_salt(scope: TypingMapping[str, Any]) -> str | None:
    session_salt = scope.get("session_salt")
    if session_salt:
        return str(session_salt)

    salt_parts = [scope.get("trace_id"), scope.get("case_id"), scope.get("tenant_id")]
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
    *args: Any,
    **kwargs: Any,
):
    """Clone a Celery signature and schedule it with the given scope.

    Example
    -------
    >>> from celery import chain
    >>> scoped = with_scope_apply_async(
    ...     chain(task_a.s(), task_b.s()),
    ...     {"tenant_id": "t-1", "case_id": "c-1", "trace_id": "tr-1"},
    ... )

    All tasks in the chain receive ``tenant_id``, ``case_id`` and
    ``session_salt`` keyword arguments so they can establish the masking scope.
    """

    if not isinstance(signature, Signature):
        raise TypeError("signature must be a celery Signature instance")

    scope_kwargs: dict[str, Any] = {}
    for key in _SCOPE_KWARG_KEYS:
        value = scope.get(key)
        if value:
            scope_kwargs[key] = value

    derived_salt = _derive_session_salt(scope)
    if derived_salt:
        scope_kwargs.setdefault("session_salt", derived_salt)

    otel_headers: dict[str, str] = {}
    if _OTEL_AVAILABLE:
        TraceContextTextMapPropagator().inject(otel_headers)

    if not scope_kwargs and not otel_headers:
        return signature.apply_async(*args, **kwargs)

    scoped_signature = _clone_with_scope(signature, scope_kwargs, otel_headers)
    return scoped_signature.apply_async(*args, **kwargs)
