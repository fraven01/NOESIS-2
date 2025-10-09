from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Mapping as TypingMapping

from celery import Task
from celery.canvas import Signature

from .constants import HEADER_CANDIDATE_MAP
from .logging import bind_log_context, clear_log_context
from ai_core.infra.pii_flags import (
    clear_pii_config,
    load_tenant_pii_config,
    set_pii_config,
)
from ai_core.infra.policy import (
    clear_session_scope,
    get_session_scope,
    set_session_scope,
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
        context = self._gather_context(args, kwargs)
        if context:
            bind_log_context(**context)

        try:
            return super().__call__(*args, **kwargs)
        finally:
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
        if not isinstance(meta, Mapping):
            return {}

        context: dict[str, str] = {}

        trace_id = meta.get("trace_id")
        if trace_id:
            context["trace_id"] = self._normalize(trace_id)

        case = meta.get("case_id")
        if case:
            context["case_id"] = self._normalize(case)

        tenant = meta.get("tenant_id")
        if tenant:
            context["tenant"] = self._normalize(tenant)

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
            return super().__call__(*args, **call_kwargs)
        finally:
            clear_pii_config()
            clear_session_scope()


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


def _clone_with_scope(signature: Signature, scope_kwargs: dict[str, Any]) -> Signature:
    cloned = signature.clone()
    if hasattr(cloned, "tasks"):
        tasks = getattr(cloned, "tasks")
        scoped_tasks = [_clone_with_scope(sub_sig, scope_kwargs) for sub_sig in tasks]
        cloned.tasks = type(tasks)(scoped_tasks)
        body = getattr(cloned, "body", None)
        if body is not None:
            cloned.body = _clone_with_scope(body, scope_kwargs)
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

    if not scope_kwargs:
        return signature.apply_async(*args, **kwargs)

    scoped_signature = _clone_with_scope(signature, scope_kwargs)
    return scoped_signature.apply_async(*args, **kwargs)
