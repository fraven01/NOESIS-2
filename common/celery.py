from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from celery import Task

from .logging import bind_log_context, clear_log_context


class ContextTask(Task):
    """Celery task base class that binds log context from headers/kwargs."""

    abstract = True

    _HEADER_CANDIDATES = {
        "trace_id": ("x-trace-id", "trace_id", "trace-id"),
        "case_id": ("x-case-id", "case_id", "case"),
        "tenant": ("x-tenant-id", "tenant", "tenant_id"),
        "key_alias": ("x-key-alias", "key_alias", "key-alias"),
    }

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
            context.update(self._from_meta(getattr(request, "kwargs", {}).get("meta")))

        context.update(self._from_meta(kwargs.get("meta")))

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

        case = meta.get("case_id") or meta.get("case")
        if case:
            context["case_id"] = self._normalize(case)

        tenant = meta.get("tenant")
        if tenant:
            context["tenant"] = self._normalize(tenant)

        key_alias = meta.get("key_alias")
        if key_alias:
            context["key_alias"] = self._normalize(key_alias)

        return context

    @staticmethod
    def _normalize(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return str(value)
