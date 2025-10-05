"""Input validation utilities for vector store router calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from common.logging import get_log_context

from ai_core.infra import tracing
from ai_core.rag.limits import (
    CandidatePoolPolicy,
    get_limit_setting,
    normalize_max_candidates,
    normalize_top_k,
    resolve_candidate_pool_policy,
)
from ai_core.rag.selector_utils import normalise_selector_value

_ROUTER_DOC_HINT = "See README.md (Fehlercodes Abschnitt) for remediation guidance."


logger = logging.getLogger(__name__)


class RouterInputError(ValueError):
    """Raised when router inputs are invalid."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        context: Mapping[str, object | None] | None = None,
    ) -> None:
        detail = f"{code}: {message}. {_ROUTER_DOC_HINT}"
        super().__init__(detail)
        self.code = code
        self.field = field
        self.message = message
        self.context = dict(context or {})


class RouterInputErrorCode:
    """Machine-readable router input error codes."""

    TENANT_REQUIRED = "ROUTER_TENANT_REQUIRED"
    TOP_K_INVALID = "ROUTER_TOP_K_INVALID"
    MAX_CANDIDATES_INVALID = "ROUTER_MAX_CANDIDATES_INVALID"
    MAX_CANDIDATES_LT_TOP_K = "ROUTER_MAX_CANDIDATES_LT_TOP_K"


def map_router_error_to_status(code: str) -> int:
    """Return the HTTP status code associated with a router validation error."""

    client_error_codes = {
        RouterInputErrorCode.TENANT_REQUIRED,
        RouterInputErrorCode.TOP_K_INVALID,
        RouterInputErrorCode.MAX_CANDIDATES_INVALID,
        RouterInputErrorCode.MAX_CANDIDATES_LT_TOP_K,
    }
    if code in client_error_codes:
        return 400
    return 400


@dataclass(frozen=True, slots=True)
class SearchValidationResult:
    """Sanitised router inputs and their effective limits."""

    tenant_id: str
    process: str | None
    doc_class: str | None
    top_k: int | None
    max_candidates: int | None
    effective_top_k: int
    top_k_source: str
    effective_max_candidates: int
    max_candidates_source: str
    context: dict[str, object | None]


def _normalise_optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value).strip() or None


def _coerce_positive_int(
    value: Any,
    *,
    code: str,
    field: str,
    context: Mapping[str, object | None],
) -> int:
    if isinstance(value, bool):
        candidate = int(value)
    elif isinstance(value, int):
        candidate = int(value)
    elif isinstance(value, float):
        if not value.is_integer():
            raise RouterInputError(
                code,
                f"{field} must be a positive integer",
                field=field,
                context=context,
            )
        candidate = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise RouterInputError(
                code,
                f"{field} must be a positive integer",
                field=field,
                context=context,
            )
        try:
            candidate = int(text)
        except ValueError as exc:  # pragma: no cover - defensive
            raise RouterInputError(
                code,
                f"{field} must be a positive integer",
                field=field,
                context=context,
            ) from exc
    else:
        raise RouterInputError(
            code,
            f"{field} must be a positive integer",
            field=field,
            context=context,
        )
    if candidate <= 0:
        raise RouterInputError(
            code,
            f"{field} must be a positive integer",
            field=field,
            context=context,
        )
    return candidate


def validate_search_inputs(
    *,
    tenant_id: object | None,
    process: str | None = None,
    doc_class: str | None = None,
    top_k: object | None = None,
    max_candidates: object | None = None,
) -> SearchValidationResult:
    """Validate router inputs and return sanitised values."""

    tenant = _normalise_optional_text(tenant_id)
    sanitized_process = normalise_selector_value(process)
    sanitized_doc_class = normalise_selector_value(doc_class)
    context = {
        "tenant_id": tenant,
        "process": sanitized_process,
        "doc_class": sanitized_doc_class,
    }
    policy = resolve_candidate_pool_policy()
    context["candidate_policy"] = policy.value
    if not tenant:
        raise RouterInputError(
            RouterInputErrorCode.TENANT_REQUIRED,
            "tenant_id is required for retrieval requests",
            field="tenant_id",
            context=context,
        )

    sanitized_top_k: int | None
    if top_k is None:
        sanitized_top_k = None
    elif isinstance(top_k, str) and not top_k.strip():
        sanitized_top_k = None
    else:
        context["top_k"] = top_k
        sanitized_top_k = _coerce_positive_int(
            top_k,
            code=RouterInputErrorCode.TOP_K_INVALID,
            field="top_k",
            context=context,
        )
        context["top_k"] = sanitized_top_k

    sanitized_max: int | None
    if max_candidates is None:
        sanitized_max = None
    elif isinstance(max_candidates, str) and not max_candidates.strip():
        sanitized_max = None
    else:
        context["max_candidates"] = max_candidates
        sanitized_max = _coerce_positive_int(
            max_candidates,
            code=RouterInputErrorCode.MAX_CANDIDATES_INVALID,
            field="max_candidates",
            context=context,
        )
        context["max_candidates"] = sanitized_max

    normalized_top_k, top_k_source = normalize_top_k(
        sanitized_top_k,
        default=5,
        minimum=1,
        maximum=10,
        return_source=True,
    )
    context["top_k_effective"] = normalized_top_k
    context["top_k_source"] = top_k_source

    max_candidates_cap = int(get_limit_setting("RAG_MAX_CANDIDATES", 200))
    context["max_candidates_cap"] = max_candidates_cap

    effective_max_candidates: int
    max_candidates_source: str
    if sanitized_max is None:
        effective_max_candidates, max_candidates_source = normalize_max_candidates(
            normalized_top_k,
            None,
            max_candidates_cap,
            return_source=True,
        )
        context.setdefault("max_candidates", effective_max_candidates)
    else:
        effective_max_candidates = sanitized_max
        max_candidates_source = "from_state"
        if effective_max_candidates < normalized_top_k:
            if policy is CandidatePoolPolicy.NORMALIZE:
                requested = effective_max_candidates
                effective_max_candidates = normalized_top_k
                context["max_candidates"] = effective_max_candidates
                context["candidate_policy_action"] = "normalized_to_top_k"
                logger.warning(
                    "rag.hybrid.candidate_pool.normalized",
                    extra={
                        "tenant": tenant,
                        "routing_process": sanitized_process,
                        "routing_doc_class": sanitized_doc_class,
                        "requested": requested,
                        "top_k": normalized_top_k,
                    },
                )
            else:
                detail_context = dict(context)
                detail_context.update(
                    {
                        "top_k": normalized_top_k,
                        "max_candidates": effective_max_candidates,
                    }
                )
                raise RouterInputError(
                    RouterInputErrorCode.MAX_CANDIDATES_LT_TOP_K,
                    "max_candidates must be greater than or equal to top_k",
                    field="max_candidates",
                    context=detail_context,
                )
        if (
            max_candidates_cap is not None
            and effective_max_candidates > max_candidates_cap
        ):
            effective_max_candidates = max_candidates_cap

    context["max_candidates_effective"] = effective_max_candidates
    context["max_candidates_source"] = max_candidates_source

    normalized_max_for_result = (
        effective_max_candidates if sanitized_max is not None else None
    )

    return SearchValidationResult(
        tenant_id=tenant,
        process=sanitized_process,
        doc_class=sanitized_doc_class,
        top_k=sanitized_top_k,
        max_candidates=normalized_max_for_result,
        effective_top_k=normalized_top_k,
        top_k_source=top_k_source,
        effective_max_candidates=effective_max_candidates,
        max_candidates_source=max_candidates_source,
        context=context,
    )


def emit_router_validation_failure(error: RouterInputError) -> None:
    """Log and trace a router validation failure with mandatory tags."""

    context = dict(error.context)
    metadata = {
        "tenant": context.get("tenant_id"),
        "process": context.get("process"),
        "doc_class": context.get("doc_class"),
        "top_k": context.get("top_k"),
        "max_candidates": context.get("max_candidates"),
        "error_code": error.code,
    }

    log_context = get_log_context()
    trace_id = log_context.get("trace_id")
    if trace_id:
        tracing.emit_span(
            trace_id=trace_id,
            node_name="rag.router.validation_failed",
            metadata=metadata,
        )


__all__ = [
    "RouterInputError",
    "RouterInputErrorCode",
    "SearchValidationResult",
    "emit_router_validation_failure",
    "map_router_error_to_status",
    "validate_search_inputs",
]
