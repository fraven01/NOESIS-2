from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterator, Literal, Mapping, Sequence, TypedDict

import requests

from ai_core.infra.config import get_config
from ai_core.infra.circuit_breaker import get_litellm_circuit_breaker
from ai_core.infra.observability import observe_span, update_observation
from ai_core.infra import ledger
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)
from common.logging import get_logger, mask_value
from .routing import resolve

logger = get_logger(__name__)


class LlmClientError(Exception):
    """Base exception for LLM client errors."""

    def __init__(
        self,
        detail: str | None = None,
        *,
        status: int | None = None,
        code: str | None = None,
    ) -> None:
        self.detail = detail
        self.status = status
        self.code = code
        message = detail or "LLM client error"
        parts: list[str] = []
        if status is not None:
            parts.append(f"status={status}")
        if code:
            parts.append(f"code={code}")
        if parts:
            message = f"{message} ({', '.join(parts)})"
        super().__init__(message)


class RateLimitError(LlmClientError):
    """Raised when the LLM client is rate limited."""


class LlmUpstreamError(LlmClientError):
    """Raised when the LiteLLM upstream is unavailable or returns 5xx."""


class LlmTimeoutError(LlmClientError, TimeoutError):
    """Raised when the LLM client exceeds the configured timeout."""


class StreamChunk(TypedDict, total=False):
    event: Literal["delta", "final", "error"]
    text: str | None
    usage: Mapping[str, Any] | None
    cost: Mapping[str, Any] | None
    cost_usd: float | None
    latency_ms: float | None
    cache_hit: bool | None
    model: str | None
    finish_reason: str | None
    error: str | None


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_text(resp: requests.Response | None) -> str | None:
    if resp is None:
        return None
    resp_text = getattr(resp, "text", "")
    if resp_text is None:
        return ""
    if not isinstance(resp_text, str):
        resp_text = str(resp_text)
    return resp_text.strip()


_DEFAULT_LLM_TIMEOUT_S = 30.0


def _resolve_timeout_value(
    timeouts: Mapping[str, int], keys: Sequence[str], default: float | None
) -> float | None:
    for key in keys:
        raw_value = timeouts.get(key)
        if raw_value is None:
            continue
        try:
            candidate = float(raw_value)
        except (TypeError, ValueError):
            continue
        if candidate <= 0:
            return None
        return candidate
    return default


def _resolve_timeouts(label: str, cfg, *, stream: bool) -> tuple[float, float] | None:
    timeouts = cfg.timeouts or {}
    keys: list[str] = []
    if stream:
        keys.extend([f"{label}_stream", "stream"])
    keys.extend([label, "default"])
    timeout_s = _resolve_timeout_value(timeouts, keys, _DEFAULT_LLM_TIMEOUT_S)
    if timeout_s is None:
        return None
    return (timeout_s, timeout_s)


def _normalise_text_parts(value: object) -> str | None:
    """Best-effort conversion of nested content structures into plain text."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or ""
    if isinstance(value, Mapping):
        # Providers may nest text under various keys; try common variants.
        for key in ("text", "content", "output_text", "value"):
            if key in value:
                candidate = _normalise_text_parts(value[key])
                if candidate:
                    return candidate
        if "parts" in value:
            candidate = _normalise_text_parts(value["parts"])
            if candidate:
                return candidate
        return None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            candidate = _normalise_text_parts(item)
            if candidate:
                parts.append(candidate)
        if parts:
            return "\n".join(parts)
        return None
    try:
        return str(value).strip()
    except Exception:
        return None


def _normalise_stream_text_parts(value: object) -> str | None:
    """Best-effort conversion for streaming deltas (preserve whitespace)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "output_text", "value"):
            if key in value:
                candidate = _normalise_stream_text_parts(value[key])
                if candidate is not None:
                    return candidate
        if "parts" in value:
            candidate = _normalise_stream_text_parts(value["parts"])
            if candidate is not None:
                return candidate
        return None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            candidate = _normalise_stream_text_parts(item)
            if candidate is not None:
                parts.append(candidate)
        if parts:
            return "".join(parts)
        return None
    try:
        return str(value)
    except Exception:
        return None


def _coerce_choice_text(choice: Mapping[str, Any]) -> str:
    """Extract assistant text from a chat completion choice."""

    message = choice.get("message") or {}
    text = None
    if isinstance(message, Mapping):
        for key in ("content", "text", "parts"):
            if key in message:
                text = _normalise_text_parts(message[key])
                if text:
                    break
    if not text:
        # Some providers (e.g. text completion fallback) surface text directly
        alt = choice.get("text")
        if alt is not None:
            text = _normalise_text_parts(alt)
    if not text:
        raise KeyError("content")
    return text


def _coerce_stream_delta_text(choice: Mapping[str, Any]) -> str | None:
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        for key in ("content", "text", "parts"):
            if key in delta:
                candidate = _normalise_stream_text_parts(delta[key])
                if candidate is not None:
                    return candidate
    message = choice.get("message")
    if isinstance(message, Mapping):
        for key in ("content", "text", "parts"):
            if key in message:
                candidate = _normalise_stream_text_parts(message[key])
                if candidate is not None:
                    return candidate
    alt = choice.get("text")
    if alt is not None:
        return _normalise_stream_text_parts(alt)
    return None


def _iter_sse_lines(resp: requests.Response) -> Iterator[str]:
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        if not isinstance(raw, str):
            try:
                raw = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue
        text = raw.rstrip("\r\n")
        if not text.strip():
            continue
        stripped = text.lstrip()
        if stripped.startswith("data:"):
            text = stripped[5:]
            if text.startswith(" "):
                text = text[1:]
        if text:
            yield text


def _safe_update_observation(**fields: Any) -> None:
    try:
        update_observation(**fields)
    except Exception:
        pass


def _interpret_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "hit"}:
            return True
        if lowered in {"0", "false", "no", "n", "miss"}:
            return False
    return None


def _detect_cache_hit(
    resp: requests.Response | None, data: Mapping[str, Any]
) -> bool | None:
    cache_hit: bool | None = None
    headers = getattr(resp, "headers", {}) or {}
    if isinstance(headers, Mapping):
        for key in ("x-litellm-cache-hit", "x-cache-hit", "x-cache"):
            if key in headers:
                cache_hit = _interpret_bool(headers.get(key))
                if cache_hit is not None:
                    break

    if cache_hit is not None:
        return cache_hit

    for key in ("cache_hit", "cacheHit"):
        if key in data:
            cache_hit = _interpret_bool(data.get(key))
            if cache_hit is not None:
                return cache_hit

    cache_section = data.get("cache")
    if isinstance(cache_section, Mapping):
        for key in ("hit", "cache_hit"):
            if key in cache_section:
                cache_hit = _interpret_bool(cache_section.get(key))
                if cache_hit is not None:
                    return cache_hit

    return None


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text[:limit]


def _supports_temperature(model_id: str) -> bool:
    return "gpt-5" not in model_id


@observe_span(name="llm.call")
def call(
    label: str,
    prompt: str,
    metadata: Dict[str, Any],
    *,
    response_format: Dict[str, Any] | None = None,
    extra_params: Dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> Dict[str, Any]:
    """Call the LLM via LiteLLM proxy using a routing ``label``.

    Parameters
    ----------
    label:
        Routing label that resolves to a model id.
    prompt:
        Prompt text which will be PII-masked before sending.
    metadata:
        Dict containing at least ``tenant_id``, ``case_id``, and ``trace_id``,
        plus optional ``user_id`` for telemetry attribution.
    response_format:
        Optional response format specification, e.g. ``{"type": "json_object"}``
        for structured JSON output.
    extra_params:
        Optional dict of additional parameters to pass to the LLM API.
        These are merged into the request payload after standard parameters.
    """

    model_id = resolve(label)
    cfg = get_config()
    url = f"{cfg.litellm_base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.litellm_api_key}"}
    tenant_value = metadata.get("tenant_id")
    case_value = metadata.get("case_id")
    user_value = metadata.get("user_id")
    propagated_headers = {
        X_TRACE_ID_HEADER: metadata.get("trace_id"),
        X_CASE_ID_HEADER: case_value,
        X_TENANT_ID_HEADER: tenant_value,
    }
    key_alias = metadata.get("key_alias")
    if key_alias:
        propagated_headers[X_KEY_ALIAS_HEADER] = key_alias
    headers.update({k: v for k, v in propagated_headers.items() if v})
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Optional generation controls via ENV to keep latency predictable
    try:
        max_tokens_env = os.getenv("LITELLM_MAX_TOKENS")
        if max_tokens_env is not None:
            max_tokens_val = int(max_tokens_env)
            if max_tokens_val > 0:
                payload["max_tokens"] = max_tokens_val
    except Exception:
        pass
    if _supports_temperature(model_id):
        try:
            temperature_env = os.getenv("LITELLM_TEMPERATURE")
            if temperature_env is not None:
                payload["temperature"] = float(temperature_env)
        except Exception:
            pass

    # Add response_format if specified (e.g., {"type": "json_object"})
    if response_format is not None:
        payload["response_format"] = response_format

    # Merge extra_params last to allow overrides
    if extra_params:
        payload.update(extra_params)

    prompt_version = metadata.get("prompt_version") or "default"
    case_id = case_value or "unknown-case"
    idempotency_key = f"{case_id}:{label}:{prompt_version}"
    log_extra = {
        "trace_id": mask_value(metadata.get("trace_id")),
        "case_id": mask_value(case_value),
        "tenant_id": mask_value(tenant_value),
        "key_alias": mask_value(metadata.get("key_alias")),
    }
    # Attach lightweight context to the current observation (no PII payloads)
    _safe_update_observation(
        tags=["llm", "litellm", f"label:{label}", f"model:{model_id}"],
        user_id=str(user_value) if user_value else None,
        session_id=str(case_value) if case_value else None,
        metadata={
            "trace_id": metadata.get("trace_id"),
            "prompt_version": prompt_version,
            "tenant_id": tenant_value,
        },
    )

    status: int | None = None
    text: str | None = None
    data: Dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    cache_hit: bool | None = None
    masked_prompt_preview = _truncate(prompt, 512)
    resp: requests.Response | None = None
    attempt_headers = headers.copy()
    attempt_headers[IDEMPOTENCY_KEY_HEADER] = idempotency_key
    if timeout_s is not None and timeout_s > 0:
        timeout = (timeout_s, timeout_s)
    else:
        timeout = _resolve_timeouts(label, cfg, stream=False)
    breaker = get_litellm_circuit_breaker()
    if not breaker.allow_request():
        retry_after_ms = None
        next_retry_at = breaker.next_retry_at
        if next_retry_at is not None:
            retry_after_ms = max(0, int((next_retry_at - time.time()) * 1000))
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "error.type": "LlmUpstreamError",
                "error.message": "LiteLLM circuit breaker open",
                "circuit_breaker.state": breaker.state,
                "retry_after_ms": retry_after_ms,
            }
        )
        raise LlmUpstreamError(
            "LiteLLM circuit breaker open",
            status=503,
            code="circuit_open",
        )

    try:
        start_ts = time.perf_counter()
        resp = requests.post(
            url,
            headers=attempt_headers,
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
    except requests.Timeout as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        status = None
        breaker.record_failure(reason="timeout")
        logger.warning(
            "llm request timeout",
            exc_info=exc,
            extra={**log_extra, "status": status, "timeout_s": timeout},
        )
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmTimeoutError",
                "error.message": _truncate(str(exc) or "LLM request timeout", 256),
                "provider.http_status": status,
                "timeout_s": timeout,
            }
        )
        raise LlmTimeoutError(
            str(exc) or "LLM request timeout",
            status=None,
            code="timeout",
        ) from exc
    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        status = None
        breaker.record_failure(reason="request_error")
        logger.warning(
            "llm request error",
            exc_info=exc,
            extra={**log_extra, "status": status},
        )
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": type(exc).__name__,
                "error.message": _truncate(str(exc) or "LLM client error", 256),
                "provider.http_status": status,
            }
        )
        raise LlmUpstreamError(str(exc) or "LLM client error") from exc
    else:
        status = resp.status_code
        latency_ms = (time.perf_counter() - start_ts) * 1000.0

    if status and 500 <= status < 600:
        breaker.record_failure(reason="upstream_5xx")
        logger.warning("llm 5xx response", extra={**log_extra, "status": status})
        payload_json = _safe_json(resp)
        cache_hit = _detect_cache_hit(
            resp, payload_json if isinstance(payload_json, Mapping) else {}
        )
        detail = payload_json.get("detail") or _safe_text(resp)
        status_val = payload_json.get("status") or status
        code = payload_json.get("code")
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmClientError",
                "error.message": _truncate(detail or "LLM client error", 256),
                "provider.http_status": status_val,
            }
        )
        raise LlmUpstreamError(
            detail or "LLM client error", status=status_val, code=code
        ) from None

    if status == 429:
        breaker.record_failure(reason="rate_limit")
        logger.warning("llm rate limited", extra={**log_extra, "status": status})
        payload_json = _safe_json(resp)
        cache_hit = _detect_cache_hit(
            resp, payload_json if isinstance(payload_json, Mapping) else {}
        )
        detail = payload_json.get("detail") or _safe_text(resp)
        status_val = payload_json.get("status") or status
        code = payload_json.get("code")
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "RateLimitError",
                "error.message": _truncate(detail or "LLM client error", 256),
                "provider.http_status": status_val,
            }
        )
        raise RateLimitError(
            detail or "LLM client error", status=status_val, code=code
        ) from None

    if status and 400 <= status < 500:
        logger.warning("llm 4xx response", extra={**log_extra, "status": status})
        payload_json = _safe_json(resp)
        cache_hit = _detect_cache_hit(
            resp, payload_json if isinstance(payload_json, Mapping) else {}
        )
        detail = payload_json.get("detail") or _safe_text(resp)
        status_val = payload_json.get("status") or status
        code = payload_json.get("code")
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmClientError",
                "error.message": _truncate(detail or "LLM client error", 256),
                "provider.http_status": status_val,
            }
        )
        raise LlmClientError(detail or "LLM client error", status=status_val, code=code)

    data = resp.json()
    cache_hit = _detect_cache_hit(resp, data if isinstance(data, Mapping) else {})
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], Mapping):
        raw_finish = choices[0].get("finish_reason")
        finish_reason = str(raw_finish) if raw_finish is not None else None
    try:
        text = _coerce_choice_text(choices[0])
    except (KeyError, IndexError, TypeError):
        message = "LLM response missing content"
        if finish_reason:
            message = f"{message} (finish_reason={finish_reason})"
        logger.warning("llm.response_missing_content", extra=log_extra)
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmClientError",
                "error.message": _truncate(message, 256),
                "provider.http_status": status,
            }
        )
        breaker.record_failure(reason="invalid_response")
        raise LlmClientError(message, status=status) from None

    choices = data.get("choices") or []
    try:
        text = text if text is not None else _coerce_choice_text(choices[0])
    except (KeyError, IndexError, TypeError):
        message = "LLM response missing content"
        if finish_reason:
            message = f"{message} (finish_reason={finish_reason})"
        logger.warning("llm.response_missing_content", extra=log_extra)
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmClientError",
                "error.message": _truncate(message, 256),
                "provider.http_status": status,
            }
        )
        breaker.record_failure(reason="invalid_response")
        raise LlmClientError(message, status=status) from None
    usage_raw = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
    prompt_tokens = usage_raw.get("prompt_tokens", 0) or 0
    completion_tokens = usage_raw.get("completion_tokens", 0) or 0
    total_tokens = usage_raw.get("total_tokens")
    if total_tokens is None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    cost_data: Mapping[str, Any] | None = None
    if isinstance(usage_raw, Mapping):
        usage_cost = usage_raw.get("cost")
        if isinstance(usage_cost, Mapping):
            cost_data = usage_cost
    response_cost = data.get("cost")
    if isinstance(response_cost, Mapping) and cost_data is None:
        cost_data = response_cost
    cost_usd: float | None = None
    if isinstance(cost_data, Mapping):
        for key in ("usd", "USD", "total_cost"):
            value = cost_data.get(key)
            try:
                cost_usd = float(value)  # type: ignore[arg-type]
                break
            except (TypeError, ValueError):
                continue
    usage: dict[str, Any] = dict(usage_raw) if isinstance(usage_raw, Mapping) else {}
    if cost_data is not None:
        usage_cost: dict[str, Any] = {}
        raw_usage_cost = usage.get("cost")
        if isinstance(raw_usage_cost, Mapping):
            usage_cost.update(raw_usage_cost)
        if cost_usd is not None:
            usage_cost.setdefault("usd", cost_usd)
            usage_cost.setdefault("total", cost_usd)
        if usage_cost:
            usage["cost"] = usage_cost
    if cost_usd is not None:
        usage.setdefault("cost_usd", cost_usd)
    result = {
        "text": text,
        "usage": usage,
        "model": model_id,
        "prompt_version": metadata.get("prompt_version"),
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
    }
    if cost_data is not None:
        result["cost"] = cost_data

    ledger_payload = {
        "tenant_id": tenant_value,
        "case_id": case_value,
        "trace_id": metadata.get("trace_id"),
        "label": label,
        "model": model_id,
        "usage": usage,
        "ts": time.time(),
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
    }
    if cost_data is not None:
        ledger_payload["cost"] = cost_data
    if cost_usd is not None:
        ledger_payload["cost_usd"] = cost_usd

    ledger_logger = (
        metadata.get("ledger_logger") if isinstance(metadata, Mapping) else None
    )
    if callable(ledger_logger):
        try:
            ledger_logger(ledger_payload)
        except Exception:
            logger.debug("ledger_logger failed", exc_info=True)

    ledger.record(ledger_payload)

    observation_metadata = {
        "status": "success",
        "model.id": model_id,
        "usage.prompt_tokens": prompt_tokens,
        "usage.completion_tokens": completion_tokens,
        "usage.total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "input.masked_prompt": masked_prompt_preview,
    }
    if cost_usd is not None:
        observation_metadata["cost.usd"] = cost_usd

    _safe_update_observation(metadata=observation_metadata)

    breaker.record_success()
    return result


@observe_span(name="llm.call_stream")
def call_stream(
    label: str,
    prompt: str,
    metadata: Dict[str, Any],
    *,
    extra_params: Dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> Iterator[StreamChunk]:
    """Stream tokens from the LLM via LiteLLM proxy (streaming mode).

    Parameters
    ----------
    label:
        Routing label that resolves to a model id.
    prompt:
        Prompt text which will be PII-masked before sending.
    metadata:
        Dict containing at least ``tenant_id``, ``case_id``, and ``trace_id``,
        plus optional ``user_id`` for telemetry attribution.
    extra_params:
        Optional dict of additional parameters to pass to the LLM API.
        These are merged into the request payload after standard parameters.
    """

    model_id = resolve(label)
    cfg = get_config()
    url = f"{cfg.litellm_base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.litellm_api_key}"}
    tenant_value = metadata.get("tenant_id")
    case_value = metadata.get("case_id")
    user_value = metadata.get("user_id")
    propagated_headers = {
        X_TRACE_ID_HEADER: metadata.get("trace_id"),
        X_CASE_ID_HEADER: case_value,
        X_TENANT_ID_HEADER: tenant_value,
    }
    key_alias = metadata.get("key_alias")
    if key_alias:
        propagated_headers[X_KEY_ALIAS_HEADER] = key_alias
    headers.update({k: v for k, v in propagated_headers.items() if v})
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    try:
        max_tokens_env = os.getenv("LITELLM_MAX_TOKENS")
        if max_tokens_env is not None:
            max_tokens_val = int(max_tokens_env)
            if max_tokens_val > 0:
                payload["max_tokens"] = max_tokens_val
    except Exception:
        pass
    if _supports_temperature(model_id):
        try:
            temperature_env = os.getenv("LITELLM_TEMPERATURE")
            if temperature_env is not None:
                payload["temperature"] = float(temperature_env)
        except Exception:
            pass

    # Merge extra_params last to allow overrides
    if extra_params:
        payload.update(extra_params)

    prompt_version = metadata.get("prompt_version") or "default"
    case_id = case_value or "unknown-case"
    idempotency_key = f"{case_id}:{label}:{prompt_version}"
    log_extra = {
        "trace_id": mask_value(metadata.get("trace_id")),
        "case_id": mask_value(case_value),
        "tenant_id": mask_value(tenant_value),
        "key_alias": mask_value(metadata.get("key_alias")),
    }
    _safe_update_observation(
        tags=["llm", "litellm", f"label:{label}", f"model:{model_id}"],
        user_id=str(user_value) if user_value else None,
        session_id=str(case_value) if case_value else None,
        metadata={
            "trace_id": metadata.get("trace_id"),
            "prompt_version": prompt_version,
            "tenant_id": tenant_value,
        },
    )

    status: int | None = None
    cache_hit: bool | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    usage_final: Mapping[str, Any] | None = None
    usage_accum: dict[str, Any] | None = None
    cost_data: Mapping[str, Any] | None = None
    cost_usd: float | None = None
    masked_prompt_preview = _truncate(prompt, 512)
    resp: requests.Response | None = None
    attempt_headers = headers.copy()
    attempt_headers[IDEMPOTENCY_KEY_HEADER] = idempotency_key
    if timeout_s is not None and timeout_s > 0:
        timeout = (timeout_s, timeout_s)
    else:
        timeout = _resolve_timeouts(label, cfg, stream=True)
    breaker = get_litellm_circuit_breaker()
    if not breaker.allow_request():
        retry_after_ms = None
        next_retry_at = breaker.next_retry_at
        if next_retry_at is not None:
            retry_after_ms = max(0, int((next_retry_at - time.time()) * 1000))
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "error.type": "LlmUpstreamError",
                "error.message": "LiteLLM circuit breaker open",
                "circuit_breaker.state": breaker.state,
                "retry_after_ms": retry_after_ms,
            }
        )
        err = LlmUpstreamError(
            "LiteLLM circuit breaker open",
            status=503,
            code="circuit_open",
        )
        yield {"event": "error", "error": str(err)}
        raise err

    try:
        start_ts = time.perf_counter()
        resp = requests.post(
            url,
            headers=attempt_headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        status = None
        breaker.record_failure(reason="timeout")
        logger.warning(
            "llm request timeout",
            exc_info=exc,
            extra={**log_extra, "status": status, "timeout_s": timeout},
        )
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmTimeoutError",
                "error.message": _truncate(str(exc) or "LLM request timeout", 256),
                "provider.http_status": status,
                "timeout_s": timeout,
            }
        )
        err = LlmTimeoutError(
            str(exc) or "LLM request timeout",
            status=None,
            code="timeout",
        )
        yield {"event": "error", "error": str(err)}
        raise err from exc
    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        status = None
        breaker.record_failure(reason="request_error")
        logger.warning(
            "llm request error",
            exc_info=exc,
            extra={**log_extra, "status": status},
        )
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": type(exc).__name__,
                "error.message": _truncate(str(exc) or "LLM client error", 256),
                "provider.http_status": status,
            }
        )
        err = LlmUpstreamError(str(exc) or "LLM client error")
        yield {"event": "error", "error": str(err)}
        raise err from exc
    else:
        status = resp.status_code

    if status and 500 <= status < 600:
        breaker.record_failure(reason="upstream_5xx")
        logger.warning("llm 5xx response", extra={**log_extra, "status": status})
        payload_json = _safe_json(resp)
        cache_hit = _detect_cache_hit(
            resp, payload_json if isinstance(payload_json, Mapping) else {}
        )
        detail = payload_json.get("detail") or _safe_text(resp)
        status_val = payload_json.get("status") or status
        code = payload_json.get("code")
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmClientError",
                "error.message": _truncate(detail or "LLM client error", 256),
                "provider.http_status": status_val,
            }
        )
        err = LlmUpstreamError(
            detail or "LLM client error", status=status_val, code=code
        )
        if resp is not None:
            resp.close()
        yield {"event": "error", "error": str(err)}
        raise err

    if status == 429:
        breaker.record_failure(reason="rate_limit")
        logger.warning("llm rate limited", extra={**log_extra, "status": status})
        payload_json = _safe_json(resp)
        cache_hit = _detect_cache_hit(
            resp, payload_json if isinstance(payload_json, Mapping) else {}
        )
        detail = payload_json.get("detail") or _safe_text(resp)
        status_val = payload_json.get("status") or status
        code = payload_json.get("code")
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "RateLimitError",
                "error.message": _truncate(detail or "LLM client error", 256),
                "provider.http_status": status_val,
            }
        )
        err = RateLimitError(detail or "LLM client error", status=status_val, code=code)
        if resp is not None:
            resp.close()
        yield {"event": "error", "error": str(err)}
        raise err

    if status and 400 <= status < 500:
        logger.warning("llm 4xx response", extra={**log_extra, "status": status})
        payload_json = _safe_json(resp)
        cache_hit = _detect_cache_hit(
            resp, payload_json if isinstance(payload_json, Mapping) else {}
        )
        detail = payload_json.get("detail") or _safe_text(resp)
        status_val = payload_json.get("status") or status
        code = payload_json.get("code")
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmClientError",
                "error.message": _truncate(detail or "LLM client error", 256),
                "provider.http_status": status_val,
            }
        )
        err = LlmClientError(detail or "LLM client error", status=status_val, code=code)
        if resp is not None:
            resp.close()
        yield {"event": "error", "error": str(err)}
        raise err

    try:
        cache_hit = _detect_cache_hit(resp, {}) if resp is not None else None
        for line in _iter_sse_lines(resp):
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, Mapping):
                continue
            if cache_hit is None and resp is not None:
                cache_hit = _detect_cache_hit(resp, data)
            usage_payload = data.get("usage")
            if isinstance(usage_payload, Mapping):
                usage_final = dict(usage_payload)
                if usage_accum is None:
                    usage_accum = {}
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    raw_value = usage_payload.get(key)
                    try:
                        value = int(raw_value)
                    except (TypeError, ValueError):
                        continue
                    usage_accum[key] = usage_accum.get(key, 0) + value
                for key, value in usage_payload.items():
                    if key in {"prompt_tokens", "completion_tokens", "total_tokens"}:
                        continue
                    usage_accum.setdefault(key, value)

            cost_payload = data.get("cost")
            if isinstance(cost_payload, Mapping):
                cost_data = dict(cost_payload)
            elif isinstance(usage_payload, Mapping):
                usage_cost = usage_payload.get("cost")
                if isinstance(usage_cost, Mapping):
                    cost_data = dict(usage_cost)

            choices = data.get("choices") or []
            if choices and isinstance(choices[0], Mapping):
                raw_finish = choices[0].get("finish_reason")
                if raw_finish is not None:
                    finish_reason = str(raw_finish)
                delta_text = _coerce_stream_delta_text(choices[0])
                if delta_text:
                    yield {"event": "delta", "text": delta_text}
    except requests.Timeout as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        breaker.record_failure(reason="timeout")
        logger.warning(
            "llm request timeout",
            exc_info=exc,
            extra={**log_extra, "status": status, "timeout_s": timeout},
        )
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmTimeoutError",
                "error.message": _truncate(str(exc) or "LLM request timeout", 256),
                "provider.http_status": status,
                "timeout_s": timeout,
            }
        )
        err = LlmTimeoutError(
            str(exc) or "LLM request timeout",
            status=None,
            code="timeout",
        )
        yield {"event": "error", "error": str(err)}
        raise err from exc
    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        breaker.record_failure(reason="request_error")
        logger.warning(
            "llm request error",
            exc_info=exc,
            extra={**log_extra, "status": status},
        )
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": type(exc).__name__,
                "error.message": _truncate(str(exc) or "LLM client error", 256),
                "provider.http_status": status,
            }
        )
        err = LlmUpstreamError(str(exc) or "LLM client error")
        yield {"event": "error", "error": str(err)}
        raise err from exc
    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    latency_ms = (time.perf_counter() - start_ts) * 1000.0
    usage_raw = usage_final or usage_accum
    usage: dict[str, Any] | None = None
    if isinstance(usage_raw, Mapping):
        usage = dict(usage_raw)

    if usage is not None:
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            usage["total_tokens"] = total_tokens
        if cost_data is None:
            usage_cost = usage.get("cost")
            if isinstance(usage_cost, Mapping):
                cost_data = dict(usage_cost)

    if isinstance(cost_data, Mapping):
        for key in ("usd", "USD", "total_cost"):
            value = cost_data.get(key)
            try:
                cost_usd = float(value)  # type: ignore[arg-type]
                break
            except (TypeError, ValueError):
                continue

    if usage is not None and cost_usd is not None:
        usage_cost: dict[str, Any] = {}
        raw_usage_cost = usage.get("cost")
        if isinstance(raw_usage_cost, Mapping):
            usage_cost.update(raw_usage_cost)
        usage_cost.setdefault("usd", cost_usd)
        usage_cost.setdefault("total", cost_usd)
        usage["cost"] = usage_cost

    ledger_payload = {
        "tenant_id": tenant_value,
        "case_id": case_value,
        "trace_id": metadata.get("trace_id"),
        "label": label,
        "model": model_id,
        "usage": usage,
        "ts": time.time(),
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
    }
    if cost_data is not None:
        ledger_payload["cost"] = cost_data
    if cost_usd is not None:
        ledger_payload["cost_usd"] = cost_usd

    ledger_logger = (
        metadata.get("ledger_logger") if isinstance(metadata, Mapping) else None
    )
    if callable(ledger_logger):
        try:
            ledger_logger(ledger_payload)
        except Exception:
            logger.debug("ledger_logger failed", exc_info=True)

    ledger.record(ledger_payload)

    observation_metadata = {
        "status": "success",
        "model.id": model_id,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "input.masked_prompt": masked_prompt_preview,
    }
    if usage is not None:
        observation_metadata["usage.prompt_tokens"] = prompt_tokens
        observation_metadata["usage.completion_tokens"] = completion_tokens
        observation_metadata["usage.total_tokens"] = total_tokens
    if cost_usd is not None:
        observation_metadata["cost.usd"] = cost_usd

    _safe_update_observation(metadata=observation_metadata)

    breaker.record_success()
    yield {
        "event": "final",
        "usage": usage,
        "cost": cost_data,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "model": model_id,
        "finish_reason": finish_reason,
    }
