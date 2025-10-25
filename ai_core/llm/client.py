from __future__ import annotations

import datetime
import math
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Mapping, Sequence
import os

import requests

from ai_core.infra.config import get_config
from ai_core.infra.observability import observe_span, update_observation
from ai_core.infra import ledger
from ai_core.llm.pricing import calculate_chat_completion_cost
from common.logging import get_logger, mask_value
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_RETRY_ATTEMPT_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)
from .routing import resolve

logger = get_logger(__name__)


# Conservative defaults to avoid web worker timeouts in dev/prod.
# Projects can override via LITELLM_TIMEOUTS env (see ai_core.infra.config).
# "synthesize" often produces longer responses; align default with tests/expectations.
DEFAULT_LABEL_TIMEOUTS: dict[str, int] = {"synthesize": 45}


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


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
        cache_hit = _detect_cache_hit(
            resp, data if isinstance(data, Mapping) else {}
        )
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


def _detect_cache_hit(resp: requests.Response | None, data: Mapping[str, Any]) -> bool | None:
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


@observe_span(name="llm.call")
def call(label: str, prompt: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM via LiteLLM proxy using a routing ``label``.

    Parameters
    ----------
    label:
        Routing label that resolves to a model id.
    prompt:
        Prompt text which will be PII-masked before sending.
    metadata:
        Dict containing at least ``tenant_id``/``tenant``, ``case_id``/``case`` and ``trace_id``.
    """

    model_id = resolve(label)
    cfg = get_config()
    url = f"{cfg.litellm_base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.litellm_api_key}"}
    # Accept both new (tenant_id/case_id) and legacy (tenant/case) keys
    tenant_value = metadata.get("tenant_id") or metadata.get("tenant")
    case_value = metadata.get("case_id") or metadata.get("case")
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
    default_max_tokens: int | None = None
    try:
        max_tokens_env = os.getenv("LITELLM_MAX_TOKENS")
        if max_tokens_env is not None:
            max_tokens_val = int(max_tokens_env)
            if max_tokens_val > 0:
                payload["max_tokens"] = max_tokens_val
                default_max_tokens = max_tokens_val
    except Exception:
        pass
    try:
        temperature_env = os.getenv("LITELLM_TEMPERATURE")
        if temperature_env is not None:
            payload["temperature"] = float(temperature_env)
    except Exception:
        pass

    # Allow env override to fail fast during dev
    max_retries = 3
    try:
        env_retries = os.getenv("LITELLM_MAX_RETRIES")
        if env_retries is not None:
            env_retries_i = int(env_retries)
            if env_retries_i >= 0:
                max_retries = env_retries_i
    except Exception:
        pass
    prompt_version = metadata.get("prompt_version") or "default"
    case_id = case_value or "unknown-case"
    idempotency_key = f"{case_id}:{label}:{prompt_version}"
    default_timeout = DEFAULT_LABEL_TIMEOUTS.get(label, 20)
    timeout = cfg.timeouts.get(label, default_timeout)
    log_extra = {
        "trace_id": mask_value(metadata.get("trace_id")),
        "case_id": mask_value(case_value),
        "tenant": mask_value(tenant_value),
        "key_alias": mask_value(metadata.get("key_alias")),
    }
    # Attach lightweight context to the current observation (no PII payloads)
    _safe_update_observation(
        tags=["llm", "litellm", f"label:{label}", f"model:{model_id}"],
        user_id=str(tenant_value) if tenant_value else None,
        session_id=str(case_value) if case_value else None,
        metadata={
            "trace_id": metadata.get("trace_id"),
            "prompt_version": prompt_version,
        },
    )

    status: int | None = None
    extended_for_length = False
    text: str | None = None
    data: Dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    cache_hit: bool | None = None
    masked_prompt_preview = _truncate(prompt, 512)
    for attempt in range(max_retries):
        resp: requests.Response | None = None
        attempt_headers = headers.copy()
        attempt_headers[IDEMPOTENCY_KEY_HEADER] = idempotency_key
        if attempt > 0:
            attempt_headers[X_RETRY_ATTEMPT_HEADER] = str(attempt + 1)
        try:
            start_ts = time.perf_counter()
            resp = requests.post(
                url, headers=attempt_headers, json=payload, timeout=timeout
            )
            latency_ms = (time.perf_counter() - start_ts) * 1000.0
        except requests.RequestException as exc:
            latency_ms = (time.perf_counter() - start_ts) * 1000.0
            status = None
            err = exc
        else:
            status = resp.status_code
            err = None

        if status and 500 <= status < 600:
            logger.warning("llm 5xx response", extra={**log_extra, "status": status})
            if attempt == max_retries - 1:
                logger.warning(
                    "llm retries exhausted", extra={**log_extra, "status": status}
                )
                payload = _safe_json(resp)
                cache_hit = _detect_cache_hit(
                    resp, payload if isinstance(payload, Mapping) else {}
                )
                detail = payload.get("detail") or _safe_text(resp)
                status_val = payload.get("status") or status
                code = payload.get("code")
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
                raise LlmClientError(
                    detail or "LLM client error", status=status_val, code=code
                ) from None
            time.sleep(min(5, 2**attempt))
            continue

        if err is not None:
            logger.warning(
                "llm request error",
                exc_info=err,
                extra={**log_extra, "status": status},
            )
            if attempt == max_retries - 1:
                logger.warning(
                    "llm retries exhausted", extra={**log_extra, "status": status}
                )
                _safe_update_observation(
                    metadata={
                        "status": "error",
                        "model.id": model_id,
                        "latency_ms": latency_ms,
                        "cache_hit": cache_hit,
                        "input.masked_prompt": masked_prompt_preview,
                        "error.type": type(err).__name__,
                        "error.message": _truncate(str(err) or "LLM client error", 256),
                        "provider.http_status": status,
                    }
                )
                raise LlmClientError(str(err) or "LLM client error") from err
            time.sleep(min(5, 2**attempt))
            continue

        if status == 429:
            logger.warning("llm rate limited", extra={**log_extra, "status": status})
            if attempt == max_retries - 1:
                logger.warning(
                    "llm retries exhausted", extra={**log_extra, "status": status}
                )
                payload = _safe_json(resp)
                cache_hit = _detect_cache_hit(
                    resp, payload if isinstance(payload, Mapping) else {}
                )
                detail = payload.get("detail") or _safe_text(resp)
                status_val = payload.get("status") or status
                code = payload.get("code")
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

            retry_after_raw = resp.headers.get("Retry-After") if resp else None
            sleep_for: float | None = None
            if retry_after_raw:
                retry_after_raw = retry_after_raw.strip()
                try:
                    sleep_for = float(retry_after_raw)
                except ValueError:
                    try:
                        retry_after_dt = parsedate_to_datetime(retry_after_raw)
                    except (TypeError, ValueError):
                        sleep_for = None
                    else:
                        if retry_after_dt.tzinfo is None:
                            retry_after_dt = retry_after_dt.replace(
                                tzinfo=datetime.timezone.utc
                            )
                        target_ts = retry_after_dt.timestamp()
                        delta = target_ts - time.time()
                        sleep_for = max(0.0, float(math.ceil(delta)))
            if sleep_for is None:
                base_delay = min(5, 2**attempt)
                sleep_for = base_delay + random.uniform(0, 0.3)
            time.sleep(max(0.0, sleep_for))
            continue

        if status and 400 <= status < 500:
            logger.warning("llm 4xx response", extra={**log_extra, "status": status})
            payload = _safe_json(resp)
            cache_hit = _detect_cache_hit(
                resp, payload if isinstance(payload, Mapping) else {}
            )
            detail = payload.get("detail") or _safe_text(resp)
            status_val = payload.get("status") or status
            code = payload.get("code")
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
            raise LlmClientError(
                detail or "LLM client error", status=status_val, code=code
            )

        data = resp.json()
        cache_hit = _detect_cache_hit(
            resp, data if isinstance(data, Mapping) else {}
        )
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], Mapping):
            raw_finish = choices[0].get("finish_reason")
            finish_reason = str(raw_finish) if raw_finish is not None else None
        try:
            text = _coerce_choice_text(choices[0])
        except (KeyError, IndexError, TypeError):
            if (
                finish_reason == "length"
                and not extended_for_length
                and (default_max_tokens or payload.get("max_tokens") or 0) < 4096
            ):
                current_max = payload.get("max_tokens")
                if not current_max:
                    current_max = default_max_tokens or 1024
                new_max = int(min(max(current_max * 2, current_max + 512), 4096))
                payload["max_tokens"] = new_max
                extended_for_length = True
                logger.info(
                    "llm.extend_max_tokens",
                    extra={
                        **log_extra,
                        "previous_max_tokens": current_max,
                        "new_max_tokens": new_max,
                    },
                )
                continue
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
            raise LlmClientError(message, status=status) from None
        break
    else:
        _safe_update_observation(
            metadata={
                "status": "error",
                "model.id": model_id,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit,
                "input.masked_prompt": masked_prompt_preview,
                "error.type": "LlmClientError",
                "error.message": "LLM client error",
                "provider.http_status": status,
            }
        )
        raise LlmClientError("LLM client error", status=status)

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
        raise LlmClientError(message, status=status) from None
    usage_raw = data.get("usage", {})
    prompt_tokens = usage_raw.get("prompt_tokens", 0) or 0
    completion_tokens = usage_raw.get("completion_tokens", 0) or 0
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    cost_usd = calculate_chat_completion_cost(
        model_id, prompt_tokens, completion_tokens
    )
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": {"usd": cost_usd},
    }
    result = {
        "text": text,
        "usage": usage,
        "model": model_id,
        "prompt_version": metadata.get("prompt_version"),
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
    }

    ledger.record(
        {
            "tenant": tenant_value,
            "case": case_value,
            "trace_id": metadata.get("trace_id"),
            "label": label,
            "model": model_id,
            "usage": usage,
            "ts": time.time(),
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
        }
    )

    _safe_update_observation(
        metadata={
            "status": "success",
            "model.id": model_id,
            "usage.prompt_tokens": prompt_tokens,
            "usage.completion_tokens": completion_tokens,
            "usage.total_tokens": total_tokens,
            "cost.usd": cost_usd,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "input.masked_prompt": masked_prompt_preview,
        }
    )

    return result
