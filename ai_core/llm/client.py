from __future__ import annotations

import datetime
import math
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict
import os

import requests

from ai_core.infra.config import get_config
from ai_core.infra.observability import observe_span, update_observation
from ai_core.infra import ledger
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
DEFAULT_LABEL_TIMEOUTS: dict[str, int] = {"synthesize": 20}


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
    try:
        max_tokens_env = os.getenv("LITELLM_MAX_TOKENS")
        if max_tokens_env is not None:
            max_tokens_val = int(max_tokens_env)
            if max_tokens_val > 0:
                payload["max_tokens"] = max_tokens_val
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
    try:
        update_observation(
            tags=["llm", "litellm", f"label:{label}", f"model:{model_id}"],
            user_id=str(tenant_value) if tenant_value else None,
            session_id=str(case_value) if case_value else None,
            metadata={
                "trace_id": metadata.get("trace_id"),
                "prompt_version": prompt_version,
            },
        )
    except Exception:
        pass

    for attempt in range(max_retries):
        resp: requests.Response | None = None
        attempt_headers = headers.copy()
        attempt_headers[IDEMPOTENCY_KEY_HEADER] = idempotency_key
        if attempt > 0:
            attempt_headers[X_RETRY_ATTEMPT_HEADER] = str(attempt + 1)
        try:
            resp = requests.post(
                url, headers=attempt_headers, json=payload, timeout=timeout
            )
        except requests.RequestException as exc:
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
                detail = payload.get("detail") or _safe_text(resp)
                status_val = payload.get("status") or status
                code = payload.get("code")
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
                detail = payload.get("detail") or _safe_text(resp)
                status_val = payload.get("status") or status
                code = payload.get("code")
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
            detail = payload.get("detail") or _safe_text(resp)
            status_val = payload.get("status") or status
            code = payload.get("code")
            raise LlmClientError(
                detail or "LLM client error", status=status_val, code=code
            )

        data = resp.json()
        break

    text = data["choices"][0]["message"]["content"]
    usage_raw = data.get("usage", {})
    usage = {
        "in_tokens": usage_raw.get("prompt_tokens", 0),
        "out_tokens": usage_raw.get("completion_tokens", 0),
        "cost": 0.0,
    }
    result = {
        "text": text,
        "usage": usage,
        "model": model_id,
        "prompt_version": metadata.get("prompt_version"),
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
        }
    )

    return result
