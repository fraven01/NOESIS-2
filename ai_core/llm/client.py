from __future__ import annotations

import datetime
import logging
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict

import requests

from ai_core.infra.config import get_config
from ai_core.infra.pii import mask_prompt
from ai_core.infra import ledger
from common.logging import mask_value
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_RETRY_ATTEMPT_HEADER,
    X_TENANT_ID_HEADER,
    X_TRACE_ID_HEADER,
)
from .routing import resolve

logger = logging.getLogger(__name__)


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


def call(label: str, prompt: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM via LiteLLM proxy using a routing ``label``.

    Parameters
    ----------
    label:
        Routing label that resolves to a model id.
    prompt:
        Prompt text which will be PII-masked before sending.
    metadata:
        Dict containing at least ``tenant``, ``case`` and ``trace_id``.
    """

    model_id = resolve(label)
    prompt_safe = mask_prompt(prompt, placeholder_only=True)

    cfg = get_config()
    url = f"{cfg.litellm_base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.litellm_api_key}"}
    propagated_headers = {
        X_TRACE_ID_HEADER: metadata.get("trace_id"),
        X_CASE_ID_HEADER: metadata.get("case"),
        X_TENANT_ID_HEADER: metadata.get("tenant"),
    }
    key_alias = metadata.get("key_alias")
    if key_alias:
        propagated_headers[X_KEY_ALIAS_HEADER] = key_alias
    headers.update({k: v for k, v in propagated_headers.items() if v})
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_safe}],
    }

    max_retries = 3
    prompt_version = metadata.get("prompt_version") or "default"
    case_id = metadata.get("case") or "unknown-case"
    idempotency_key = f"{case_id}:{label}:{prompt_version}"
    timeout = cfg.timeouts.get(label, 20)
    log_extra = {
        "trace_id": mask_value(metadata.get("trace_id")),
        "case_id": mask_value(metadata.get("case")),
        "tenant": mask_value(metadata.get("tenant")),
        "key_alias": mask_value(metadata.get("key_alias")),
    }
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
            logger.warning(
                "llm 5xx response", extra={**log_extra, "status": status}
            )
            if attempt == max_retries - 1:
                logger.warning(
                    "llm retries exhausted", extra={**log_extra, "status": status}
                )
                payload = _safe_json(resp)
                detail = payload.get("detail") or (
                    (resp.text or "").strip() if resp is not None else None
                )
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
            logger.warning(
                "llm rate limited", extra={**log_extra, "status": status}
            )
            if attempt == max_retries - 1:
                logger.warning(
                    "llm retries exhausted", extra={**log_extra, "status": status}
                )
                payload = _safe_json(resp)
                detail = payload.get("detail") or (
                    (resp.text or "").strip() if resp is not None else None
                )
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
                        sleep_for = target_ts - time.time()
            if sleep_for is None:
                base_delay = min(5, 2**attempt)
                sleep_for = base_delay + random.uniform(0, 0.3)
            time.sleep(max(0.0, sleep_for))
            continue

        if status and 400 <= status < 500:
            logger.warning(
                "llm 4xx response", extra={**log_extra, "status": status}
            )
            payload = _safe_json(resp)
            detail = payload.get("detail") or (
                (resp.text or "").strip() if resp is not None else None
            )
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
            "tenant": metadata.get("tenant"),
            "case": metadata.get("case"),
            "trace_id": metadata.get("trace_id"),
            "label": label,
            "model": model_id,
            "usage": usage,
            "ts": time.time(),
        }
    )

    return result
