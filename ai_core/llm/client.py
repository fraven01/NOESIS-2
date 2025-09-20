from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests

from ai_core.infra.config import get_config
from ai_core.infra.pii import mask_prompt
from ai_core.infra import ledger
from .routing import resolve

logger = logging.getLogger(__name__)


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
        "X-Trace-ID": metadata.get("trace_id"),
        "X-Case-ID": metadata.get("case"),
        "X-Tenant-ID": metadata.get("tenant"),
    }
    key_alias = metadata.get("key_alias")
    if key_alias:
        propagated_headers["X-Key-Alias"] = key_alias
    headers.update({k: v for k, v in propagated_headers.items() if v})
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_safe}],
    }

    max_retries = 3
    timeout = 20
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            status = None
            err = exc
        else:
            status = resp.status_code
            err = None

        if status and 500 <= status < 600:
            logger.warning("llm 5xx response", extra={"status": status})
            if attempt == max_retries - 1:
                raise ValueError("llm error") from None
            time.sleep(min(5, 2**attempt))
            continue

        if err is not None:
            logger.warning("llm request error", exc_info=err)
            if attempt == max_retries - 1:
                raise ValueError("llm error") from None
            time.sleep(min(5, 2**attempt))
            continue

        if status and 400 <= status < 500:
            logger.warning("llm 4xx response", extra={"status": status})
            raise ValueError("llm error")

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
