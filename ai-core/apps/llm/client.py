"""Minimal LiteLLM client with routing and ledger recording."""

from __future__ import annotations

import os
from typing import Any, Dict

import requests

from apps.infra import ledger
from . import routing

TIMEOUT = 10
MAX_RETRIES = 3


def call(label: str, prompt: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Call a model based on label and log metadata to the ledger."""
    route = routing.route(label)
    payload = {
        "model": f"{route.get('provider')}/{route.get('model')}",
        "prompt": prompt,
    }
    base_url = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(base_url, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            result = {
                "text": data.get("text", ""),
                "usage": {
                    "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                    "output_tokens": data.get("usage", {}).get("output_tokens", 0),
                    "cost": data.get("usage", {}).get("cost", 0.0),
                    "model": payload["model"],
                },
            }
            ledger.record(metadata)
            return result
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
    # Should not reach here
    return {
        "text": "",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "model": payload["model"],
        },
    }
