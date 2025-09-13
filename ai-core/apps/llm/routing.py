"""LLM label to model routing via YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

ROUTING_PATH = Path(__file__).with_name("MODEL_ROUTING.yaml")

try:
    with ROUTING_PATH.open("r", encoding="utf-8") as f:
        MODEL_ROUTING: Dict[str, Dict[str, str]] = yaml.safe_load(f) or {}
except FileNotFoundError:
    MODEL_ROUTING = {}


def route(label: str) -> Dict[str, str]:
    """Return provider/model mapping for a given label."""
    return MODEL_ROUTING.get(label, {})
