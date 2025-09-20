from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
ROUTING_FILE = ROOT_DIR / "MODEL_ROUTING.yaml"
LOCAL_OVERRIDE_FILE = ROOT_DIR / "MODEL_ROUTING.local.yaml"


@lru_cache(maxsize=1)
def load_map() -> dict[str, str]:
    """Load and cache the label→model mapping from ``MODEL_ROUTING.yaml``."""

    # Prefer local override when present (e.g., docker dev without Vertex)
    file = LOCAL_OVERRIDE_FILE if LOCAL_OVERRIDE_FILE.exists() else ROUTING_FILE

    with file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Support either flat mapping or nested under 'labels'
    if isinstance(data, dict) and "labels" in data and isinstance(data["labels"], dict):
        return data["labels"]
    return data if isinstance(data, dict) else {}


def resolve(label: str) -> str:
    """Return the model id for ``label``.

    Raises
    ------
    ValueError
        If the label does not exist in the routing map.
    """

    mapping = load_map()
    try:
        return mapping[label]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown label: {label}") from exc
