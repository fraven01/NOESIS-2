from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROUTING_FILE = Path(__file__).resolve().parents[2] / "MODEL_ROUTING.yaml"


@lru_cache(maxsize=1)
def load_map() -> dict[str, str]:
    """Load and cache the label→model mapping from ``MODEL_ROUTING.yaml``."""

    with ROUTING_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


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
