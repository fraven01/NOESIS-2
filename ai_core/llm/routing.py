from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
ROUTING_FILE = ROOT_DIR / "MODEL_ROUTING.yaml"
LOCAL_OVERRIDE_FILE = ROOT_DIR / "MODEL_ROUTING.local.yaml"


def _parse_routing_file(file: Path) -> dict[str, str]:
    """Return the label→model mapping contained in ``file``."""

    if not file.exists():
        return {}

    with file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        return {}

    mapping = data.get("labels") if isinstance(data.get("labels"), dict) else data
    if not isinstance(mapping, dict):
        return {}

    return {str(label): value for label, value in mapping.items()}


@lru_cache(maxsize=1)
def load_map() -> dict[str, str]:
    """Load and cache the label→model mapping from routing files."""

    base_map = _parse_routing_file(ROUTING_FILE)
    override_map = _parse_routing_file(LOCAL_OVERRIDE_FILE)

    combined_map = base_map.copy()
    combined_map.update(override_map)
    return combined_map


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
