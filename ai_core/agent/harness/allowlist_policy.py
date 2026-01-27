from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml


def check_allowlist_removal_by(path: Path, today: date) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError("allowlist must be a list")

    overdue: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        removal_by = entry.get("removal_by")
        entry_path = entry.get("path", "<unknown>")
        if not isinstance(removal_by, str) or not removal_by:
            continue
        try:
            removal_date = date.fromisoformat(removal_by)
        except ValueError:
            overdue.append(f"{entry_path}::invalid_removal_by")
            continue
        if removal_date < today:
            overdue.append(f"{entry_path}::{removal_by}")

    return sorted(overdue)


__all__ = ["check_allowlist_removal_by"]
