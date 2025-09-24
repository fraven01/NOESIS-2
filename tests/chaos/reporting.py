"""Pytest plugin that records chaos test results as JSON artifacts."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from _pytest.config import Config
from _pytest.reports import TestReport

from .fixtures import CHAOS_ENV_REGISTRY

CHAOS_LOG_ROOT = Path("logs/app/chaos")
_RUN_ID: str | None = None


def pytest_runtest_setup(item) -> None:
    """Persist chaos marker metadata on the test item before execution."""

    if "chaos" not in item.keywords:
        return
    markers = sorted({mark.name for mark in item.iter_markers()})
    item.user_properties.append(("chaos_markers", markers))


def pytest_configure(config: Config) -> None:
    """Initialise the chaos reporting directory and run metadata."""

    CHAOS_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    global _RUN_ID
    _RUN_ID = datetime.now(timezone.utc).strftime("chaos-%Y%m%dT%H%M%S%fZ")


def pytest_runtest_logreport(report: TestReport) -> None:
    """Persist a JSON artifact for each chaos test outcome."""

    if "chaos" not in report.keywords:
        return

    is_terminal_phase = report.when == "call"
    if report.outcome == "skipped" and report.when == "setup":
        is_terminal_phase = True

    if not is_terminal_phase:
        return

    metadata = _serialise_user_properties(report.user_properties)
    env_state = CHAOS_ENV_REGISTRY.get(report.nodeid)
    if env_state is not None:
        metadata.setdefault("chaos_env", env_state.values)
    markers = metadata.pop("chaos_markers", None)
    payload: Dict[str, Any] = {
        "test_suite": "chaos",
        "run_id": _RUN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "phase": report.when,
        "duration": getattr(report, "duration", None),
        "worker": os.getenv("PYTEST_XDIST_WORKER", "master"),
        "markers": markers if markers is not None else _serialise_keywords(report.keywords),
        "metadata": metadata,
    }

    if report.failed:
        payload["longrepr"] = getattr(report, "longreprtext", str(report.longrepr))

    path = CHAOS_LOG_ROOT / _build_filename(report.nodeid)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)


def _serialise_keywords(keywords: Mapping[str, Any] | Iterable[Any]) -> List[str]:
    """Return a sorted list of marker names from the report keywords."""

    if isinstance(keywords, Mapping):
        raw: Iterable[Any] = keywords.keys()
    else:
        raw = keywords
    skip = {"", "pytestmark", "tests"}
    names = {
        entry
        for entry in (str(candidate) for candidate in raw if isinstance(candidate, str))
        if entry not in skip
        and "::" not in entry
        and "/" not in entry
        and "." not in entry
        and "-" not in entry
        and entry.isidentifier()
    }
    return sorted(names)


def _serialise_user_properties(properties: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    """Convert pytest user properties to a JSON-friendly dictionary."""

    data: Dict[str, Any] = {}
    for key, value in properties:
        try:
            json.dumps(value)
        except TypeError:
            data[key] = str(value)
        else:
            data[key] = value
    return data


def _build_filename(nodeid: str) -> str:
    """Return a filesystem-safe filename for a test nodeid."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", nodeid)
    return f"{timestamp}_{slug}.json"
