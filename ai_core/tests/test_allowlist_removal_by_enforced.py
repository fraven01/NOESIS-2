from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from ai_core.agent.harness.allowlist_policy import check_allowlist_removal_by


def _write_allowlist(path: Path, removal_by: str) -> None:
    content = [
        {
            "path": "ai_core/services/rag_query.py",
            "owner": "test",
            "removal_by": removal_by,
        }
    ]
    path.write_text(yaml.safe_dump(content), encoding="utf-8")


def test_allowlist_overdue_is_detected(tmp_path):
    allowlist = tmp_path / "allowlist.yaml"
    _write_allowlist(allowlist, "2026-01-01")

    overdue = check_allowlist_removal_by(allowlist, today=date(2026, 1, 27))
    assert overdue == ["ai_core/services/rag_query.py::2026-01-01"]


def test_allowlist_future_is_not_overdue(tmp_path):
    allowlist = tmp_path / "allowlist.yaml"
    _write_allowlist(allowlist, "2026-12-31")

    overdue = check_allowlist_removal_by(allowlist, today=date(2026, 1, 27))
    assert overdue == []
