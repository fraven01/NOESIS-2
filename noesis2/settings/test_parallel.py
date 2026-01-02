from __future__ import annotations

import os
import re

from .development import *  # noqa: F403


def _worker_suffix(name: str | None, worker: str | None) -> str | None:
    if not name or not worker:
        return name
    token = f"_{worker}"
    if name.endswith(token) or token in name:
        return name
    return f"{name}{token}"


def _apply_search_path(options: dict[str, object], schema: str) -> None:
    existing = str(options.get("options", "")).strip()
    if existing:
        existing = re.sub(r"-c\\s*search_path=\\S+", "", existing)
        existing = re.sub(r"-csearch_path=\\S+", "", existing)
        existing = " ".join(existing.split())
    options["options"] = f"{existing} -c search_path={schema}".strip()


_worker = os.getenv("PYTEST_XDIST_WORKER")
_db = DATABASES["default"]  # noqa: F405

# Ensure we have a TEST configuration to avoid running against production DB
if "TEST" not in _db:
    _db["TEST"] = {}

# Set base test DB name (pytest-django would add test_ prefix, but we set it explicitly)
base_test_name = _db["TEST"].get("NAME") or f"test_{_db.get('NAME', 'noesis2')}"
_db["TEST"]["NAME"] = _worker_suffix(base_test_name, _worker)

# Keep production DB name with worker suffix for settings consistency
_db["NAME"] = _worker_suffix(_db.get("NAME"), _worker)

TEST_TENANT_SCHEMA = f"autotest_{_worker}" if _worker else "autotest"

public_schema = globals().get("PUBLIC_SCHEMA_NAME", "public")
_apply_search_path(_db.setdefault("OPTIONS", {}), public_schema)
