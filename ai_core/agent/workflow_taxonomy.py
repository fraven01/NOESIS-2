from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ALLOWED_WORKFLOW_IDS = {"INITIAL", "ONGOING", "END"}


@dataclass(frozen=True)
class WorkflowSuppression:
    owner: str
    removal_by: str


WORKFLOW_ID_ALLOWLIST: Mapping[str, Mapping[str, WorkflowSuppression]] = {
    "ai_core.tests.test_workflow_taxonomy_enforcement": {
        "MIGRATION": WorkflowSuppression(owner="test", removal_by="2026-12-31"),
    }
}


def is_workflow_allowed(workflow_id: str, *, caller_module: str | None) -> bool:
    if workflow_id in ALLOWED_WORKFLOW_IDS:
        return True
    if caller_module is None:
        return False
    module_allowlist = WORKFLOW_ID_ALLOWLIST.get(caller_module, {})
    return workflow_id in module_allowlist


__all__ = [
    "ALLOWED_WORKFLOW_IDS",
    "WorkflowSuppression",
    "WORKFLOW_ID_ALLOWLIST",
    "is_workflow_allowed",
]
