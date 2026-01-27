from __future__ import annotations

import inspect
from typing import Any

from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.scope import ExecutionScope
from ai_core.agent.workflow_taxonomy import is_workflow_allowed
from ai_core.tool_contracts.base import ToolContext


class PolicyViolation(RuntimeError):
    """Raised when a mutation violates execution scope policy."""


MUTATION_ACTIONS = (
    "document_upsert",
    "document_delete",
    "asset_delete",
    "document_lifecycle_record",
    "ingestion_run_queued",
    "ingestion_run_completed",
    "case_event",
    "vector_upsert",
)


def guard_mutation(
    action: str,
    tool_context: ToolContext,
    runtime_config: RuntimeConfig,
    details: dict[str, Any] | None = None,
) -> None:
    if not isinstance(tool_context, ToolContext):
        raise TypeError("tool_context must be a ToolContext instance")
    if not isinstance(runtime_config, RuntimeConfig):
        raise TypeError("runtime_config must be a RuntimeConfig instance")

    scope_value = runtime_config.execution_scope
    if scope_value is None:
        raise PolicyViolation("execution_scope is required for mutations")
    try:
        scope = ExecutionScope(scope_value)
    except ValueError as exc:
        raise PolicyViolation(f"unknown execution_scope: {scope_value}") from exc

    if scope is ExecutionScope.SYSTEM:
        raise PolicyViolation(f"mutation_not_allowed_in_system_scope:{action}")

    if scope is ExecutionScope.TENANT and action == "case_event":
        raise PolicyViolation("case_event_not_allowed_in_tenant_scope")

    if scope is ExecutionScope.CASE:
        case_id = tool_context.business.case_id
        workflow_id = tool_context.business.workflow_id
        if not case_id or not workflow_id:
            raise PolicyViolation("case_scope_requires_case_id_and_workflow_id")
        caller_module = _caller_module_name()
        if not is_workflow_allowed(str(workflow_id), caller_module=caller_module):
            raise PolicyViolation("workflow_id_not_allowed_in_case_scope")


def _caller_module_name() -> str | None:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None or frame.f_back.f_back is None:
        return None
    return frame.f_back.f_back.f_globals.get("__name__")


__all__ = ["PolicyViolation", "guard_mutation", "MUTATION_ACTIONS"]
