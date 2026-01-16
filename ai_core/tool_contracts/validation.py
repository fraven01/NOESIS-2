from __future__ import annotations

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def require_business_field(
    business: BusinessContext,
    field_name: str,
    operation: str | None = None,
) -> str:
    """Require a business context field to be present."""
    from ai_core.tool_contracts import ContextError

    if not hasattr(business, field_name):
        raise ContextError(
            f"BusinessContext has no field '{field_name}'",
            field=field_name,
        )
    value = getattr(business, field_name)
    if value is None:
        op = operation or "Operation"
        raise ContextError(
            f"{op} requires business.{field_name}",
            field=field_name,
        )
    if isinstance(value, str) and not value.strip():
        op = operation or "Operation"
        raise ContextError(
            f"{op} requires business.{field_name}",
            field=field_name,
        )
    return str(value)


def require_runtime_id(scope: ScopeContext, prefer: str = "run_id") -> str:
    """Require at least one runtime ID and return the preferred one when present."""
    from ai_core.tool_contracts import ContextError

    if prefer not in {"run_id", "ingestion_run_id"}:
        raise ContextError(
            "prefer must be 'run_id' or 'ingestion_run_id'",
            field="prefer",
        )
    primary = getattr(scope, prefer)
    secondary = scope.ingestion_run_id if prefer == "run_id" else scope.run_id
    if primary:
        return str(primary)
    if secondary:
        return str(secondary)
    raise ContextError(
        "At least one of run_id or ingestion_run_id must be provided",
        field=prefer,
    )
