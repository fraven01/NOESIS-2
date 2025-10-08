from typing import Any, Mapping, Optional


def strict_match(
    meta: Mapping[str, Any], tenant: Optional[str], case: Optional[str]
) -> bool:
    """Return True if filters match; `None` acts as a wildcard.

    - When `tenant` is None, do not filter by tenant.
    - When `case` is None, do not filter by case.
    """
    tenant_meta = meta.get("tenant_id")
    if tenant_meta is None and "tenant_id" not in meta:
        tenant_meta = meta.get("tenant")
    if tenant is not None and tenant_meta != tenant:
        return False
    case_meta = meta.get("case_id")
    if case_meta is None and "case_id" not in meta:
        case_meta = meta.get("case")
    if case is not None and case_meta != case:
        return False
    return True
