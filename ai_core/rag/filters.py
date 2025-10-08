from typing import Any, Mapping, Optional


def strict_match(
    meta: Mapping[str, Any], tenant: Optional[str], case: Optional[str]
) -> bool:
    """Return True if filters match; `None` acts as a wildcard.

    - When `tenant` is None, do not filter by tenant.
    - When `case` is None, do not filter by case.
    """
    if tenant is not None and meta.get("tenant_id") != tenant:
        return False
    if case is not None and meta.get("case_id") != case:
        return False
    return True
