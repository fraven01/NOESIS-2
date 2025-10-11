from typing import Any, Mapping


def strict_match(meta: Mapping[str, Any], tenant: str | None, case: str | None) -> bool:
    """Return ``True`` when metadata matches the requested tenant and case."""

    if tenant is None or case is None:
        return False

    tenant_in_meta = meta.get("tenant_id")
    case_in_meta = meta.get("case_id")
    if tenant_in_meta is None or case_in_meta is None:
        return False

    return tenant_in_meta == tenant and case_in_meta == case
