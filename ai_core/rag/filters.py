from typing import Any, Mapping


def strict_match(meta: Mapping[str, Any], tenant: str, case: str) -> bool:
    """Return True if the chunk metadata matches tenant and case exactly."""
    return meta.get("tenant") == tenant and meta.get("case") == case
