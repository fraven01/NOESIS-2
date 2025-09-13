"""Filtering helpers for tenant and case isolation."""


def strict_filters(meta: dict, tenant: str, case: str) -> bool:
    """Return ``True`` when the metadata matches the tenant and case."""

    return meta.get("tenant") == tenant and meta.get("case") == case


__all__ = ["strict_filters"]
