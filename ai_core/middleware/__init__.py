"""Middleware for AI Core."""

__all__ = ["PIISessionScopeMiddleware", "RequestContextMiddleware"]


def __getattr__(name: str):  # pragma: no cover - simple import proxy
    if name == "RequestContextMiddleware":
        from .context import RequestContextMiddleware

        return RequestContextMiddleware
    if name == "PIISessionScopeMiddleware":
        from .pii_session import PIISessionScopeMiddleware

        return PIISessionScopeMiddleware
    raise AttributeError(name)
