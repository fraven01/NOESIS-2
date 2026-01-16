from __future__ import annotations

from collections.abc import Awaitable, Callable

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from channels.sessions import CookieMiddleware, SessionMiddleware
from django.conf import settings
from django.contrib.auth import (
    BACKEND_SESSION_KEY,
    HASH_SESSION_KEY,
    SESSION_KEY,
    get_user_model,
    load_backend,
)
from django.db import connection
from django.utils.crypto import constant_time_compare
from django_tenants.utils import get_tenant_domain_model, remove_www, schema_context
from urllib.parse import parse_qs

from channels.auth import UserLazyObject


class TenantWebsocketMiddleware:
    """Resolve tenant schema for WebSocket connections based on host name."""

    def __init__(self, inner: Callable[..., Awaitable[None]]):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "websocket":
            return await self.inner(scope, receive, send)

        tenant_hint = _tenant_schema_from_scope(scope)
        hostname = _hostname_from_scope(scope)
        await database_sync_to_async(connection.set_schema_to_public)()

        tenant = None
        if tenant_hint:
            tenant = await database_sync_to_async(_resolve_tenant_by_identifier)(
                tenant_hint
            )

        if tenant is None and hostname:
            domain_model = get_tenant_domain_model()
            try:
                tenant = await database_sync_to_async(_resolve_tenant)(
                    domain_model, hostname
                )
            except domain_model.DoesNotExist:
                if not getattr(settings, "SHOW_PUBLIC_IF_NO_TENANT_FOUND", False):
                    await send({"type": "websocket.close", "code": 1008})
                    return None

        if tenant is not None:
            await database_sync_to_async(connection.set_tenant)(tenant)
            scope["tenant"] = tenant
            scope["tenant_schema"] = tenant.schema_name
        elif tenant_hint:
            scope["tenant_schema"] = tenant_hint

        return await self.inner(scope, receive, send)


@database_sync_to_async
def _get_user_with_schema(scope):
    tenant_schema = scope.get("tenant_schema")
    if tenant_schema:
        with schema_context(tenant_schema):
            return _resolve_user(scope)
    return _resolve_user(scope)


def _resolve_user(scope):
    from django.contrib.auth.models import AnonymousUser

    if "session" not in scope:
        raise ValueError(
            "Cannot find session in scope. You should wrap your consumer in "
            "SessionMiddleware."
        )
    session = scope["session"]
    user = None
    try:
        user_id = _get_user_session_key(session)
        backend_path = session[BACKEND_SESSION_KEY]
    except KeyError:
        pass
    else:
        if backend_path in settings.AUTHENTICATION_BACKENDS:
            backend = load_backend(backend_path)
            user = backend.get_user(user_id)
            if hasattr(user, "get_session_auth_hash"):
                session_hash = session.get(HASH_SESSION_KEY)
                session_hash_verified = session_hash and constant_time_compare(
                    session_hash, user.get_session_auth_hash()
                )
                if not session_hash_verified:
                    session.flush()
                    user = None
    return user or AnonymousUser()


def _get_user_session_key(session):
    return get_user_model()._meta.pk.to_python(session[SESSION_KEY])


class TenantAuthMiddleware(BaseMiddleware):
    """Auth middleware that resolves users inside the tenant schema."""

    def populate_scope(self, scope):
        if "session" not in scope:
            raise ValueError(
                "AuthMiddleware cannot find session in scope. "
                "SessionMiddleware must be above it."
            )
        if "user" not in scope:
            scope["user"] = UserLazyObject()

    async def resolve_scope(self, scope):
        scope["user"]._wrapped = await _get_user_with_schema(scope)

    async def __call__(self, scope, receive, send):
        scope = dict(scope)
        self.populate_scope(scope)
        await self.resolve_scope(scope)
        return await super().__call__(scope, receive, send)


def TenantAuthMiddlewareStack(inner):
    return CookieMiddleware(SessionMiddleware(TenantAuthMiddleware(inner)))


def _hostname_from_scope(scope) -> str:
    headers = dict(scope.get("headers", []))
    raw_host = headers.get(b"host", b"").decode("latin1")
    if not raw_host:
        return ""

    host = raw_host
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            host = host[1:end]
        else:
            host = host.lstrip("[").split("]")[0]
    else:
        host = host.split(":")[0]

    return remove_www(host)


def _tenant_schema_from_scope(scope) -> str:
    raw = scope.get("query_string", b"")
    if not raw:
        return ""
    try:
        parsed = parse_qs(raw.decode("latin1"))
    except Exception:
        return ""
    for key in ("tenant_schema", "tenant_id", "tenant"):
        values = parsed.get(key)
        if values:
            value = values[0].strip()
            if value:
                return value
    return ""


def _resolve_tenant(domain_model, hostname):
    return domain_model.objects.select_related("tenant").get(domain=hostname).tenant


def _resolve_tenant_by_identifier(identifier: str):
    from customers.tenant_context import TenantContext

    return TenantContext.resolve_identifier(identifier, allow_pk=True)
