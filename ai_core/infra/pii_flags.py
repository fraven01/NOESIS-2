"""PII configuration access helpers."""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

from django.conf import settings
from django.core.exceptions import ValidationError

_MASKING_DISABLED_STATE: Mapping[str, object] = MappingProxyType(
    {
        "mode": "off",
        "policy": "off",
        "deterministic": False,
        "post_response": False,
        "logging_redaction": False,
        "hmac_secret": None,
        "name_detection": False,
        "session_scope": None,
    }
)

_PII_CONFIG: ContextVar[Mapping[str, object]] = ContextVar(
    "ai_core_pii_config",
    default=_MASKING_DISABLED_STATE,
)
_PII_CONFIG_VERSION: ContextVar[int] = ContextVar(
    "ai_core_pii_config_version",
    default=0,
)


def _normalize_secret(secret: Any) -> bytes | None:
    if secret in (None, ""):
        return None
    if isinstance(secret, (bytes, bytearray)):
        value = bytes(secret)
        return value or None
    text = str(secret)
    if not text:
        return None
    return text.encode("utf-8")


def _finalize_config(
    *,
    mode: str,
    policy: str,
    deterministic: bool,
    post_response: bool,
    logging_redaction: bool,
    hmac_secret: Any,
    name_detection: bool,
) -> dict[str, object]:
    secret_bytes = _normalize_secret(hmac_secret)
    deterministic_enabled = bool(deterministic) and bool(secret_bytes)
    mode_text = str(mode)
    config = {
        "mode": mode_text,
        "policy": str(policy),
        "deterministic": deterministic_enabled,
        "post_response": bool(post_response),
        "logging_redaction": bool(logging_redaction),
        "hmac_secret": secret_bytes if deterministic_enabled else None,
        "name_detection": bool(name_detection) and mode_text == "gold",
        "session_scope": None,
    }
    return config


def _coalesce_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        if value.strip() == "":
            return default
        return value
    text = str(value)
    return text if text else default


def _coalesce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    return bool(value)


def _coalesce_secret(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str) and value == "":
        return default
    return value


def _config_from_settings() -> dict[str, object]:
    return _finalize_config(
        mode=settings.PII_MODE,
        policy=settings.PII_POLICY,
        deterministic=settings.PII_DETERMINISTIC,
        post_response=settings.PII_POST_RESPONSE,
        logging_redaction=settings.PII_LOGGING_REDACTION,
        hmac_secret=settings.PII_HMAC_SECRET,
        name_detection=settings.PII_NAME_DETECTION,
    )


def get_pii_config() -> dict[str, object]:
    """Return the active PII configuration."""

    config = _PII_CONFIG.get()
    if config is _MASKING_DISABLED_STATE:
        return _config_from_settings()
    return dict(config)


def get_pii_config_version() -> int:
    """Return the current ContextVar configuration version."""

    return _PII_CONFIG_VERSION.get()


def set_pii_config(config: Mapping[str, object]) -> None:
    """Activate a scoped PII configuration for the current context.

    The provided mapping may be partial. We merge it with the defaults from
    Django settings and apply the same finalization rules as for settings-based
    configs (e.g., only enable ``deterministic`` when a non-empty secret is
    present). ``session_scope`` is passed through as-is.
    """

    # Start from settings-backed defaults
    base = _config_from_settings()

    # Apply overrides from the provided mapping (partial allowed)
    mode = _coalesce_str(config.get("mode", base["mode"]), base["mode"])  # type: ignore[arg-type]
    policy = _coalesce_str(config.get("policy", base["policy"]), base["policy"])  # type: ignore[arg-type]
    deterministic = _coalesce_bool(
        config.get("deterministic", base["deterministic"]),  # type: ignore[arg-type]
        bool(base["deterministic"]),
    )
    post_response = _coalesce_bool(
        config.get("post_response", base["post_response"]),  # type: ignore[arg-type]
        bool(base["post_response"]),
    )
    logging_redaction = _coalesce_bool(
        config.get("logging_redaction", base["logging_redaction"]),  # type: ignore[arg-type]
        bool(base["logging_redaction"]),
    )
    name_detection = _coalesce_bool(
        config.get("name_detection", base.get("name_detection", False)),  # type: ignore[arg-type]
        bool(base.get("name_detection", False)),
    )
    # Prefer explicit override for secret even if empty string provided
    hmac_secret = _coalesce_secret(
        config.get("hmac_secret", base.get("hmac_secret")),  # type: ignore[arg-type]
        base.get("hmac_secret"),
    )

    finalized = _finalize_config(
        mode=mode,
        policy=policy,
        deterministic=deterministic,
        post_response=post_response,
        logging_redaction=logging_redaction,
        hmac_secret=hmac_secret,
        name_detection=name_detection,
    )

    # Preserve optional session scope if provided
    if "session_scope" in config:
        finalized["session_scope"] = config.get("session_scope")

    _PII_CONFIG.set(MappingProxyType(finalized))
    current = _PII_CONFIG_VERSION.get()
    _PII_CONFIG_VERSION.set(current + 1)


def clear_pii_config() -> None:
    """Reset the scoped PII configuration."""

    _PII_CONFIG.set(_MASKING_DISABLED_STATE)
    current = _PII_CONFIG_VERSION.get()
    _PII_CONFIG_VERSION.set(current + 1)


def resolve_tenant_pii_config(tenant: Any) -> dict[str, object] | None:
    """Derive the PII configuration for a tenant instance."""

    if tenant is None:
        return None

    mode = _coalesce_str(getattr(tenant, "pii_mode", None), settings.PII_MODE)
    policy = _coalesce_str(getattr(tenant, "pii_policy", None), settings.PII_POLICY)
    deterministic = _coalesce_bool(
        getattr(tenant, "pii_deterministic", None), settings.PII_DETERMINISTIC
    )
    post_response = _coalesce_bool(
        getattr(tenant, "pii_post_response", None), settings.PII_POST_RESPONSE
    )
    logging_redaction = _coalesce_bool(
        getattr(tenant, "pii_logging_redaction", None), settings.PII_LOGGING_REDACTION
    )
    name_detection = _coalesce_bool(
        getattr(tenant, "pii_name_detection", None), settings.PII_NAME_DETECTION
    )
    hmac_secret = _coalesce_secret(
        getattr(tenant, "pii_hmac_secret", None), settings.PII_HMAC_SECRET
    )

    return _finalize_config(
        mode=mode,
        policy=policy,
        deterministic=deterministic,
        post_response=post_response,
        logging_redaction=logging_redaction,
        hmac_secret=hmac_secret,
        name_detection=name_detection,
    )


@lru_cache(maxsize=256)
def _load_tenant_pii_config_cached(cache_key: str) -> dict[str, object] | None:
    """Cached helper to resolve tenant configuration by identifier.

    The preferred lookup key is the tenant primary key.  For backwards compatibility
    we also accept the schema name which is unique per tenant.
    """

    from customers.models import Tenant

    tenant = None
    try:
        tenant_pk = Tenant._meta.pk.to_python(cache_key)
    except (TypeError, ValueError, ValidationError):
        tenant_pk = None

    if tenant_pk is not None:
        try:
            tenant = Tenant.objects.get(pk=tenant_pk)
        except Tenant.DoesNotExist:
            tenant = None

    if tenant is None and isinstance(cache_key, str):
        try:
            tenant = Tenant.objects.get(schema_name=cache_key)
        except Tenant.DoesNotExist:
            return None

    if tenant is None:
        return None

    config = resolve_tenant_pii_config(tenant)
    return MappingProxyType(config) if config is not None else None


def load_tenant_pii_config(tenant_id: Any) -> dict[str, object] | None:
    """Return the PII configuration for a tenant by identifier.

    `tenant_id` **must** be the tenant's primary key.  Passing the schema name is
    still supported for legacy callers, but new code should always rely on the
    primary key to avoid ambiguity.
    """

    if not tenant_id:
        return None

    try:
        cached = _load_tenant_pii_config_cached(str(tenant_id))
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        message = str(exc)
        if "Database access not allowed" in message:
            return None
        raise

    if cached is None:
        return None
    return dict(cached)


def clear_tenant_pii_config_cache() -> None:
    """Reset the cached tenant configuration mapping."""

    _load_tenant_pii_config_cached.cache_clear()


def masking_disabled_config() -> dict[str, object]:
    """Expose the masking disabled configuration for tests."""

    return dict(_MASKING_DISABLED_STATE)
