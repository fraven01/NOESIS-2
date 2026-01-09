import pytest
from celery import current_app
from django.conf import settings
from django.test import override_settings
from django_tenants.utils import get_public_schema_name, schema_context

from ai_core.infra.pii_flags import (
    clear_pii_config,
    clear_tenant_pii_config_cache,
    get_pii_config,
    get_pii_config_version,
    load_tenant_pii_config,
    set_pii_config,
)
from ai_core.infra.policy import clear_session_scope, set_session_scope
from ai_core.infra.pii import mask_text
from common.celery import ScopedTask


@pytest.fixture(autouse=True)
def reset_pii_context():
    clear_pii_config()
    clear_tenant_pii_config_cache()
    yield
    clear_pii_config()
    clear_tenant_pii_config_cache()


def test_get_pii_config_defaults():
    """Smoke test that the default PII configuration is wired correctly."""

    with override_settings(
        PII_MODE="industrial",
        PII_POLICY="balanced",
        PII_DETERMINISTIC=True,
        PII_POST_RESPONSE=False,
        PII_LOGGING_REDACTION=True,
        PII_HMAC_SECRET="",
        PII_NAME_DETECTION=False,
    ):
        clear_pii_config()

        assert settings.PII_MODE == "industrial"
        assert settings.PII_POLICY == "balanced"
        assert settings.PII_DETERMINISTIC is True
        assert settings.PII_POST_RESPONSE is False
        assert settings.PII_LOGGING_REDACTION is True
        assert settings.PII_HMAC_SECRET == ""
        assert settings.PII_NAME_DETECTION is False

        config = get_pii_config()

        assert config == {
            "mode": "industrial",
            "policy": "balanced",
            "deterministic": False,
            "post_response": False,
            "logging_redaction": True,
            "hmac_secret": None,
            "name_detection": False,
            "session_scope": None,
        }


def test_pii_config_version_increments_on_set_and_clear():
    base_version = get_pii_config_version()

    set_pii_config(
        {
            "mode": "gold",
            "policy": "strict",
            "deterministic": False,
            "post_response": False,
            "logging_redaction": True,
            "hmac_secret": None,
            "name_detection": False,
            "session_scope": None,
        }
    )

    after_set = get_pii_config_version()
    assert after_set == base_version + 1

    clear_pii_config()

    after_clear = get_pii_config_version()
    assert after_clear == after_set + 1


@pytest.mark.gold
def test_get_pii_config_gold_mode(pii_config_env):
    overrides = pii_config_env(
        PII_MODE="gold",
        PII_POLICY="strict",
        PII_DETERMINISTIC=True,
        PII_POST_RESPONSE=True,
        PII_LOGGING_REDACTION=False,
        PII_HMAC_SECRET="secret",
        PII_NAME_DETECTION=True,
    )

    assert overrides == {
        "PII_MODE": "gold",
        "PII_POLICY": "strict",
        "PII_DETERMINISTIC": True,
        "PII_POST_RESPONSE": True,
        "PII_LOGGING_REDACTION": False,
        "PII_HMAC_SECRET": "secret",
        "PII_NAME_DETECTION": True,
    }

    config = get_pii_config()

    assert config == {
        "mode": "gold",
        "policy": "strict",
        "deterministic": True,
        "post_response": True,
        "logging_redaction": False,
        "hmac_secret": b"secret",
        "name_detection": True,
        "session_scope": None,
    }


def test_mask_text_deterministic_under_scope(pii_config_env):
    # Ensure masking is enabled regardless of ambient defaults
    pii_config_env(
        PII_MODE="industrial",
        PII_POLICY="balanced",
        PII_HMAC_SECRET="scope-secret",
        PII_DETERMINISTIC=True,
    )

    config = get_pii_config()
    assert config["deterministic"] is True
    assert config["hmac_secret"] == b"scope-secret"

    set_pii_config(config)
    try:
        scoped_config = get_pii_config()
        set_session_scope(
            tenant_id="tenant-a",
            case_id="case-42",
            session_salt="trace-99",
        )
        try:
            first = mask_text(
                "Contact user@example.com",
                scoped_config["policy"],
                scoped_config["deterministic"],
                scoped_config["hmac_secret"],
                mode=scoped_config["mode"],
            )
            second = mask_text(
                "Contact user@example.com",
                scoped_config["policy"],
                scoped_config["deterministic"],
                scoped_config["hmac_secret"],
                mode=scoped_config["mode"],
            )
        finally:
            clear_session_scope()
    finally:
        clear_pii_config()

    assert first == second
    assert "<EMAIL_" in first


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_scoped_task_uses_tenant_profiles(tenant_pool):
    def _update_tenant(tenant, **fields):
        with schema_context(get_public_schema_name()):
            for key, value in fields.items():
                setattr(tenant, key, value)
            tenant.save(update_fields=list(fields.keys()))
            tenant.refresh_from_db()
        return tenant

    tenant_disabled = _update_tenant(
        tenant_pool["alpha"],
        pii_mode="off",
        pii_policy="off",
        pii_deterministic=False,
        pii_hmac_secret="",
    )
    tenant_enabled = _update_tenant(
        tenant_pool["beta"],
        pii_mode="gold",
        pii_policy="strict",
        pii_deterministic=True,
        pii_hmac_secret="tenant-secret",
    )
    captured: list[dict[str, object]] = []

    class _TenantTask(ScopedTask):
        abstract = False
        name = "tests.capture_tenant_pii"

        def run(self, label: str) -> str:  # type: ignore[override]
            config = get_pii_config()
            masked = mask_text(
                "Reach user@example.com",
                config["policy"],
                config["deterministic"],
                config["hmac_secret"],
                mode=config["mode"],
            )
            captured.append({"label": label, "masked": masked, "config": config})
            return masked

    task = _TenantTask()
    task.bind(current_app)

    disabled_result = task.__call__(
        "disabled",
        tenant_id=tenant_disabled.id,
        case_id="case-a",
        trace_id="trace-a",
    )
    enabled_result = task.__call__(
        "enabled",
        tenant_id=tenant_enabled.id,
        case_id="case-b",
        trace_id="trace-b",
    )

    assert disabled_result == "Reach user@example.com"
    assert "user@example.com" in disabled_result
    assert "<EMAIL_" not in disabled_result

    assert "user@example.com" not in enabled_result
    assert "<EMAIL_" in enabled_result

    assert [entry["label"] for entry in captured] == ["disabled", "enabled"]
    disabled_config = captured[0]["config"]
    enabled_config = captured[1]["config"]

    assert disabled_config["mode"] == "off"
    assert enabled_config["mode"] == "gold"
    assert enabled_config["deterministic"] is True
    assert enabled_config["hmac_secret"]
    assert disabled_config["session_scope"]
    assert enabled_config["session_scope"]


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_load_tenant_pii_config_is_cached(tenant_pool, django_assert_num_queries):
    with schema_context(get_public_schema_name()):
        tenant = tenant_pool["gamma"]
        tenant.pii_mode = "gold"
        tenant.pii_policy = "strict"
        tenant.pii_deterministic = True
        tenant.pii_hmac_secret = "cache-secret"
        tenant.save(
            update_fields=[
                "pii_mode",
                "pii_policy",
                "pii_deterministic",
                "pii_hmac_secret",
            ]
        )

    with django_assert_num_queries(4):
        first = load_tenant_pii_config(tenant.id)

    with django_assert_num_queries(0):
        second = load_tenant_pii_config(tenant.id)
        third = load_tenant_pii_config(str(tenant.id))

    assert first == second == third
    assert first["mode"] == "gold"
    assert first["deterministic"] is True


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_load_tenant_pii_config_accepts_schema_name(tenant_pool):
    with schema_context(get_public_schema_name()):
        tenant = tenant_pool["delta"]
        tenant.pii_mode = "gold"
        tenant.pii_policy = "strict"
        tenant.pii_deterministic = True
        tenant.pii_hmac_secret = "cache-secret"
        tenant.save(
            update_fields=[
                "pii_mode",
                "pii_policy",
                "pii_deterministic",
                "pii_hmac_secret",
            ]
        )

    config = load_tenant_pii_config(tenant.schema_name)
    assert config is not None
    assert config["mode"] == "gold"
    assert config["deterministic"] is True


def test_load_tenant_pii_config_gracefully_handles_blocked_db(monkeypatch):
    def _raise_runtime_error(_cache_key: str):
        raise RuntimeError(
            'Database access not allowed, use the "django_db" mark, or the "db" or '
            '"transactional_db" fixtures to enable it.'
        )

    _raise_runtime_error.cache_clear = lambda: None  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "ai_core.infra.pii_flags._load_tenant_pii_config_cached",
        _raise_runtime_error,
    )

    config = load_tenant_pii_config("tenant-1")

    assert config is None


def test_pii_middleware_order(settings):
    middleware = list(settings.MIDDLEWARE)
    assert "ai_core.middleware.PIISessionScopeMiddleware" in middleware
    assert "ai_core.middleware.RequestContextMiddleware" in middleware

    scope_index = middleware.index("ai_core.middleware.PIISessionScopeMiddleware")
    request_context_index = middleware.index(
        "ai_core.middleware.RequestContextMiddleware"
    )

    assert (
        scope_index < request_context_index
    ), "PII session scope middleware must run before request context logging middleware"
