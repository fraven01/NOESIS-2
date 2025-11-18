from __future__ import annotations

import pytest
from celery import chain, current_app
from celery.canvas import Signature, _chain
from django.http import HttpResponse
from django.test import RequestFactory

from ai_core.infra.pii import mask_text
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.policy import get_session_scope
from ai_core.middleware import PIISessionScopeMiddleware
from common.celery import ScopedTask, with_scope_apply_async


pytestmark = pytest.mark.django_db


@pytest.fixture
def scope_headers_shared() -> dict[str, str]:
    return {
        "HTTP_X_TENANT_ID": "tenant-alpha",
        "HTTP_X_CASE_ID": "case-123",
        "HTTP_X_TRACE_ID": "trace-shared",
    }


@pytest.fixture
def scope_headers_new_trace(scope_headers_shared: dict[str, str]) -> dict[str, str]:
    headers = scope_headers_shared.copy()
    headers["HTTP_X_TRACE_ID"] = "trace-isolated"
    return headers


@pytest.fixture
def configure_autotest_tenant():
    from customers.models import Tenant
    from django_tenants.utils import get_public_schema_name, schema_context

    public_schema = get_public_schema_name()
    with schema_context(public_schema):
        tenant = Tenant.objects.get(schema_name="autotest")
        original = {
            "pii_mode": tenant.pii_mode,
            "pii_policy": tenant.pii_policy,
            "pii_logging_redaction": tenant.pii_logging_redaction,
            "pii_post_response": tenant.pii_post_response,
            "pii_deterministic": tenant.pii_deterministic,
            "pii_hmac_secret": tenant.pii_hmac_secret,
            "pii_name_detection": tenant.pii_name_detection,
        }

    def _configure(**overrides):
        with schema_context(public_schema):
            tenant.refresh_from_db()
            if not overrides:
                return tenant
            for field, value in overrides.items():
                setattr(tenant, field, value)
            tenant.save(update_fields=list(overrides.keys()))
            tenant.refresh_from_db()
            return tenant

    yield _configure

    with schema_context(public_schema):
        Tenant.objects.filter(pk=tenant.pk).update(**original)
        tenant.refresh_from_db()


@pytest.fixture
def tenant_factory():
    from customers.models import Tenant
    from django.core.management import call_command
    from django_tenants.utils import get_public_schema_name, schema_context

    public_schema = get_public_schema_name()
    created: list[Tenant] = []

    def _create(schema_name: str, **fields):
        data = {
            "schema_name": schema_name,
            "name": fields.pop("name", f"Tenant {schema_name}"),
        }
        data.update(fields)
        with schema_context(public_schema):
            tenant = Tenant.objects.create(**data)
            tenant.create_schema(check_if_exists=True)
            call_command(
                "migrate_schemas",
                tenant=True,
                schema_name=tenant.schema_name,
                interactive=False,
                verbosity=0,
            )
        created.append(tenant)
        return tenant

    yield _create

    for tenant in created:
        with schema_context(tenant.schema_name):
            tenant.delete(force_drop=True)


def test_session_scope_middleware_sets_scope_and_masks(
    configure_autotest_tenant, scope_headers_shared
):
    tenant = configure_autotest_tenant(
        pii_mode="gold",
        pii_policy="strict",
        pii_deterministic=True,
        pii_hmac_secret="secret-key",
        pii_name_detection=True,
    )
    factory = RequestFactory()
    observed_scopes: list[tuple[str, ...]] = []
    responses: list[str] = []

    def _view(request):
        config = get_pii_config()
        observed_scopes.append(get_session_scope())
        masked = mask_text(
            "Email a@b.de bitte prüfen",
            config["policy"],
            config["deterministic"],
            config["hmac_secret"],
            mode=config["mode"],
            name_detection=config["name_detection"],
        )
        responses.append(masked)
        return HttpResponse(masked)

    middleware = PIISessionScopeMiddleware(_view)

    request1 = factory.get("/echo/", **scope_headers_shared)
    request1.tenant = tenant
    response1 = middleware(request1)
    assert response1.status_code == 200
    body1 = response1.content.decode("utf-8")
    assert body1 == responses[0]
    assert "<EMAIL_" in body1
    assert "a@b.de" not in body1
    expected_scope = (
        scope_headers_shared["HTTP_X_TENANT_ID"],
        scope_headers_shared["HTTP_X_CASE_ID"],
        "||".join(
            (
                scope_headers_shared["HTTP_X_TRACE_ID"],
                scope_headers_shared["HTTP_X_CASE_ID"],
                scope_headers_shared["HTTP_X_TENANT_ID"],
            )
        ),
    )
    assert observed_scopes[0] == expected_scope
    assert get_session_scope() is None

    request2 = factory.get("/echo/", **scope_headers_shared)
    request2.tenant = tenant
    response2 = middleware(request2)
    assert response2.status_code == 200
    body2 = response2.content.decode("utf-8")
    assert body2 == responses[1]
    assert observed_scopes[1] == expected_scope
    assert body2 == body1
    assert get_session_scope() is None


def test_session_scope_middleware_clears_scope_on_exception(
    configure_autotest_tenant, scope_headers_shared
):
    tenant = configure_autotest_tenant(
        pii_mode="gold",
        pii_policy="strict",
        pii_deterministic=True,
        pii_hmac_secret="secret-key",
    )
    factory = RequestFactory()
    observed_scopes: list[tuple[str, ...]] = []

    def _view(request):
        observed_scopes.append(get_session_scope())
        raise RuntimeError("boom")

    middleware = PIISessionScopeMiddleware(_view)

    request = factory.get("/echo/", **scope_headers_shared)
    request.tenant = tenant
    with pytest.raises(RuntimeError):
        middleware(request)
    expected_scope = (
        scope_headers_shared["HTTP_X_TENANT_ID"],
        scope_headers_shared["HTTP_X_CASE_ID"],
        "||".join(
            (
                scope_headers_shared["HTTP_X_TRACE_ID"],
                scope_headers_shared["HTTP_X_CASE_ID"],
                scope_headers_shared["HTTP_X_TENANT_ID"],
            )
        ),
    )
    assert observed_scopes == [expected_scope]
    assert get_session_scope() is None


@pytest.mark.parametrize(
    "tenant_header_source",
    ["pk", "schema"],
    ids=["primary-key", "schema-name"],
)
def test_session_scope_middleware_loads_tenant_from_header_when_missing_request_tenant(
    tenant_factory,
    scope_headers_shared,
    tenant_header_source,
    monkeypatch,
):
    tenant = tenant_factory(
        "fallback-tenant",
        pii_mode="gold",
        pii_policy="strict",
        pii_logging_redaction=True,
        pii_post_response=True,
        pii_deterministic=True,
        pii_hmac_secret="header-secret",
        pii_name_detection=True,
    )
    monkeypatch.setattr(
        "ai_core.middleware.pii_session.get_current_tenant", lambda: None
    )

    factory = RequestFactory()
    headers = scope_headers_shared.copy()
    headers["HTTP_X_TENANT_ID"] = (
        str(tenant.pk) if tenant_header_source == "pk" else tenant.schema_name
    )

    observed_configs: list[dict[str, object]] = []

    def _view(request):
        observed_configs.append(get_pii_config())
        return HttpResponse("ok")

    middleware = PIISessionScopeMiddleware(_view)

    request = factory.get("/echo/", **headers)
    response = middleware(request)

    assert response.status_code == 200
    assert len(observed_configs) == 1
    config = observed_configs[0]
    assert config["mode"] == "gold"
    assert config["policy"] == "strict"
    assert config["deterministic"] is True
    assert config["hmac_secret"] == b"header-secret"
    assert config["logging_redaction"] is True
    assert config["post_response"] is True
    assert config["name_detection"] is True
    expected_scope = (
        headers["HTTP_X_TENANT_ID"],
        headers["HTTP_X_CASE_ID"],
        "||".join(
            (
                headers["HTTP_X_TRACE_ID"],
                headers["HTTP_X_CASE_ID"],
                headers["HTTP_X_TENANT_ID"],
            )
        ),
    )
    assert config["session_scope"] == expected_scope


def test_session_scope_middleware_changes_token_with_new_trace(
    configure_autotest_tenant, scope_headers_shared, scope_headers_new_trace
):
    tenant = configure_autotest_tenant(
        pii_mode="gold",
        pii_policy="strict",
        pii_deterministic=True,
        pii_hmac_secret="secret-key",
    )
    factory = RequestFactory()
    placeholders: list[str] = []

    def _view(request):
        config = get_pii_config()
        masked = mask_text(
            "Email a@b.de bitte prüfen",
            config["policy"],
            config["deterministic"],
            config["hmac_secret"],
            mode=config["mode"],
            name_detection=config["name_detection"],
        )
        placeholders.append(masked)
        return HttpResponse(masked)

    middleware = PIISessionScopeMiddleware(_view)

    request_same = factory.get("/echo/", **scope_headers_shared)
    request_same.tenant = tenant
    response_same = middleware(request_same)
    assert response_same.status_code == 200
    first_placeholder = response_same.content.decode("utf-8")
    assert placeholders[0] == first_placeholder

    request_new = factory.get("/echo/", **scope_headers_new_trace)
    request_new.tenant = tenant
    response_new = middleware(request_new)
    assert response_new.status_code == 200
    second_placeholder = response_new.content.decode("utf-8")
    assert placeholders[1] == second_placeholder

    assert first_placeholder != second_placeholder
    assert "<EMAIL_" in first_placeholder
    assert "<EMAIL_" in second_placeholder
    assert get_session_scope() is None


def test_middleware_uses_tenant_specific_profiles(
    configure_autotest_tenant, tenant_factory, scope_headers_shared
):
    tenant_disabled = configure_autotest_tenant(
        pii_mode="off",
        pii_policy="off",
        pii_deterministic=False,
        pii_hmac_secret="",
    )
    tenant_enabled = tenant_factory(
        "tenant-beta",
        pii_mode="gold",
        pii_policy="strict",
        pii_deterministic=True,
        pii_hmac_secret="beta-secret",
        pii_name_detection=True,
    )
    factory = RequestFactory()
    responses: list[str] = []

    def _view(request):
        config = get_pii_config()
        masked = mask_text(
            "Kontakt user@example.com",
            config["policy"],
            config["deterministic"],
            config["hmac_secret"],
            mode=config["mode"],
            name_detection=config["name_detection"],
        )
        responses.append(masked)
        return HttpResponse(masked)

    middleware = PIISessionScopeMiddleware(_view)

    disabled_headers = scope_headers_shared.copy()
    disabled_headers["HTTP_X_TENANT_ID"] = tenant_disabled.schema_name
    first_request = factory.get("/echo/", **disabled_headers)
    first_request.tenant = tenant_disabled
    first_response = middleware(first_request)
    assert first_response.status_code == 200
    plain_body = first_response.content.decode("utf-8")
    assert plain_body == "Kontakt user@example.com"

    enabled_headers = scope_headers_shared.copy()
    enabled_headers["HTTP_X_TENANT_ID"] = tenant_enabled.schema_name
    enabled_headers["HTTP_X_TRACE_ID"] = "trace-beta"
    second_request = factory.get("/echo/", **enabled_headers)
    second_request.tenant = tenant_enabled
    second_response = middleware(second_request)
    assert second_response.status_code == 200
    masked_body = second_response.content.decode("utf-8")
    assert "user@example.com" not in masked_body
    assert "<EMAIL_" in masked_body

    assert responses == [plain_body, masked_body]
    assert get_session_scope() is None


def test_request_to_task_chain_preserves_session_scope(
    configure_autotest_tenant, scope_headers_shared, monkeypatch
):
    tenant = configure_autotest_tenant(
        pii_mode="gold",
        pii_policy="strict",
        pii_deterministic=True,
        pii_hmac_secret="secret-key",
    )
    factory = RequestFactory()
    request_tokens: list[str] = []
    task_records: list[dict[str, object]] = []

    class _CaptureTask(ScopedTask):
        abstract = False
        name = "tests.capture_scope_task"

        def run(self, step: str) -> str:  # type: ignore[override]
            config = get_pii_config()
            masked = mask_text(
                "Email a@b.de bitte prüfen",
                config["policy"],
                config["deterministic"],
                config["hmac_secret"],
                mode=config["mode"],
                name_detection=config["name_detection"],
            )
            task_records.append(
                {
                    "step": step,
                    "scope": get_session_scope(),
                    "masked": masked,
                }
            )
            return masked

    capture_task = _CaptureTask()
    capture_task.bind(current_app)

    def _execute_signature(signature: Signature):
        return signature.type.__call__(*signature.args, **signature.kwargs)

    def _fake_apply_async(self, *args, **kwargs):
        if hasattr(self, "tasks"):
            results = []
            for sub_sig in self.tasks:
                results.append(_execute_signature(sub_sig))
            body = getattr(self, "body", None)
            if body is not None:
                results.append(_execute_signature(body))
            return results
        return _execute_signature(self)

    monkeypatch.setattr(Signature, "apply_async", _fake_apply_async, raising=False)
    monkeypatch.setattr(_chain, "apply_async", _fake_apply_async, raising=False)

    def _view(request):
        config = get_pii_config()
        masked = mask_text(
            "Email a@b.de bitte prüfen",
            config["policy"],
            config["deterministic"],
            config["hmac_secret"],
            mode=config["mode"],
            name_detection=config["name_detection"],
        )
        request_tokens.append(masked)
        tenant_obj = getattr(request, "tenant", None)
        tenant_kwarg = getattr(tenant_obj, "id", None)
        scope = {
            "tenant_id": (
                tenant_kwarg
                if tenant_kwarg is not None
                else request.META["HTTP_X_TENANT_ID"]
            ),
            "case_id": request.META["HTTP_X_CASE_ID"],
            "trace_id": request.META["HTTP_X_TRACE_ID"],
            "session_salt": "||".join(
                (
                    request.META["HTTP_X_TRACE_ID"],
                    request.META["HTTP_X_CASE_ID"],
                    request.META["HTTP_X_TENANT_ID"],
                )
            ),
        }
        pipeline = chain(
            capture_task.s(step="producer"),
            capture_task.s(step="worker"),
        )
        with_scope_apply_async(pipeline, scope)
        return HttpResponse(masked)

    middleware = PIISessionScopeMiddleware(_view)

    request = factory.get("/echo/", **scope_headers_shared)
    request.tenant = tenant
    response = middleware(request)
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert request_tokens == [body]
    assert len(task_records) == 2

    tenant_identifier = getattr(request, "tenant", None)
    tenant_scope_id = str(
        getattr(tenant_identifier, "id", scope_headers_shared["HTTP_X_TENANT_ID"])
    )
    expected_scope = (
        tenant_scope_id,
        scope_headers_shared["HTTP_X_CASE_ID"],
        "||".join(
            (
                scope_headers_shared["HTTP_X_TRACE_ID"],
                scope_headers_shared["HTTP_X_CASE_ID"],
                scope_headers_shared["HTTP_X_TENANT_ID"],
            )
        ),
    )

    for record in task_records:
        assert record["scope"] == expected_scope

    task_masks = {record["masked"] for record in task_records}
    assert len(task_masks) == 1
    task_mask = task_masks.pop()
    assert "a@b.de" not in task_mask
    assert "<EMAIL_" in task_mask
    assert "a@b.de" not in body
    assert "<EMAIL_" in body

    assert get_session_scope() is None
