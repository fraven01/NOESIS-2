import pytest
from django.urls import reverse
from django.contrib.auth import get_user_model
from theme.middleware import SimulatedUserMiddleware
from ai_core.contracts.scope import ScopeContext


@pytest.fixture
def simulated_user_middleware():
    return SimulatedUserMiddleware(get_response=lambda r: None)


@pytest.mark.django_db
def test_simulated_user_middleware_switches_identity(client, settings, monkeypatch):
    settings.DEBUG = True
    settings.RAG_TOOLS_ENABLED = True

    # Patch tenant resolution to avoid needing full tenant setup
    monkeypatch.setattr(
        "theme.views._scope_context_from_request",
        lambda r: ScopeContext(
            tenant_id="dev-tenant",
            tenant_schema="dev-schema",
            trace_id="trace",
            invocation_id="invocation",
            run_id="run",
            service_id="test-worker",
        ),
    )

    # Create users
    User = get_user_model()
    real_user = User.objects.create_user(
        "real_user", "real@example.com", "password", is_staff=True
    )
    simulated_user = User.objects.create_user(
        "simulated_alice", "alice@example.com", "password"
    )

    # Login as real user
    client.force_login(real_user)

    # 1. Access rag-tools as real user
    url = reverse("rag-tools")
    resp = client.get(url)
    assert resp.status_code == 200
    # In the template, we'd expect the identity switcher to show "real_user" or similar,
    # but programmatically request.user should be real_user
    assert resp.context["request"].user == real_user
    assert not getattr(resp.context["request"], "is_simulated_user", False)

    # 2. Switch identity via POST (sets session)
    switch_url = reverse("rag_tools_identity_switch")
    resp = client.post(switch_url, {"user_id": simulated_user.pk})
    assert resp.status_code == 302

    # 3. Access rag-tools again - Middleware should swap user
    resp = client.get(url)
    assert resp.status_code == 200
    assert resp.context["request"].user == simulated_user
    assert getattr(resp.context["request"], "is_simulated_user", True)
    assert resp.context["current_simulated_user_id"] == str(simulated_user.pk)


@pytest.mark.django_db
def test_simulated_user_middleware_ignored_in_production(client, settings, monkeypatch):
    settings.DEBUG = False
    settings.RAG_TOOLS_ENABLED = False
    settings.TESTING = (
        True  # Allow view access, but middleware should still be disabled
    )

    # Patch tenant resolution to avoid needing full tenant setup
    monkeypatch.setattr(
        "theme.views._scope_context_from_request",
        lambda r: ScopeContext(
            tenant_id="dev-tenant",
            tenant_schema="dev-schema",
            trace_id="trace",
            invocation_id="invocation",
            run_id="run",
            service_id="test-worker",
        ),
    )

    User = get_user_model()
    simulated_user = User.objects.create_user(
        "simulated_bob", "bob@example.com", "password"
    )

    # Manually inject session
    session = client.session
    session["rag_tools_simulated_user_id"] = str(simulated_user.pk)
    session.save()

    url = reverse("rag-tools")
    # This might redirect to login because we aren't authenticated as real user
    # and middleware shouldn't kick in to provide a user.
    # Assuming rag-tools requires login or handles anon:
    resp = client.get(url)

    # If middleware worked, user would be bob. If not, it's AnonymousUser.
    assert str(resp.wsgi_request.user) == "AnonymousUser"
    assert not getattr(resp.wsgi_request, "is_simulated_user", False)
