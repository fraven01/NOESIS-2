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


def test_session_scope_middleware_sets_scope_and_masks(
    pii_config_env, scope_headers_shared
):
    pii_config_env(PII_DETERMINISTIC=True, PII_HMAC_SECRET="secret-key")
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

    try:
        request1 = factory.get("/echo/", **scope_headers_shared)
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
        response2 = middleware(request2)
        assert response2.status_code == 200
        body2 = response2.content.decode("utf-8")
        assert body2 == responses[1]
        assert observed_scopes[1] == expected_scope
        assert body2 == body1
        assert get_session_scope() is None
    finally:
        pii_config_env()


def test_session_scope_middleware_clears_scope_on_exception(
    pii_config_env, scope_headers_shared
):
    pii_config_env(PII_DETERMINISTIC=True, PII_HMAC_SECRET="secret-key")
    factory = RequestFactory()
    observed_scopes: list[tuple[str, ...]] = []

    def _view(request):
        observed_scopes.append(get_session_scope())
        raise RuntimeError("boom")

    middleware = PIISessionScopeMiddleware(_view)

    try:
        request = factory.get("/echo/", **scope_headers_shared)
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
    finally:
        pii_config_env()


def test_session_scope_middleware_changes_token_with_new_trace(
    pii_config_env, scope_headers_shared, scope_headers_new_trace
):
    pii_config_env(PII_DETERMINISTIC=True, PII_HMAC_SECRET="secret-key")
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

    try:
        request_same = factory.get("/echo/", **scope_headers_shared)
        response_same = middleware(request_same)
        assert response_same.status_code == 200
        first_placeholder = response_same.content.decode("utf-8")
        assert placeholders[0] == first_placeholder

        request_new = factory.get("/echo/", **scope_headers_new_trace)
        response_new = middleware(request_new)
        assert response_new.status_code == 200
        second_placeholder = response_new.content.decode("utf-8")
        assert placeholders[1] == second_placeholder

        assert first_placeholder != second_placeholder
        assert "<EMAIL_" in first_placeholder
        assert "<EMAIL_" in second_placeholder
        assert get_session_scope() is None
    finally:
        pii_config_env()


def test_request_to_task_chain_preserves_session_scope(
    pii_config_env, scope_headers_shared, monkeypatch
):
    pii_config_env(PII_DETERMINISTIC=True, PII_HMAC_SECRET="secret-key")
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
        scope = {
            "tenant_id": request.META["HTTP_X_TENANT_ID"],
            "case_id": request.META["HTTP_X_CASE_ID"],
            "trace_id": request.META["HTTP_X_TRACE_ID"],
        }
        pipeline = chain(
            capture_task.s(step="producer"),
            capture_task.s(step="worker"),
        )
        with_scope_apply_async(pipeline, scope)
        return HttpResponse(masked)

    middleware = PIISessionScopeMiddleware(_view)

    try:
        request = factory.get("/echo/", **scope_headers_shared)
        response = middleware(request)
        assert response.status_code == 200
        body = response.content.decode("utf-8")
        assert request_tokens == [body]
        assert len(task_records) == 2

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

        for record in task_records:
            assert record["masked"] == body
            assert record["scope"] == expected_scope

        assert get_session_scope() is None
    finally:
        pii_config_env()
