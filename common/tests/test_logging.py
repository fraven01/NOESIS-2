import importlib
import logging

import pytest
from django.http import HttpResponse
from django.test import RequestFactory
from structlog.testing import capture_logs

from common import logging as common_logging
from common.constants import (
    META_CASE_ID_KEY,
    META_KEY_ALIAS_KEY,
    META_TENANT_ID_KEY,
    META_TRACE_ID_KEY,
)
from common.middleware import RequestLogContextMiddleware


@pytest.fixture(autouse=True)
def _clear_log_context():
    common_logging.clear_log_context()
    yield
    common_logging.clear_log_context()


def _build_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )


def test_filter_masks_values_when_masking_enabled(settings):
    settings.LOGGING_ALLOW_UNMASKED_CONTEXT = False
    filt = common_logging.RequestTaskContextFilter()

    with common_logging.log_context(
        trace_id="trace-abcdef",
        case_id="case-12345",
        tenant="tenant-42",
        key_alias="alias-secret",
    ):
        record = _build_record()
        assert filt.filter(record) is True

    assert record.trace_id.startswith("tr")
    assert record.trace_id.endswith("ef")
    assert "***" in record.trace_id
    assert record.case_id != "case-12345"
    assert record.tenant != "tenant-42"
    assert record.key_alias != "alias-secret"


def test_filter_allows_unmasked_values_when_opted_in(settings):
    settings.LOGGING_ALLOW_UNMASKED_CONTEXT = True
    filt = common_logging.RequestTaskContextFilter()

    with common_logging.log_context(
        trace_id="trace-xyz",
        case_id="case-xyz",
        tenant="tenant-xyz",
        key_alias="alias-xyz",
    ):
        record = _build_record()
        filt.filter(record)

    assert record.trace_id == "trace-xyz"
    assert record.case_id == "case-xyz"
    assert record.tenant == "tenant-xyz"
    assert record.key_alias == "alias-xyz"


def test_filter_sets_placeholders_when_context_missing(settings):
    settings.LOGGING_ALLOW_UNMASKED_CONTEXT = False
    record = _build_record()
    filt = common_logging.RequestTaskContextFilter()

    filt.filter(record)

    assert record.trace_id == "-"
    assert record.case_id == "-"
    assert record.tenant == "-"
    assert record.key_alias == "-"


def test_structlog_logger_includes_context_and_service(monkeypatch):
    monkeypatch.setitem(common_logging._SERVICE_CONTEXT, "service.name", "svc-test")
    monkeypatch.setitem(
        common_logging._SERVICE_CONTEXT, "deployment.environment", "pytest"
    )

    with capture_logs() as logs:
        with common_logging.log_context(
            trace_id="trace-xyz123", case_id="case-789", tenant="tenant-42"
        ):
            common_logging.get_logger("common.tests").info("hello", extra_field="value")

    assert logs, "expected log entry"
    event = logs[0]
    assert event["event"] == "hello"
    assert event["extra_field"] == "value"
    assert event["service.name"] == "svc-test"
    assert event["deployment.environment"] == "pytest"
    assert event["trace_id"].startswith("tr")
    assert event["trace_id"].endswith("23")
    assert "***" in event["trace_id"]
    assert event["case_id"].startswith("ca")
    assert event["tenant"].startswith("te")


def test_request_log_context_middleware_binds_and_clears():
    factory = RequestFactory()
    request = factory.get(
        "/ai/ping/",
        **{
            META_TRACE_ID_KEY: "trace-123",
            META_CASE_ID_KEY: "case-456",
            META_TENANT_ID_KEY: "tenant-789",
            META_KEY_ALIAS_KEY: "alias-001",
        },
    )

    observed: dict[str, str] = {}

    def _view(inner_request):
        nonlocal observed
        observed = common_logging.get_log_context()
        return HttpResponse("ok")

    middleware = RequestLogContextMiddleware(_view)
    response = middleware(request)

    assert response.status_code == 200
    assert observed["trace_id"] == "trace-123"
    assert observed["case_id"] == "case-456"
    assert observed["tenant"] == "tenant-789"
    assert observed["key_alias"] == "alias-001"
    assert common_logging.get_log_context() == {}


def _reload_logging_module():
    """Reload the logging module to rebuild service context from env vars."""

    return importlib.reload(common_logging)


def test_service_context_prefers_deploy_env(monkeypatch):
    monkeypatch.setenv("DEPLOY_ENV", "staging")
    monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "legacy")

    module = _reload_logging_module()
    assert module._SERVICE_CONTEXT["deployment.environment"] == "staging"

    monkeypatch.delenv("DEPLOY_ENV", raising=False)
    monkeypatch.delenv("DEPLOYMENT_ENVIRONMENT", raising=False)
    _reload_logging_module()


def test_service_context_falls_back_to_legacy_env(monkeypatch):
    monkeypatch.delenv("DEPLOY_ENV", raising=False)
    monkeypatch.setenv("DEPLOYMENT_ENVIRONMENT", "legacy")

    module = _reload_logging_module()
    assert module._SERVICE_CONTEXT["deployment.environment"] == "legacy"

    monkeypatch.delenv("DEPLOYMENT_ENVIRONMENT", raising=False)
    _reload_logging_module()


def test_service_context_defaults_to_unknown(monkeypatch):
    monkeypatch.delenv("DEPLOY_ENV", raising=False)
    monkeypatch.delenv("DEPLOYMENT_ENVIRONMENT", raising=False)

    module = _reload_logging_module()
    assert module._SERVICE_CONTEXT["deployment.environment"] == "unknown"

    _reload_logging_module()
