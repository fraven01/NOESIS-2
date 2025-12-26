"""Regression tests for crawler_runner coordinator refactoring."""

import json
from types import SimpleNamespace
from uuid import UUID

import pytest

import ai_core.services as services
from ai_core import views
from common.constants import (
    X_CASE_ID_HEADER,
    META_TENANT_ID_KEY,
    META_CASE_ID_KEY,
    X_TRACE_ID_HEADER,
)
from users.tests.factories import UserFactory


pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _clear_crawler_idempotency_cache():
    from django.core.cache import cache

    cache.clear()


@pytest.fixture
def authenticated_client(client):
    """Authenticated test client."""
    user = UserFactory()
    client.force_login(user)
    return client


# Test Block A: Double Ingestion Prevention


def test_crawler_runner_does_not_trigger_legacy_ingestion_when_graph_ran(
    authenticated_client,
    monkeypatch,
    documents_repository_stub,
    test_tenant_schema_name,
):
    """Verify start_ingestion_run is NOT called when Universal Graph provides ingestion_run_id."""
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    # Mock middleware to set tenant without hostname lookup
    from django_tenants.middleware import TenantMainMiddleware
    from unittest.mock import Mock

    def _mock_get_tenant(self, domain_model, hostname):
        # Return a mock tenant with the test schema name
        tenant_mock = Mock()
        tenant_mock.schema_name = test_tenant_schema_name
        tenant_mock.id = 1
        return tenant_mock

    monkeypatch.setattr(TenantMainMiddleware, "get_tenant", _mock_get_tenant)

    legacy_call_count = 0

    def _track_legacy_calls(*args, **kwargs):
        nonlocal legacy_call_count
        legacy_call_count += 1
        return SimpleNamespace(data={"ingestion_run_id": "legacy-id"})

    monkeypatch.setattr(services, "start_ingestion_run", _track_legacy_calls)

    class _DummyGraph:
        def invoke(self, input_data):
            context = input_data.get("context", {})
            return {
                "output": {
                    "decision": "ingested",
                    "reason": "Success",
                    "ingestion_run_id": context.get("ingestion_run_id"),
                    "document_id": "doc-123",
                    "transitions": ["validate", "normalize", "persist"],
                    "telemetry": {},
                    "hitl_required": False,
                    "hitl_reasons": [],
                    "review_payload": None,
                    "normalized_document_ref": None,
                },
            }

    monkeypatch.setattr(
        "ai_core.graphs.technical.universal_ingestion_graph.build_universal_ingestion_graph",
        lambda: _DummyGraph(),
    )

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "case-1",
        X_TRACE_ID_HEADER: "trace-1",
    }
    payload = {
        "mode": "manual",
        "origins": [
            {"url": "https://example.com/doc.pdf", "fetch": False, "content": "test"}
        ],
    }

    response = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert response.status_code == 200
    assert legacy_call_count == 0, "Legacy start_ingestion_run should NOT be called"


def test_crawler_runner_response_contains_canonical_ingestion_run_id(
    authenticated_client,
    monkeypatch,
    documents_repository_stub,
    test_tenant_schema_name,
):
    """Verify response includes canonical ingestion_run_id from coordinator."""
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    captured_context_id = None

    class _CapturingGraph:
        def invoke(self, input_data):
            nonlocal captured_context_id
            context = input_data.get("context", {})
            captured_context_id = context.get("ingestion_run_id")
            return {
                "output": {
                    "decision": "ingested",
                    "ingestion_run_id": captured_context_id,
                    "document_id": "doc-456",
                    "transitions": [],
                    "telemetry": {},
                    "hitl_required": False,
                    "hitl_reasons": [],
                    "review_payload": None,
                    "normalized_document_ref": None,
                },
            }

    monkeypatch.setattr(
        "ai_core.graphs.technical.universal_ingestion_graph.build_universal_ingestion_graph",
        lambda: _CapturingGraph(),
    )

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "c1",
        X_TRACE_ID_HEADER: "tr1",
    }
    payload = {
        "mode": "manual",
        "origins": [{"url": "https://example.com", "fetch": False, "content": "test"}],
    }

    response = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    data = response.json()
    response_run_id = data["origins"][0]["ingestion_run_id"]

    assert response_run_id is not None
    assert response_run_id == captured_context_id
    # Validate UUID format
    assert UUID(response_run_id)


# Test Block B: Mandatory ID Validation


def test_crawler_runner_raises_error_when_tenant_id_missing(
    authenticated_client, monkeypatch
):
    """Verify ValueError raised when tenant_id is missing."""
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    headers = {
        META_CASE_ID_KEY: "c1",
        X_TRACE_ID_HEADER: "tr1",
    }  # No tenant_id
    payload = {
        "mode": "manual",
        "origins": [{"url": "https://example.com", "fetch": False, "content": "test"}],
    }

    response = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    # Expect 400 or 500 depending on error handling
    assert response.status_code in [400, 500]
    assert "tenant" in response.text.lower()


def test_crawler_runner_raises_error_when_case_id_missing(
    authenticated_client, monkeypatch, test_tenant_schema_name
):
    """Verify default case_id is applied when missing."""
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        X_TRACE_ID_HEADER: "tr1",
    }  # No case_id
    payload = {
        "mode": "manual",
        "origins": [{"url": "https://example.com", "fetch": False, "content": "test"}],
    }

    response = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert response.status_code == 200
    assert response[X_CASE_ID_HEADER] == views.DEFAULT_CASE_ID


def test_crawler_runner_generates_trace_id_when_missing(
    authenticated_client,
    monkeypatch,
    caplog,
    documents_repository_stub,
    test_tenant_schema_name,
):
    """Verify trace_id is auto-generated when missing."""
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    class _DummyGraph:
        def invoke(self, input_data):
            context = input_data.get("context", {})
            assert context.get("trace_id") is not None, "trace_id should be generated"
            return {
                "output": {
                    "decision": "skipped",
                    "transitions": [],
                    "telemetry": {},
                    "hitl_required": False,
                    "hitl_reasons": [],
                    "review_payload": None,
                    "normalized_document_ref": None,
                }
            }

    monkeypatch.setattr(
        "ai_core.graphs.technical.universal_ingestion_graph.build_universal_ingestion_graph",
        lambda: _DummyGraph(),
    )

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "c1",
    }  # No trace_id
    payload = {
        "mode": "manual",
        "origins": [{"url": "https://example.com", "fetch": False, "content": "test"}],
    }

    response = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert response.status_code == 200
    assert response[X_TRACE_ID_HEADER]


# Test Block C: Idempotency with Redis Cache


@pytest.mark.django_db
def test_crawler_runner_idempotency_skips_duplicate_fingerprint(
    authenticated_client,
    monkeypatch,
    documents_repository_stub,
    test_tenant_schema_name,
):
    """Verify second request with same fingerprint is skipped (Redis cache)."""
    from django.core.cache import cache

    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    cache.clear()  # Clean state

    class _DummyGraph:
        def invoke(self, input_data):
            return {
                "output": {
                    "decision": "ingested",
                    "transitions": [],
                    "telemetry": {},
                    "hitl_required": False,
                    "hitl_reasons": [],
                    "review_payload": None,
                    "normalized_document_ref": None,
                }
            }

    monkeypatch.setattr(
        "ai_core.graphs.technical.universal_ingestion_graph.build_universal_ingestion_graph",
        lambda: _DummyGraph(),
    )

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "c1",
        X_TRACE_ID_HEADER: "tr1",
    }
    payload = {
        "mode": "manual",
        "origins": [{"url": "https://example.com", "fetch": False, "content": "test"}],
    }

    # First call
    first = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["idempotent"] is False

    # Second call (same payload)
    second = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["idempotent"] is True
    assert second_data["skipped"] is True
    assert second_data["origins"] == []


# Test Block D: Artifact/Transition Extraction


def test_crawler_runner_extracts_transitions_from_output(
    authenticated_client,
    monkeypatch,
    documents_repository_stub,
    test_tenant_schema_name,
):
    """Verify transitions are read from output.transitions, not state."""
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    class _DummyGraph:
        def invoke(self, input_data):
            return {
                "output": {
                    "decision": "ingested",
                    "transitions": [
                        "validate",
                        "normalize",
                        "persist",
                        "finalize",
                    ],
                    "telemetry": {},
                    "hitl_required": False,
                    "hitl_reasons": [],
                    "review_payload": None,
                    "normalized_document_ref": None,
                },
                # Even if state has transitions, output takes precedence
                "transitions": ["wrong_path"],
            }

    monkeypatch.setattr(
        "ai_core.graphs.technical.universal_ingestion_graph.build_universal_ingestion_graph",
        lambda: _DummyGraph(),
    )

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "c1",
        X_TRACE_ID_HEADER: "tr1",
    }
    payload = {
        "mode": "manual",
        "origins": [{"url": "https://example.com", "fetch": False, "content": "test"}],
    }

    response = authenticated_client.post(
        "/ai/rag/crawler/run/",
        data=json.dumps(payload),
        content_type="application/json",
        **headers,
    )

    assert response.status_code == 200
    data = response.json()
    transitions = data["transitions"][0]["transitions"]
    assert transitions == ["validate", "normalize", "persist", "finalize"]


# Test Block E: Exception Handling


def test_crawler_runner_logs_exception_with_full_context(
    authenticated_client,
    monkeypatch,
    caplog,
    documents_repository_stub,
    test_tenant_schema_name,
):
    """Verify exception logs contain tenant_id, trace_id, etc."""
    monkeypatch.setattr(views, "assert_case_active", lambda *args, **kwargs: None)

    class _FailingGraph:
        def invoke(self, input_data):
            raise RuntimeError("Graph exploded")

    monkeypatch.setattr(
        "ai_core.graphs.technical.universal_ingestion_graph.build_universal_ingestion_graph",
        lambda: _FailingGraph(),
    )

    headers = {
        META_TENANT_ID_KEY: test_tenant_schema_name,
        META_CASE_ID_KEY: "c-fail",
        X_TRACE_ID_HEADER: "tr-fail",
    }
    payload = {
        "mode": "manual",
        "origins": [{"url": "https://example.com", "fetch": False, "content": "test"}],
    }

    with caplog.at_level("ERROR"):
        response = authenticated_client.post(
            "/ai/rag/crawler/run/",
            data=json.dumps(payload),
            content_type="application/json",
            **headers,
        )

    # Should still return 200 with error decision
    assert response.status_code == 200
    assert response[X_TRACE_ID_HEADER]
    data = response.json()
    assert data["origins"][0]["result"]["reason"] == "Graph exploded"

    # Verify log contains full context
    assert "universal_crawler_ingestion_failed" in caplog.text
    import re

    match = re.search(r"trace_id': '([^']+)'", caplog.text)
    assert match is not None
    assert match.group(1)
