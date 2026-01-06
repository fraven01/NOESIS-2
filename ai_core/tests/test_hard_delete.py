import uuid

import pytest
from django.core.exceptions import PermissionDenied

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.rag.hard_delete import hard_delete
from documents.service_facade import DELETE_OUTBOX
from organizations.models import Organization, OrgMembership
from profiles.models import UserProfile
from users.models import User


@pytest.fixture(autouse=True)
def _clear_outbox():
    DELETE_OUTBOX.clear()


def _build_meta(
    *,
    tenant_id: str,
    case_id: str | None = None,
    trace_id: str | None = None,
    tenant_schema: str | None = None,
    ingestion_run_id: str | None = None,
) -> dict[str, object]:
    scope_context = ScopeContext.model_validate(
        {
            "tenant_id": tenant_id,
            "trace_id": trace_id or uuid.uuid4().hex,
            "invocation_id": uuid.uuid4().hex,
            "run_id": uuid.uuid4().hex,
            "tenant_schema": tenant_schema,
            "ingestion_run_id": ingestion_run_id,
        }
    )
    business_context = BusinessContext(case_id=case_id)
    tool_context = scope_context.to_tool_context(business=business_context)
    return {
        "scope_context": scope_context.model_dump(mode="json", exclude_none=True),
        "business_context": business_context.model_dump(mode="json", exclude_none=True),
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
    }


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_service_key(monkeypatch, settings):
    settings.RAG_INTERNAL_KEYS = ["service-key"]
    tenant_id = str(uuid.uuid4())

    document_id = str(uuid.uuid4())

    spans: list[dict[str, object]] = []

    def _capture_span(
        name: str,
        *,
        attributes: dict[str, object] | None = None,
        trace_id: str | None = None,
    ) -> None:
        spans.append(
            {"trace_id": trace_id, "node_name": name, "metadata": attributes or {}}
        )

    monkeypatch.setattr("ai_core.rag.hard_delete.record_span", _capture_span)

    meta = _build_meta(tenant_id=tenant_id, trace_id="trace-123")
    state = {
        "tenant_id": tenant_id,
        "document_ids": [document_id],
        "reason": "cleanup",
        "ticket_ref": "TCK-1",
        "trace_id": "trace-123",
    }
    result = hard_delete(
        state,
        meta,
        actor={"internal_key": "service-key", "label": "ops-bot"},
    )

    assert result["status"] == "deleted"
    assert result["deleted_ids"] == []  # Document doesn't exist in DB
    assert result["not_found"] == 1
    assert result["visibility"] == "deleted"
    # DELETE_OUTBOX is not populated for non-existent documents

    repeat_meta = _build_meta(tenant_id=tenant_id, trace_id="trace-124")
    repeat_state = {
        "tenant_id": tenant_id,
        "document_ids": [document_id],
        "reason": "cleanup",
        "ticket_ref": "TCK-1",
        "trace_id": "trace-124",
    }
    repeat = hard_delete(
        repeat_state,
        repeat_meta,
        actor={"internal_key": "service-key"},
    )

    assert repeat["status"] == "deleted"
    assert repeat["not_found"] == 1
    assert len(spans) == 2, "expected Langfuse spans for each invocation"
    assert spans[0]["node_name"] == "rag.hard_delete"
    assert spans[0]["trace_id"] == "trace-123"
    assert spans[0]["metadata"]["documents_requested"] == 1
    assert spans[1]["trace_id"] == "trace-124"


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_service_key_with_scoped_session(monkeypatch, settings):
    settings.RAG_INTERNAL_KEYS = ["service-key"]
    tenant_id = str(uuid.uuid4())

    document_id = str(uuid.uuid4())

    trace_id = "trace-321"
    case_id = "case-123"
    session_salt = "||".join((trace_id, case_id, tenant_id))

    spans: list[dict[str, object]] = []

    def _capture_span(
        trace_id: str, node_name: str, metadata: dict[str, object]
    ) -> None:
        spans.append(
            {"trace_id": trace_id, "node_name": node_name, "metadata": metadata}
        )

    monkeypatch.setattr("ai_core.rag.hard_delete.record_span", _capture_span)

    meta = _build_meta(
        tenant_id=tenant_id,
        case_id=case_id,
        trace_id=trace_id,
        tenant_schema="tenant_schema",
        ingestion_run_id="ing-1",
    )
    state = {
        "tenant_id": tenant_id,
        "document_ids": [document_id],
        "reason": "cleanup",
        "ticket_ref": "TCK-5",
        "case_id": case_id,
        "tenant_schema": "tenant_schema",
        "ingestion_run_id": "ing-1",
        "trace_id": trace_id,
    }
    result = hard_delete(
        state,
        meta,
        actor={"internal_key": "service-key"},
    )

    assert result["status"] == "deleted"
    assert result["deleted_ids"] == []  # Document doesn't exist in DB
    assert result["not_found"] == 1
    assert spans, "expected span emission"
    assert spans[0]["metadata"].get("session_salt") == session_salt
    # DELETE_OUTBOX is not populated for non-existent documents


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_requires_authorisation(monkeypatch, settings):
    settings.RAG_INTERNAL_KEYS = ["service-key"]
    tenant_id = str(uuid.uuid4())

    document_id = str(uuid.uuid4())

    with pytest.raises(PermissionDenied):
        hard_delete(
            {
                "tenant_id": tenant_id,
                "document_ids": [document_id],
                "reason": "cleanup",
                "ticket_ref": "TCK-2",
            },
            _build_meta(tenant_id=tenant_id),
        )


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_allows_admin_user(monkeypatch, settings):
    tenant_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    user = User.objects.create_user(username="admin", email="admin@example.com")
    UserProfile.objects.update_or_create(
        user=user,
        defaults={"role": UserProfile.Roles.TENANT_ADMIN, "is_active": True},
    )

    meta = _build_meta(
        tenant_id=tenant_id,
        tenant_schema="tenant_schema",
        ingestion_run_id="ing-2",
    )
    state = {
        "tenant_id": tenant_id,
        "document_ids": [document_id],
        "reason": "manual",
        "ticket_ref": "TCK-3",
        "tenant_schema": "tenant_schema",
        "ingestion_run_id": "ing-2",
    }
    result = hard_delete(
        state,
        meta,
        actor={"user_id": user.pk},
    )

    assert result["status"] == "deleted"
    assert result["deleted_ids"] == []  # Document doesn't exist in DB
    assert result["not_found"] == 1
    # DELETE_OUTBOX is not populated for non-existent documents


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_allows_org_admin(monkeypatch, settings):
    tenant_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    user = User.objects.create_user(username="org-admin", email="org@example.com")
    UserProfile.objects.update_or_create(user=user, defaults={"is_active": True})
    organization = Organization.objects.create(name="Org", slug="org")
    OrgMembership.objects.create(
        organization=organization,
        user=user,
        role=OrgMembership.Role.ADMIN,
    )

    meta = _build_meta(tenant_id=tenant_id)
    state = {
        "tenant_id": tenant_id,
        "document_ids": [document_id],
        "reason": "manual",
        "ticket_ref": "TCK-4",
    }
    result = hard_delete(
        state,
        meta,
        actor={"user_id": user.pk, "organization_id": str(organization.id)},
    )

    assert result["status"] == "deleted"
    assert result["deleted_ids"] == []  # Document doesn't exist in DB
    assert result["not_found"] == 1
    # DELETE_OUTBOX is not populated for non-existent documents
