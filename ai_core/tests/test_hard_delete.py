import uuid

import pytest
from django.core.exceptions import PermissionDenied

from ai_core.rag.hard_delete import hard_delete
from documents.service_facade import DELETE_OUTBOX
from organizations.models import Organization, OrgMembership
from profiles.models import UserProfile
from users.models import User


@pytest.fixture(autouse=True)
def _clear_outbox():
    DELETE_OUTBOX.clear()


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

    result = hard_delete(
        tenant_id,
        [document_id],
        "cleanup",
        "TCK-1",
        actor={"internal_key": "service-key", "label": "ops-bot"},
        trace_id="trace-123",
    )

    assert result["status"] == "deleted"
    assert result["deleted_ids"] == []  # Document doesn't exist in DB
    assert result["not_found"] == 1
    assert result["visibility"] == "deleted"
    # DELETE_OUTBOX is not populated for non-existent documents

    repeat = hard_delete(
        tenant_id,
        [document_id],
        "cleanup",
        "TCK-1",
        actor={"internal_key": "service-key"},
        trace_id="trace-124",
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

    result = hard_delete(
        tenant_id,
        [document_id],
        "cleanup",
        "TCK-5",
        actor={"internal_key": "service-key"},
        trace_id=trace_id,
        session_salt=session_salt,
        session_scope=(tenant_id, case_id, session_salt),
        case_id=case_id,
        tenant_schema="tenant_schema",
        ingestion_run_id="ing-1",
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
        hard_delete(tenant_id, [document_id], "cleanup", "TCK-2")


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_hard_delete_allows_admin_user(monkeypatch, settings):
    tenant_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())

    user = User.objects.create_user(username="admin", email="admin@example.com")
    UserProfile.objects.update_or_create(
        user=user,
        defaults={"role": UserProfile.Roles.ADMIN, "is_active": True},
    )

    result = hard_delete(
        tenant_id,
        [document_id],
        "manual",
        "TCK-3",
        actor={"user_id": user.pk},
        tenant_schema="tenant_schema",
        ingestion_run_id="ing-2",
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

    result = hard_delete(
        tenant_id,
        [document_id],
        "manual",
        "TCK-4",
        actor={"user_id": user.pk, "organization_id": str(organization.id)},
    )

    assert result["status"] == "deleted"
    assert result["deleted_ids"] == []  # Document doesn't exist in DB
    assert result["not_found"] == 1
    # DELETE_OUTBOX is not populated for non-existent documents
