from __future__ import annotations

import pytest
from django.test import RequestFactory

from customers.models import Tenant
from documents.activity_service import ActivityTracker
from documents.models import Document
from users.models import User


pytestmark = pytest.mark.django_db


def _create_document(*, tenant: Tenant, case_id: str | None = None) -> Document:
    return Document.objects.create(
        tenant=tenant,
        hash="activity-hash",
        source="upload",
        metadata={},
        case_id=case_id,
    )


def test_log_activity_with_user_and_request(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    user = User.objects.create_user(username="activity-user")
    document = _create_document(tenant=tenant, case_id="case-1")

    factory = RequestFactory()
    request = factory.get("/documents/recent/", HTTP_USER_AGENT="agent")
    request.META["REMOTE_ADDR"] = "127.0.0.1"

    activity = ActivityTracker.log(
        document=document,
        activity_type="DOWNLOAD",
        user=user,
        request=request,
        trace_id="trace-1",
        metadata={"source": "upload"},
    )

    assert activity is not None
    assert activity.document_id == document.id
    assert activity.user_id == user.id
    assert activity.ip_address == "127.0.0.1"
    assert activity.user_agent == "agent"
    assert activity.case_id == "case-1"
    assert activity.trace_id == "trace-1"
    assert activity.metadata["source"] == "upload"


def test_log_activity_without_user(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    document = _create_document(tenant=tenant)

    activity = ActivityTracker.log(
        document=document,
        activity_type="VIEW",
    )

    assert activity is not None
    assert activity.user_id is None
    assert activity.ip_address is None
    assert activity.user_agent == ""


def test_log_activity_with_document_id(test_tenant_schema_name):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    document = _create_document(tenant=tenant, case_id="case-42")

    activity = ActivityTracker.log(
        document_id=document.id,
        activity_type="DOWNLOAD",
        tenant_schema=test_tenant_schema_name,
    )

    assert activity is not None
    assert activity.document_id == document.id
    assert activity.case_id == "case-42"
