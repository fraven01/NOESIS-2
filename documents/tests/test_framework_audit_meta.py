from __future__ import annotations

from uuid import uuid4

import pytest

from customers.models import Tenant
from documents.framework_models import FrameworkProfile
from documents.models import DocumentCollection
from documents.services.framework_service import persist_profile


pytestmark = pytest.mark.django_db


def test_persist_profile_stores_audit_meta(test_tenant_schema_name) -> None:
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)
    collection = DocumentCollection.objects.create(
        tenant=tenant,
        name="Framework Collection",
        key=f"framework-{uuid4()}",
        collection_id=uuid4(),
    )
    audit_meta = {
        "created_by_user_id": str(uuid4()),
        "last_hop_service_id": "framework-worker",
    }

    profile = persist_profile(
        tenant_schema=tenant.schema_name,
        gremium_identifier="KBR",
        gremium_name_raw="Konzernbetriebsrat",
        agreement_type="kbv",
        structure={},
        document_collection_id=str(collection.id),
        audit_meta=audit_meta,
    )

    stored = FrameworkProfile.objects.get(id=profile.id)
    assert stored.audit_meta == audit_meta
