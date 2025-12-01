import json
import pytest
from django.test import RequestFactory
from customers.tests.factories import TenantFactory
from ai_core.views import crawl_selected


@pytest.mark.django_db
def test_reproduce_crawl_selected_error():
    tenant_schema = "test_tenant"
    tenant = TenantFactory(schema_name=tenant_schema)

    # Case creation removed
    # with tenant_context(tenant):
    #     Case.objects.create(tenant=tenant, external_id="case-crawl-test")

    factory = RequestFactory()
    data = {
        "urls": ["https://example.com"],
        "collection_id": "00000000-0000-0000-0000-000000000000",
        "mode": "live",
    }

    # Create a request that mimics what web_search_ingest_selected sends
    request = factory.post(
        "/crawl-selected/", data=json.dumps(data), content_type="application/json"
    )
    request.tenant = tenant
    request.tenant_schema = tenant_schema

    # Add headers that web_search_ingest_selected adds
    request.META["HTTP_X_TENANT_ID"] = tenant_schema
    request.META["HTTP_X_TENANT_SCHEMA"] = tenant_schema
    request.META["HTTP_X_TRACE_ID"] = "test-trace-id"
    # request.META["HTTP_X_CASE_ID"] = "case-crawl-test"

    # Call the view directly
    response = crawl_selected(request)

    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response content: {response.content.decode()}")

    assert response.status_code == 202
