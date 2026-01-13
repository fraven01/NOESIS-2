import json

import pytest
import yaml
from django.conf import settings
from django.test.utils import override_settings
from django.urls import include, path, reverse
from drf_spectacular.generators import SchemaGenerator
from drf_spectacular.settings import spectacular_settings
from rest_framework import serializers, viewsets
from rest_framework.response import Response
from rest_framework.routers import DefaultRouter
from rest_framework.views import APIView

from noesis2.api import (
    RATE_LIMIT_JSON_ERROR_STATUSES,
    schema as api_schema,
)
from noesis2.settings.base import build_swagger_ui_settings
from testsupport.tenant_fixtures import ensure_tenant_domain


@pytest.fixture
def tenant(db, tenant_pool):
    tenant = tenant_pool["alpha"]
    ensure_tenant_domain(tenant, domain="testserver")
    return tenant


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_openapi_schema_endpoint_accessible_without_headers(client, tenant):
    response = client.get(reverse("api-schema"))

    assert response.status_code == 200
    assert response["content-type"].startswith("application/vnd.oai.openapi")
    schema = yaml.safe_load(response.content)
    assert "openapi" in schema


def test_schema_description_includes_common_headers():
    description = settings.SPECTACULAR_SETTINGS["DESCRIPTION"]
    assert "> **Common Headers**" in description
    assert "| Header |" in description


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@override_settings(
    ENABLE_API_DOCS=True,
    API_DOCS_TITLE="NOESIS Docs",
    API_DOCS_VERSION_LABEL="v1.2.3-deadbeef",
    ENABLE_SWAGGER_TRY_IT_OUT=False,
)
def test_swagger_ui_available(client, tenant, settings, monkeypatch):
    settings.SPECTACULAR_SETTINGS["SWAGGER_UI_SETTINGS"] = build_swagger_ui_settings(
        settings.ENABLE_SWAGGER_TRY_IT_OUT
    )
    monkeypatch.setattr(
        spectacular_settings,
        "SWAGGER_UI_SETTINGS",
        build_swagger_ui_settings(settings.ENABLE_SWAGGER_TRY_IT_OUT),
    )
    response = client.get(reverse("api-docs-swagger"))

    assert response.status_code == 200
    # Django exposes template context as a list; drf-spectacular renders into index 0
    swagger_data = response.data
    assert swagger_data["title"] == "NOESIS Docs (v1.2.3-deadbeef)"
    swagger_settings = json.loads(swagger_data["settings"])
    assert swagger_settings["supportedSubmitMethods"] == []
    assert swagger_settings["tryItOutEnabled"] is False


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@override_settings(
    ENABLE_API_DOCS=True,
    API_DOCS_TITLE="NOESIS Docs",
    API_DOCS_VERSION_LABEL="v9.9.9-stage",
)
def test_redoc_uses_custom_title(client, tenant):
    response = client.get(reverse("api-docs-redoc"))

    assert response.status_code == 200
    redoc_data = response.data
    assert redoc_data["title"] == "NOESIS Docs (v9.9.9-stage)"


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@override_settings(ENABLE_API_DOCS=True, ENABLE_SWAGGER_TRY_IT_OUT=True)
def test_swagger_ui_try_it_out_enabled(client, tenant, settings, monkeypatch):
    settings.SPECTACULAR_SETTINGS["SWAGGER_UI_SETTINGS"] = build_swagger_ui_settings(
        True
    )
    monkeypatch.setattr(
        spectacular_settings,
        "SWAGGER_UI_SETTINGS",
        build_swagger_ui_settings(True),
    )
    response = client.get(reverse("api-docs-swagger"))

    assert response.status_code == 200
    swagger_data = response.data
    swagger_settings = json.loads(swagger_data["settings"])
    assert swagger_settings["tryItOutEnabled"] is True
    assert "post" in swagger_settings["supportedSubmitMethods"]


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@override_settings(ENABLE_API_DOCS=False)
def test_swagger_ui_disabled_returns_404(client, tenant, settings):
    settings.SPECTACULAR_SETTINGS["SWAGGER_UI_SETTINGS"] = build_swagger_ui_settings(
        settings.ENABLE_SWAGGER_TRY_IT_OUT
    )
    response = client.get(reverse("api-docs-swagger"))

    assert response.status_code == 404


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
@override_settings(ENABLE_API_DOCS=False)
def test_redoc_disabled_returns_404(client, tenant):
    response = client.get(reverse("api-docs-redoc"))

    assert response.status_code == 404


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_schema_components_document_headers_and_security(client, tenant):
    response = client.get(reverse("api-schema"))

    assert response.status_code == 200
    schema = yaml.safe_load(response.content)

    parameter_components = schema["components"]["parameters"]
    expected_components = api_schema.tenant_header_components()
    for component_name, definition in expected_components.items():
        assert component_name in parameter_components
        assert parameter_components[component_name]["name"] == definition["name"]
        assert parameter_components[component_name]["in"] == "header"
        assert parameter_components[component_name]["style"] == "simple"

    trace_header = schema["components"]["headers"][
        api_schema.TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME
    ]
    assert trace_header["name"] == api_schema.TRACE_ID_RESPONSE_HEADER_COMPONENT["name"]
    assert trace_header["schema"]["type"] == "string"

    security_schemes = schema["components"]["securitySchemes"]
    bearer = security_schemes[api_schema.ADMIN_BEARER_AUTH_SCHEME]
    assert bearer["type"] == "http"
    assert bearer["scheme"] == "bearer"

    assert settings.SPECTACULAR_SETTINGS["SECURITY"] == [
        {api_schema.ADMIN_BEARER_AUTH_SCHEME: []}
    ]
    assert schema["info"]["version"] == "v1"


def _generate_schema(patterns):
    generator = SchemaGenerator(patterns=patterns)
    return generator.get_schema(request=None, public=True)


def _iter_schema_operations(schema):
    for path_item in schema.get("paths", {}).values():
        if not isinstance(path_item, dict):
            continue
        for maybe_operation in path_item.values():
            if isinstance(maybe_operation, dict):
                yield maybe_operation


def _extract_component(schema, ref):
    component_name = ref.split("/")[-1]
    return component_name, schema["components"]["schemas"][component_name]


def test_default_extend_schema_view_auto_decorates_apiview():
    schema = _generate_schema(
        [path("decorated/", _decorated_api_view.as_view(), name="decorated")]
    )

    get_operation = schema["paths"]["/decorated/"]["get"]

    parameter_names = {parameter["name"] for parameter in get_operation["parameters"]}
    expected_names = {
        definition["name"]
        for definition in api_schema.tenant_header_components().values()
    }
    assert expected_names.issubset(parameter_names)

    responses = get_operation["responses"]
    expected_error_statuses = {"400", "401", "403", "404", "503"}
    assert expected_error_statuses.issubset(responses.keys())
    assert "429" not in responses

    for status_code in expected_error_statuses:
        response = responses[status_code]
        assert response.get("headers") == api_schema.trace_response_headers()
        content = response["content"]["application/json"]
        error_ref = content["schema"]["$ref"]
        _, error_component = _extract_component(schema, error_ref)
        assert {"status", "error", "meta"}.issubset(
            error_component["properties"].keys()
        )

    bad_request_examples = responses["400"]["content"]["application/json"].get(
        "examples", {}
    )
    assert any(
        example.get("value", {}).get("error", {}).get("code") == "tenant_not_found"
        for example in bad_request_examples.values()
    )


def test_default_extend_schema_view_auto_decorates_viewset():
    router = DefaultRouter()
    router.register(
        "decorated-viewset", _decorated_viewset, basename="decorated-viewset"
    )

    schema = _generate_schema(router.urls)
    get_operation = schema["paths"]["/decorated-viewset/"]["get"]

    parameter_names = {parameter["name"] for parameter in get_operation["parameters"]}
    expected_names = {
        definition["name"]
        for definition in api_schema.tenant_header_components().values()
    }
    assert expected_names.issubset(parameter_names)

    responses = get_operation["responses"]
    for status_code, response in responses.items():
        assert response.get("headers") == api_schema.trace_response_headers()
        if status_code in {"400", "401", "403", "404", "503"}:
            content = response["content"]["application/json"]
            error_ref = content["schema"]["$ref"]
            _, error_component = _extract_component(schema, error_ref)
            assert {"status", "error", "meta"}.issubset(
                error_component["properties"].keys()
            )


def test_ai_core_endpoints_expose_serializers():
    schema = _generate_schema(
        [
            path("ai/", include("ai_core.urls")),
            path("v1/ai/", include(("ai_core.urls_v1", "ai_core_v1"))),
        ]
    )

    ping_operation = schema["paths"]["/v1/ai/ping/"]["get"]
    ping_ref = ping_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    component_name, component = _extract_component(schema, ping_ref)
    assert component_name.startswith("PingResponse")
    assert component["properties"]["ok"]["type"] == "boolean"
    assert any(
        "curl" in sample.get("source", "")
        for sample in ping_operation.get("x-codeSamples", [])
    )
    ping_examples = ping_operation["responses"]["200"]["content"][
        "application/json"
    ].get("examples", {})
    assert any(
        example.get("value", {}).get("ok") is True for example in ping_examples.values()
    )
    legacy_ping_operation = schema["paths"]["/v1/ai/ping/"]["get"]
    assert legacy_ping_operation.get("deprecated") is True
    assert not ping_operation.get("deprecated")

    intake_operation = schema["paths"]["/v1/ai/intake/"]["post"]
    intake_request_ref = intake_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    request_component_name, request_component = _extract_component(
        schema, intake_request_ref
    )
    assert request_component_name.startswith("IntakeRequest")
    assert request_component["type"] == "object"
    assert "metadata" in request_component.get("properties", {})

    intake_response_ref = intake_operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    _, intake_response_component = _extract_component(schema, intake_response_ref)
    assert "tenant_id" in intake_response_component["properties"]
    assert intake_response_component["properties"]["tenant_id"]["type"] == "string"
    assert intake_response_component["properties"]["idempotent"]["type"] == "boolean"
    intake_request_examples = intake_operation["requestBody"]["content"][
        "application/json"
    ].get("examples", {})
    assert any(
        "prompt" in example.get("value", {})
        for example in intake_request_examples.values()
    )
    intake_response_examples = intake_operation["responses"]["200"]["content"][
        "application/json"
    ].get("examples", {})
    assert any(
        example.get("value", {}).get("tenant_id") == "acme"
        for example in intake_response_examples.values()
    )
    assert any(
        "curl" in sample.get("source", "")
        for sample in intake_operation.get("x-codeSamples", [])
    )

    legacy_intake_operation = schema["paths"]["/v1/ai/intake/"]["post"]
    assert legacy_intake_operation.get("deprecated") is True

    intake_responses = intake_operation["responses"]
    for status_code in map(str, RATE_LIMIT_JSON_ERROR_STATUSES):
        assert status_code in intake_responses

    rate_limit_examples = intake_responses["429"]["content"]["application/json"].get(
        "examples", {}
    )
    assert any(
        example.get("value", {}).get("error", {}).get("code") == "rate_limit_exceeded"
        for example in rate_limit_examples.values()
    )
    unsupported_media_examples = intake_responses["415"]["content"][
        "application/json"
    ].get("examples", {})
    assert any(
        example.get("value", {}).get("error", {}).get("code")
        == "unsupported_media_type"
        for example in unsupported_media_examples.values()
    )
    service_unavailable_examples = intake_responses["503"]["content"][
        "application/json"
    ].get("examples", {})
    assert any(
        example.get("value", {}).get("error", {}).get("code") == "service_unavailable"
        for example in service_unavailable_examples.values()
    )

    rag_operation = schema["paths"]["/v1/ai/rag/query/"]["post"]
    rag_response_ref = rag_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    _, rag_component = _extract_component(schema, rag_response_ref)
    rag_properties = rag_component["properties"]
    assert {"answer", "prompt_version", "retrieval", "snippets"}.issubset(
        rag_properties
    )

    retrieval_schema = rag_properties["retrieval"]
    if "$ref" in retrieval_schema:
        _, retrieval_component = _extract_component(schema, retrieval_schema["$ref"])
    else:
        retrieval_component = retrieval_schema
    retrieval_props = retrieval_component["properties"]
    expected_retrieval = {
        "alpha",
        "min_sim",
        "top_k_effective",
        "matches_returned",
        "max_candidates_effective",
        "vector_candidates",
        "lexical_candidates",
        "deleted_matches_blocked",
        "visibility_effective",
        "took_ms",
        "routing",
    }
    assert expected_retrieval.issubset(retrieval_props)

    routing_schema = retrieval_props["routing"]
    if "$ref" in routing_schema:
        _, routing_component = _extract_component(schema, routing_schema["$ref"])
    else:
        routing_component = routing_schema
    routing_props = routing_component["properties"]
    assert {"profile", "vector_space_id"}.issubset(routing_props)

    snippets_schema = rag_properties["snippets"]
    snippet_items = snippets_schema.get("items", {})
    if "$ref" in snippet_items:
        _, snippet_component = _extract_component(schema, snippet_items["$ref"])
    else:
        snippet_component = snippet_items
    snippet_props = snippet_component["properties"]
    assert snippet_props["score"]["type"] == "number"

    error_components = [
        component
        for component in schema["components"]["schemas"].values()
        if {"status", "error", "meta"}.issubset(component.get("properties", {}).keys())
    ]
    assert error_components, "Expected shared error schema to be generated"


def test_schema_operations_do_not_duplicate_parameters():
    schema = _generate_schema(
        [
            path("ai/", include("ai_core.urls")),
            path("v1/ai/", include(("ai_core.urls_v1", "ai_core_v1"))),
        ]
    )

    for operation in _iter_schema_operations(schema):
        parameters = operation.get("parameters", [])
        names = [parameter.get("name") for parameter in parameters]
        assert len(names) == len(set(names)), operation.get("operationId")


def test_tenant_demo_endpoint_documented():
    from common.views import DemoView

    schema = _generate_schema(
        [path("tenant-demo/", DemoView.as_view(), name="tenant-demo")]
    )
    demo_operation = schema["paths"]["/tenant-demo/"]["get"]
    demo_ref = demo_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    _, demo_component = _extract_component(schema, demo_ref)
    assert demo_component["properties"]["status"]["type"] == "string"
    assert any(
        "curl" in sample.get("source", "")
        for sample in demo_operation.get("x-codeSamples", [])
    )
    demo_examples = demo_operation["responses"]["200"]["content"][
        "application/json"
    ].get("examples", {})
    assert any(
        example.get("value", {}).get("status") == "ok"
        for example in demo_examples.values()
    )


class _DecoratedAPIView(APIView):
    def get(self, request):
        return Response({"status": "ok"})


_decorated_api_view = api_schema.default_extend_schema_view(include_trace_header=True)(
    _DecoratedAPIView
)


class _DummySerializer(serializers.Serializer):
    status = serializers.CharField()


class _DecoratedViewSet(viewsets.ViewSet):
    serializer_class = _DummySerializer

    def list(self, request):
        return Response([])


_decorated_viewset = api_schema.default_extend_schema_view(include_trace_header=True)(
    _DecoratedViewSet
)
