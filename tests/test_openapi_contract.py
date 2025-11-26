import pytest
from drf_spectacular.views import SpectacularAPIView
from rest_framework.test import APIRequestFactory

from noesis2.api import schema as api_schema


@pytest.fixture(scope="module")
def openapi_schema():
    """Render the OpenAPI schema once for the module-level contract tests."""
    factory = APIRequestFactory()
    view = SpectacularAPIView.as_view()
    response = view(factory.get("/api/schema/"))
    assert response.status_code == 200
    return response.data


def _resolve_component(schema, definition):
    if "$ref" in definition:
        component_name = definition["$ref"].split("/")[-1]
        return schema["components"]["schemas"][component_name]
    return definition


def _example_values(examples):
    if isinstance(examples, dict):
        return examples.values()
    if isinstance(examples, (list, tuple)):
        return examples
    return []


def test_required_header_components_are_registered(openapi_schema):
    parameter_components = openapi_schema["components"]["parameters"]
    expected = api_schema.tenant_header_components()
    for component_name, definition in expected.items():
        assert component_name in parameter_components
        component = parameter_components[component_name]
        assert component["name"] == definition["name"]
        assert component["in"] == "header"
        assert component.get("style") == "simple"

    headers = openapi_schema["components"]["headers"]
    assert (
        api_schema.TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME in headers
    ), "Missing shared X-Trace-ID header component"


def test_expected_paths_are_present(openapi_schema):
    paths = set(openapi_schema["paths"].keys())
    required_paths = {
        "/ai/ping/",
        "/ai/intake/",
        "/ai/rag/query/",
        "/v1/ai/ping/",
        "/v1/ai/intake/",
        "/v1/ai/rag/query/",
    }
    missing = sorted(required_paths - paths)
    assert not missing, f"Missing expected API paths: {missing}"


@pytest.mark.parametrize(
    "path_prefix",
    ["/ai/", "/v1/ai/"],
)
def test_trace_header_documented_for_ai_endpoints(openapi_schema, path_prefix):
    for path, path_item in openapi_schema["paths"].items():
        if not path.startswith(path_prefix):
            continue
        for method, operation in path_item.items():
            success = operation.get("responses", {}).get("200")
            if not success:
                continue
            headers = success.get("headers", {})
            assert "X-Trace-ID" in headers
            assert headers["X-Trace-ID"]["$ref"].split("/")[-1] == (
                api_schema.TRACE_ID_RESPONSE_HEADER_COMPONENT_NAME
            )


def test_post_responses_include_idempotent_flag(openapi_schema):
    for path, path_item in openapi_schema["paths"].items():
        for method, operation in path_item.items():
            if method.lower() != "post":
                continue
            if path in {"/ai/rag/query/", "/v1/ai/rag/query/"}:
                continue

            responses = operation.get("responses", {})
            success = None
            for status in ("200", "201", "202"):
                if status in responses:
                    success = responses[status]
                    break
            if not success:
                continue
            content = success.get("content", {}).get("application/json")
            assert content, f"{path} {method} missing JSON response content"
            resolved = _resolve_component(openapi_schema, content["schema"])
            properties = resolved.get("properties", {})
            assert (
                "idempotent" in properties
            ), f"{path} {method} missing idempotent flag"
            idempotent_schema = properties["idempotent"]
            assert idempotent_schema.get("type") == "boolean"


def test_post_requests_document_unsupported_media_type(openapi_schema):
    for path, path_item in openapi_schema["paths"].items():
        for method, operation in path_item.items():
            if method.lower() != "post":
                continue

            request_body = operation.get("requestBody")
            if not request_body:
                continue

            request_content = request_body.get("content", {})
            if "application/json" not in request_content:
                continue

            responses = operation.get("responses", {})
            unsupported = responses.get("415")
            if not unsupported and path == "/v1/ai/frameworks/analyze/":
                print(f"DEBUG: {path} {method} responses keys: {list(responses.keys())}")
            assert unsupported, f"{path} {method} missing 415 response"

            content = unsupported.get("content", {}).get("application/json", {})
            examples = content.get("examples") or {}
            values = list(_example_values(examples))
            assert any(
                example.get("value", {}).get("code") == "unsupported_media_type"
                for example in values
            ), f"{path} {method} missing unsupported_media_type example"
            assert "Request payload must be encoded as application/json." in {
                example.get("value", {}).get("detail") for example in values
            }

            json_error = responses.get("400")
            if json_error:
                json_content = json_error.get("content", {}).get("application/json", {})
                json_examples = json_content.get("examples") or {}
                assert any(
                    example.get("value", {}).get("code") == "invalid_json"
                    for example in _example_values(json_examples)
                ), f"{path} {method} missing invalid_json example"


def test_operation_parameters_are_unique(openapi_schema):
    for path_item in openapi_schema["paths"].values():
        for operation in path_item.values():
            parameters = operation.get("parameters", [])
            names = [parameter.get("name") for parameter in parameters]
            assert len(names) == len(set(names)), names
