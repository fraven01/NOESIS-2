"""API views for framework agreement analysis."""

from __future__ import annotations

from uuid import uuid4

from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiParameter,
    OpenApiResponse,
    inline_serializer,
)
from drf_spectacular.types import OpenApiTypes
from rest_framework import status, serializers
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from pydantic import ValidationError

from common.constants import X_TENANT_ID_HEADER, X_TRACE_ID_HEADER
from common.logging import bind_log_context, get_logger
from noesis2.api import default_extend_schema

from ai_core.graphs.framework_analysis_graph import build_graph
from ai_core.tools.framework_contracts import (
    FrameworkAnalysisInput,
    FrameworkAnalysisErrorCode,
    map_framework_error_to_status,
)
from ai_core.infra.resp import apply_std_headers


logger = get_logger(__name__)


def _error_response(
    message: str,
    error_code: str,
    http_status: int,
) -> Response:
    """Build standardized error response."""
    return Response(
        {
            "error": message,
            "error_code": error_code,
        },
        status=http_status,
    )


def _prepare_framework_request(request: Request) -> tuple[dict, Response | None]:
    """
    Prepare and validate framework analysis request.

    Returns:
        Tuple of (meta_dict, error_response)
        If error_response is not None, request should be rejected
    """
    from customers.tenant_context import TenantContext, TenantRequiredError

    # Extract headers
    tenant_header = request.headers.get(X_TENANT_ID_HEADER)
    trace_id = request.headers.get(X_TRACE_ID_HEADER) or str(uuid4())

    # Validate tenant
    try:
        tenant_obj = TenantContext.from_request(
            request,
            allow_headers=False,
            require=True,
            use_connection_schema=False,
        )
    except TenantRequiredError as exc:
        return {}, _error_response(
            str(exc),
            "tenant_not_found",
            status.HTTP_403_FORBIDDEN,
        )

    tenant_schema = getattr(tenant_obj, "schema_name", "")
    if not tenant_schema:
        return {}, _error_response(
            "Tenant schema could not be resolved from request context.",
            "tenant_not_found",
            status.HTTP_403_FORBIDDEN,
        )

    if tenant_header is None:
        return {}, _error_response(
            "X-Tenant-ID header is required.",
            "invalid_tenant_header",
            status.HTTP_400_BAD_REQUEST,
        )

    tenant_id = tenant_header.strip()
    if not tenant_id:
        return {}, _error_response(
            "X-Tenant-ID header cannot be empty.",
            "invalid_tenant_header",
            status.HTTP_400_BAD_REQUEST,
        )

    meta = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "trace_id": trace_id,
    }

    # Bind logging context
    bind_log_context(
        tenant_id=tenant_id,
        tenant_schema=tenant_schema,
        trace_id=trace_id,
    )

    return meta, None


FRAMEWORK_ANALYSIS_REQUEST_EXAMPLES = [
    OpenApiExample(
        "Basic Analysis",
        value={
            "document_collection_id": "550e8400-e29b-41d4-a716-446655440000",
            "document_id": "650e8400-e29b-41d4-a716-446655440001",
        },
        description="Analyze a specific document in a collection",
        request_only=True,
    ),
    OpenApiExample(
        "Collection-Wide Analysis",
        value={
            "document_collection_id": "550e8400-e29b-41d4-a716-446655440000",
        },
        description="Analyze all documents in a collection",
        request_only=True,
    ),
    OpenApiExample(
        "Force Reanalysis",
        value={
            "document_collection_id": "550e8400-e29b-41d4-a716-446655440000",
            "force_reanalysis": True,
            "confidence_threshold": 0.80,
        },
        description="Force reanalysis with custom confidence threshold",
        request_only=True,
    ),
]

FRAMEWORK_ANALYSIS_REQUEST = inline_serializer(
    name="FrameworkAnalysisRequest",
    fields={
        "document_collection_id": serializers.UUIDField(),
        "document_id": serializers.UUIDField(required=False, allow_null=True),
        "force_reanalysis": serializers.BooleanField(required=False),
        "confidence_threshold": serializers.FloatField(
            required=False, min_value=0.0, max_value=1.0
        ),
    },
)

FRAMEWORK_ANALYSIS_RESPONSE_EXAMPLE = OpenApiExample(
    "Success",
    value={
        "profile_id": "750e8400-e29b-41d4-a716-446655440002",
        "version": 1,
        "gremium_identifier": "KBR",
        "completeness_score": 0.75,
        "missing_components": ["zugriffsrechte"],
        "hitl_required": False,
        "hitl_reasons": [],
        "idempotent": True,
        "structure": {
            "systembeschreibung": {
                "location": "main",
                "outline_path": "2",
                "heading": "Section 2 Systembeschreibung",
                "chunk_ids": ["chunk1", "chunk2"],
                "page_numbers": [2, 3],
                "confidence": 0.92,
                "validated": True,
                "validation_notes": "Plausible",
            },
            "funktionsbeschreibung": {
                "location": "annex",
                "outline_path": "Anlage 1",
                "confidence": 0.88,
                "validated": True,
            },
            "auswertungen": {
                "location": "annex_group",
                "outline_path": "Anlage 3",
                "annex_root": "Anlage 3",
                "subannexes": ["3.1", "3.2"],
                "confidence": 0.85,
                "validated": True,
                "validation_notes": "Plausible",
            },
            "zugriffsrechte": {
                "location": "not_found",
                "confidence": 0.0,
                "validated": False,
            },
        },
        "analysis_metadata": {
            "detected_type": "kbv",
            "type_confidence": 0.95,
            "gremium_name_raw": "Konzernbetriebsrat der Telefonica Deutschland",
            "gremium_identifier": "KBR",
            "completeness_score": 0.75,
            "missing_components": ["zugriffsrechte"],
            "analysis_timestamp": "2025-01-15T10:30:00Z",
            "model_version": "framework_analysis_v1",
        },
    },
    response_only=True,
)


FRAMEWORK_ANALYSIS_SCHEMA = {
    "summary": "Analyze Framework Agreement",
    "description": """
Analyze a framework agreement document (KBV/GBV/BV) to extract structure and components.

The analysis identifies:
- Framework type (KBV/GBV/BV/DV)
- Gremium information
- Four key components: Systembeschreibung, Funktionsbeschreibung, Auswertungen, Zugriffsrechte
- Document structure (main document vs annexes)

Returns a FrameworkProfile with structural metadata and completeness score.
""",
    "request": FRAMEWORK_ANALYSIS_REQUEST,
    "examples": FRAMEWORK_ANALYSIS_REQUEST_EXAMPLES,
    "responses": {
        200: OpenApiResponse(
            response=inline_serializer(
                name="FrameworkAnalysisResponse",
                fields={
                    "profile_id": serializers.UUIDField(),
                    "version": serializers.IntegerField(),
                    "gremium_identifier": serializers.CharField(),
                    "completeness_score": serializers.FloatField(),
                    "missing_components": serializers.ListField(
                        child=serializers.CharField()
                    ),
                    "hitl_required": serializers.BooleanField(),
                    "hitl_reasons": serializers.ListField(
                        child=serializers.CharField()
                    ),
                    "idempotent": serializers.BooleanField(),
                    "structure": serializers.DictField(),
                    "analysis_metadata": serializers.DictField(),
                },
            ),
            description="Analysis completed successfully",
            examples=[FRAMEWORK_ANALYSIS_RESPONSE_EXAMPLE],
        ),
        400: OpenApiResponse(
            response=OpenApiTypes.OBJECT,
            description="Invalid request parameters",
            examples=[
                OpenApiExample(
                    "invalid_json",
                    value={
                        "code": "invalid_json",
                        "detail": "Request body is not valid JSON.",
                    },
                    media_type="application/json",
                    response_only=True,
                )
            ],
        ),
        403: OpenApiResponse(description="Tenant not found or unauthorized"),
        404: OpenApiResponse(description="Document or collection not found"),
        409: OpenApiResponse(
            description="Profile already exists (use force_reanalysis=true)"
        ),
        415: OpenApiResponse(
            response=OpenApiTypes.OBJECT,
            description="Unsupported Media Type",
            examples=[
                OpenApiExample(
                    "unsupported_media_type",
                    value={
                        "code": "unsupported_media_type",
                        "detail": "Request payload must be encoded as application/json.",
                    },
                    media_type="application/json",
                    response_only=True,
                )
            ],
        ),
        500: OpenApiResponse(description="Analysis failed"),
    },
    "parameters": [
        OpenApiParameter(
            name=X_TENANT_ID_HEADER,
            type=str,
            location=OpenApiParameter.HEADER,
            required=True,
            description="Tenant identifier",
        ),
        OpenApiParameter(
            name=X_TRACE_ID_HEADER,
            type=str,
            location=OpenApiParameter.HEADER,
            required=False,
            description="Trace ID for observability (auto-generated if not provided)",
        ),
    ],
}


class FrameworkAnalysisView(APIView):
    """
    API endpoint for framework agreement analysis.

    POST /v1/ai/frameworks/analyze/
    """

    @default_extend_schema(include_trace_header=True, **FRAMEWORK_ANALYSIS_SCHEMA)
    def post(self, request: Request) -> Response:
        """
        Analyze a framework agreement document.

        Request body:
        {
            "document_collection_id": "uuid",
            "document_id": "uuid" (optional),
            "force_reanalysis": false (optional),
            "confidence_threshold": 0.70 (optional)
        }
        """
        # Prepare request and validate tenant
        meta, error = _prepare_framework_request(request)
        if error:
            return error

        tenant_id = meta["tenant_id"]
        tenant_schema = meta["tenant_schema"]
        trace_id = meta["trace_id"]

        # Parse request body
        try:
            request_data = dict(request.data or {})

            # Validate and parse input
            input_params = FrameworkAnalysisInput(**request_data)

        except ValidationError as e:
            logger.warning(
                "framework_analysis_invalid_input",
                extra={
                    "tenant_id": tenant_id,
                    "trace_id": trace_id,
                    "error": str(e),
                },
            )
            return _error_response(
                f"Invalid request parameters: {str(e)}",
                "invalid_input",
                status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.error(
                "framework_analysis_request_error",
                extra={
                    "tenant_id": tenant_id,
                    "trace_id": trace_id,
                    "error": str(e),
                },
            )
            return _error_response(
                "Failed to parse request body",
                "request_parse_error",
                status.HTTP_400_BAD_REQUEST,
            )

        # Execute analysis graph
        try:
            logger.info(
                "framework_analysis_starting",
                extra={
                    "tenant_id": tenant_id,
                    "trace_id": trace_id,
                    "document_collection_id": str(input_params.document_collection_id),
                    "document_id": (
                        str(input_params.document_id)
                        if input_params.document_id
                        else None
                    ),
                },
            )

            graph = build_graph()
            output = graph.run(
                input_params=input_params,
                tenant_id=tenant_id,
                tenant_schema=tenant_schema,
                trace_id=trace_id,
            )

            logger.info(
                "framework_analysis_completed",
                extra={
                    "tenant_id": tenant_id,
                    "trace_id": trace_id,
                    "profile_id": str(output.profile_id),
                    "completeness_score": output.completeness_score,
                    "hitl_required": output.hitl_required,
                },
            )

            # Convert output to dict
            response_data = output.model_dump(mode="json")

            response = Response(response_data, status=status.HTTP_200_OK)
            return apply_std_headers(response, meta)

        except Exception as e:
            logger.error(
                "framework_analysis_failed",
                extra={
                    "tenant_id": tenant_id,
                    "trace_id": trace_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )

            # Map error to appropriate HTTP status
            error_code = FrameworkAnalysisErrorCode.ANALYSIS_FAILED
            http_status = map_framework_error_to_status(error_code)

            return _error_response(
                f"Framework analysis failed: {str(e)}",
                error_code,
                http_status,
            )
