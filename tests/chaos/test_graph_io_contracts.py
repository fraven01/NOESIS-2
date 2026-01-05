"""Chaos tests for Graph I/O Spec contracts.

Tests the versioned Pydantic I/O models at graph boundaries, ensuring proper
schema_id/schema_version validation and rejection of invalid contracts.

Contract under test:
- ai_core/graphs/technical/universal_ingestion_graph.py: UniversalIngestionGraphInput/Output
- ai_core/graph/io.py: GraphIOSpec structure
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.graphs.technical.universal_ingestion_graph import (
    UNIVERSAL_INGESTION_SCHEMA_ID,
    UNIVERSAL_INGESTION_IO_VERSION_STRING,
    UniversalIngestionGraphInput,
    UniversalIngestionGraphOutput,
    UniversalIngestionInputModel,
)
from tests.chaos.conftest import _build_chaos_meta

pytestmark = pytest.mark.chaos


def test_universal_ingestion_io_spec_valid():
    """Valid I/O spec with correct schema_id and schema_version."""
    graph_input = UniversalIngestionGraphInput(
        schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
        schema_version=UNIVERSAL_INGESTION_IO_VERSION_STRING,
        input=UniversalIngestionInputModel(
            normalized_document={
                "title": "Test Document",
                "content": "Test content",
            }
        ),
        context=_build_chaos_meta(
            tenant_id="tenant-uig",
            trace_id="trace-uig",
            collection_id="col-uig",
        ),
    )

    assert graph_input.schema_id == "noesis.graphs.universal_ingestion"
    assert graph_input.schema_version == "1.0.0"
    assert isinstance(graph_input.input, UniversalIngestionInputModel)
    assert isinstance(graph_input.context, dict)


def test_universal_ingestion_wrong_schema_id():
    """Wrong schema_id rejected by Pydantic literal validation."""
    with pytest.raises(ValidationError) as exc_info:
        UniversalIngestionGraphInput(
            schema_id="wrong.schema.id",  # INVALID
            schema_version=UNIVERSAL_INGESTION_IO_VERSION_STRING,
            input=UniversalIngestionInputModel(),
            context={},
        )

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("schema_id",) for err in errors)


def test_universal_ingestion_wrong_schema_version():
    """Wrong schema_version rejected by Pydantic literal validation."""
    with pytest.raises(ValidationError) as exc_info:
        UniversalIngestionGraphInput(
            schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
            schema_version="99.99.99",  # INVALID version
            input=UniversalIngestionInputModel(),
            context={},
        )

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("schema_version",) for err in errors)


def test_graph_output_includes_schema_metadata():
    """Graph output includes schema_id/schema_version in telemetry."""
    graph_output = UniversalIngestionGraphOutput(
        schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
        schema_version=UNIVERSAL_INGESTION_IO_VERSION_STRING,
        decision="processed",
        reason_code=None,
        reason=None,
        document_id="doc-123",
        ingestion_run_id="ingestion-123",
        telemetry={
            "took_ms": 250,
            "schema_id": UNIVERSAL_INGESTION_SCHEMA_ID,
            "schema_version": UNIVERSAL_INGESTION_IO_VERSION_STRING,
        },
        formatted_status="Processed successfully",
    )

    assert graph_output.schema_id == "noesis.graphs.universal_ingestion"
    assert graph_output.schema_version == "1.0.0"
    assert graph_output.telemetry["schema_id"] == UNIVERSAL_INGESTION_SCHEMA_ID
    assert (
        graph_output.telemetry["schema_version"]
        == UNIVERSAL_INGESTION_IO_VERSION_STRING
    )


def test_graph_input_rejects_extra_fields():
    """Graph input model rejects extra fields (strict validation)."""
    with pytest.raises(ValidationError) as exc_info:
        UniversalIngestionGraphInput(
            schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
            schema_version=UNIVERSAL_INGESTION_IO_VERSION_STRING,
            input=UniversalIngestionInputModel(),
            context={},
            extra_field="not_allowed",  # INVALID - extra field
        )

    errors = exc_info.value.errors()
    assert any("extra_field" in str(err) for err in errors)


def test_graph_input_frozen_immutable():
    """Graph input model is frozen (immutable after creation)."""
    graph_input = UniversalIngestionGraphInput(
        schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
        schema_version=UNIVERSAL_INGESTION_IO_VERSION_STRING,
        input=UniversalIngestionInputModel(),
        context={},
    )

    with pytest.raises(ValidationError):
        graph_input.schema_id = "modified"  # INVALID - frozen model


def test_graph_output_decision_enum_validation():
    """Graph output validates decision field as strict enum."""
    # Valid decision values
    for decision in ["processed", "skipped", "failed"]:
        output = UniversalIngestionGraphOutput(
            decision=decision,
            reason_code=None,
            reason=None,
            document_id=None,
            ingestion_run_id=None,
            telemetry={},
            formatted_status=None,
        )
        assert output.decision == decision

    # Invalid decision value
    with pytest.raises(ValidationError) as exc_info:
        UniversalIngestionGraphOutput(
            decision="invalid_decision",  # INVALID
            reason_code=None,
            reason=None,
            document_id=None,
            ingestion_run_id=None,
            telemetry={},
            formatted_status=None,
        )

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("decision",) for err in errors)


def test_graph_output_reason_code_enum_validation():
    """Graph output validates reason_code field as strict enum."""
    # Valid reason codes
    for reason_code in [
        "DUPLICATE",
        "VALIDATION_ERROR",
        "PERSISTENCE_ERROR",
        "PROCESSING_ERROR",
        None,
    ]:
        output = UniversalIngestionGraphOutput(
            decision="failed",
            reason_code=reason_code,
            reason="Test reason",
            document_id=None,
            ingestion_run_id=None,
            telemetry={},
            formatted_status=None,
        )
        assert output.reason_code == reason_code

    # Invalid reason code
    with pytest.raises(ValidationError) as exc_info:
        UniversalIngestionGraphOutput(
            decision="failed",
            reason_code="INVALID_CODE",  # INVALID
            reason="Test reason",
            document_id=None,
            ingestion_run_id=None,
            telemetry={},
            formatted_status=None,
        )

    errors = exc_info.value.errors()
    assert any(err["loc"] == ("reason_code",) for err in errors)


def test_graph_io_version_as_literal():
    """schema_version must match exact version string (no wildcards)."""
    # Correct version
    graph_input = UniversalIngestionGraphInput(
        schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
        schema_version="1.0.0",  # Exact match
        input=UniversalIngestionInputModel(),
        context={},
    )
    assert graph_input.schema_version == "1.0.0"

    # Wrong version format (missing patch)
    with pytest.raises(ValidationError):
        UniversalIngestionGraphInput(
            schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
            schema_version="1.0",  # INVALID - must be "1.0.0"
            input=UniversalIngestionInputModel(),
            context={},
        )

    # Wrong version (future version)
    with pytest.raises(ValidationError):
        UniversalIngestionGraphInput(
            schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
            schema_version="2.0.0",  # INVALID - future version
            input=UniversalIngestionInputModel(),
            context={},
        )
