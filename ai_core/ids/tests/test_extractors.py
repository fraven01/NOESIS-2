"""Tests for shared ID extractors."""

from types import SimpleNamespace

from ai_core.ids.extractors import extract_runtime_ids


def test_extract_runtime_ids_prefers_request_attrs():
    request = SimpleNamespace(
        run_id="run-attr",
        ingestion_run_id="ing-attr",
        headers={"X-Run-ID": "run-hdr"},
        META={"HTTP_X_INGESTION_RUN_ID": "ing-meta"},
    )

    run_id, ingestion_run_id = extract_runtime_ids(request=request)

    assert run_id == "run-attr"
    assert ingestion_run_id == "ing-attr"


def test_extract_runtime_ids_from_headers():
    request = SimpleNamespace(headers={"X-Run-ID": "run-1"}, META={})

    run_id, ingestion_run_id = extract_runtime_ids(request=request)

    assert run_id == "run-1"
    assert ingestion_run_id is None


def test_extract_runtime_ids_from_meta():
    request = SimpleNamespace(headers={}, META={"HTTP_X_INGESTION_RUN_ID": "ing-1"})

    run_id, ingestion_run_id = extract_runtime_ids(request=request)

    assert run_id is None
    assert ingestion_run_id == "ing-1"


def test_extract_runtime_ids_generates_when_missing():
    run_id, ingestion_run_id = extract_runtime_ids(headers={}, meta={})

    assert run_id is not None
    assert ingestion_run_id is None
