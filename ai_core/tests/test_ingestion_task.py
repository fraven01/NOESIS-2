from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pytest
from celery.exceptions import TimeoutError as CeleryTimeoutError

from ai_core import ingestion
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts import ToolContext
from ai_core.rag.ingestion_contracts import IngestionContractErrorCode
from ai_core.tools import InputError


class DummyProcessTask:
    def __init__(self) -> None:
        self.calls: List[tuple[Any, ...]] = []

    def s(self, *args: Any) -> tuple[str, tuple[Any, ...]]:
        self.calls.append(args)
        return ("signature", args)


class DummyChildResult:
    def __init__(
        self,
        document_id: str,
        *,
        ready: bool,
        partial: Optional[Dict[str, Any]] = None,
        get_exception: Optional[BaseException] = None,
    ) -> None:
        self.document_id = document_id
        self._ready = ready
        self._partial = partial or {"document_id": document_id}
        self._get_exception = get_exception
        self.revoked: List[Dict[str, Any]] = []

    def get(self, timeout: Optional[float] = None, propagate: bool = True):
        if self._get_exception is not None:
            raise self._get_exception
        return self._partial

    def ready(self) -> bool:
        return self._ready

    def revoke(self, terminate: bool = True) -> None:
        self.revoked.append({"terminate": terminate})


class DummyAsyncResult:
    def __init__(self, results: Iterable[DummyChildResult]) -> None:
        self.results = list(results)
        self.get_calls: List[Dict[str, Any]] = []
        self._results_payload: Optional[List[Dict[str, Any]]] = None
        self.should_raise: Optional[BaseException] = None

    def get(
        self,
        timeout: Optional[float] = None,
        disable_sync_subtasks: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        self.get_calls.append(
            {"timeout": timeout, "disable_sync_subtasks": disable_sync_subtasks}
        )
        if self.should_raise is not None:
            raise self.should_raise
        assert self._results_payload is not None
        return self._results_payload


def _setup_common_monkeypatches(monkeypatch):
    dummy_process = DummyProcessTask()
    monkeypatch.setattr(ingestion, "process_document", dummy_process)

    start_calls: List[Dict[str, Any]] = []
    end_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        ingestion.pipe,
        "log_ingestion_run_start",
        lambda **kwargs: start_calls.append(kwargs),
    )
    monkeypatch.setattr(
        ingestion.pipe,
        "log_ingestion_run_end",
        lambda **kwargs: end_calls.append(kwargs),
    )

    apply_async_calls: List[Dict[str, Any]] = []

    def fake_apply_async(*args: Any, **kwargs: Any) -> None:
        apply_async_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(ingestion.record_dead_letter, "apply_async", fake_apply_async)

    schema_calls: List[Dict[str, Any]] = []

    def fake_ensure_schema(space):
        schema_calls.append(
            {
                "id": space.id,
                "schema": space.schema,
                "backend": space.backend,
                "dimension": space.dimension,
            }
        )
        return False

    monkeypatch.setattr(ingestion, "ensure_vector_space_schema", fake_ensure_schema)

    return (
        dummy_process,
        start_calls,
        end_calls,
        apply_async_calls,
        schema_calls,
    )


def _expected_ingestion_resolution(profile: str = "standard"):
    binding = ingestion.resolve_ingestion_profile(profile)
    space = binding.resolution.vector_space
    return {
        "embedding_profile": binding.profile_id,
        "vector_space_id": space.id,
        "vector_space_schema": space.schema,
        "vector_space_backend": space.backend,
        "vector_space_dimension": space.dimension,
    }


def _patch_partition(monkeypatch, valid: List[str], invalid: List[str]) -> None:
    monkeypatch.setattr(
        ingestion,
        "partition_document_ids",
        lambda tenant, case, document_ids: (valid, invalid),
    )


def _patch_group(monkeypatch, async_result: DummyAsyncResult):
    captured_signatures: List[List[Any]] = []

    def fake_group(signatures_iterable: Iterable[Any]):
        items = list(signatures_iterable)
        captured_signatures.append(items)

        class _Group:
            def apply_async(self):
                return async_result

        return _Group()

    monkeypatch.setattr(ingestion, "group", fake_group)
    return captured_signatures


def _patch_perf_counter(monkeypatch, start: float, end: float) -> None:
    values = iter((start, end))

    def fake_perf_counter() -> float:
        try:
            return next(values)
        except StopIteration:
            return end

    monkeypatch.setattr(ingestion.time, "perf_counter", fake_perf_counter)


def _build_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    scope = ScopeContext(
        tenant_id=state["tenant_id"],
        trace_id=state["trace_id"],
        invocation_id="inv-1",
        run_id=state["run_id"],
        idempotency_key=state.get("idempotency_key"),
        tenant_schema=state.get("tenant_schema"),
        service_id="ingestion-task",
    )
    business = BusinessContext(case_id=state.get("case_id"))
    tool_context = ToolContext(scope=scope, business=business)
    return {
        "scope_context": scope.model_dump(mode="json", exclude_none=True),
        "business_context": business.model_dump(mode="json", exclude_none=True),
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
    }


@pytest.mark.django_db
def test_run_ingestion_success(monkeypatch):
    (
        dummy_process,
        start_calls,
        end_calls,
        apply_async_calls,
        schema_calls,
    ) = _setup_common_monkeypatches(monkeypatch)
    valid_ids = ["doc-1", "doc-2"]
    invalid_ids = ["missing-1"]
    _patch_partition(monkeypatch, valid_ids, invalid_ids)

    async_result = DummyAsyncResult(results=[])
    async_result._results_payload = [
        {
            "document_id": "doc-1",
            "inserted": 2,
            "replaced": 0,
            "skipped": 0,
            "chunk_count": 5,
            "duration_ms": 100.0,
        },
        {
            "document_id": "doc-2",
            "inserted": 0,
            "replaced": 1,
            "skipped": 1,
            "chunk_count": 3,
            "duration_ms": 150.0,
        },
    ]

    captured_signatures = _patch_group(monkeypatch, async_result)
    _patch_perf_counter(monkeypatch, 10.0, 10.5)

    state = {
        "tenant_id": "tenant-a",
        "case_id": "case-b",
        "document_ids": list(valid_ids),
        "embedding_profile": "standard",
        "run_id": "run-123",
        "trace_id": "trace-xyz",
        "idempotency_key": "idem-1",
    }
    meta = _build_meta(state)
    response = ingestion.run_ingestion.run(
        state,
        meta,
        timeout_seconds=5.0,
    )

    assert response["status"] == "dispatched"
    assert response["count"] == len(valid_ids)
    assert response["invalid_ids"] == invalid_ids
    assert response["inserted"] == 2
    assert response["replaced"] == 1
    assert response["skipped"] == 1
    assert response["total_chunks"] == 8
    assert response["duration_ms"] == pytest.approx(500.0)

    assert dummy_process.calls == [
        (
            {
                "tenant_id": "tenant-a",
                "case_id": "case-b",
                "document_id": "doc-1",
                "embedding_profile": "standard",
                "tenant_schema": None,
                "trace_id": "trace-xyz",
            },
            meta,
        ),
        (
            {
                "tenant_id": "tenant-a",
                "case_id": "case-b",
                "document_id": "doc-2",
                "embedding_profile": "standard",
                "tenant_schema": None,
                "trace_id": "trace-xyz",
            },
            meta,
        ),
    ]
    assert len(captured_signatures) == 1
    assert len(captured_signatures[0]) == len(valid_ids)

    assert start_calls == [
        {
            "tenant": "tenant-a",
            "case": "case-b",
            "run_id": "run-123",
            "doc_count": len(valid_ids),
            "trace_id": "trace-xyz",
            "idempotency_key": "idem-1",
            "embedding_profile": "standard",
            "vector_space_id": "rag/standard@v1",
            "case_status": None,
            "case_phase": None,
        }
    ]
    assert len(end_calls) == 1
    end_call = end_calls[0]
    assert end_call["tenant"] == "tenant-a"
    assert end_call["case"] == "case-b"
    assert end_call["run_id"] == "run-123"
    assert end_call["doc_count"] == len(valid_ids)
    assert end_call["inserted"] == 2
    assert end_call["replaced"] == 1
    assert end_call["skipped"] == 1
    assert end_call["total_chunks"] == 8
    assert end_call["duration_ms"] == pytest.approx(500.0)
    assert end_call["trace_id"] == "trace-xyz"
    assert end_call["idempotency_key"] == "idem-1"
    assert end_call["embedding_profile"] == "standard"
    assert end_call["vector_space_id"] == "rag/standard@v1"
    assert end_call["case_status"] is None
    assert end_call["case_phase"] is None
    assert apply_async_calls == []
    assert schema_calls == [
        {
            "id": "rag/standard@v1",
            "schema": "rag",
            "backend": "pgvector",
            "dimension": 1536,
        }
    ]


@pytest.mark.django_db
def test_run_ingestion_timeout_dispatches_dead_letters(monkeypatch):
    (
        dummy_process,
        start_calls,
        end_calls,
        apply_async_calls,
        schema_calls,
    ) = _setup_common_monkeypatches(monkeypatch)
    valid_ids = ["doc-1", "doc-2"]
    _patch_partition(monkeypatch, valid_ids, [])

    child_success = DummyChildResult(
        "doc-1",
        ready=True,
        partial={
            "document_id": "doc-1",
            "inserted": 1,
            "replaced": 0,
            "skipped": 0,
            "chunk_count": 2,
        },
    )
    child_pending = DummyChildResult(
        "doc-2",
        ready=False,
        partial=None,
        get_exception=CeleryTimeoutError("pending"),
    )
    async_result = DummyAsyncResult(results=[child_success, child_pending])
    async_result.should_raise = CeleryTimeoutError("timed out")

    _patch_group(monkeypatch, async_result)
    _patch_perf_counter(monkeypatch, 20.0, 20.4)

    original_collect = ingestion._collect_partial_results
    collect_calls: List[Any] = []

    def fake_collect(async_result_param):
        collect_calls.append(async_result_param)
        return original_collect(async_result_param)

    monkeypatch.setattr(ingestion, "_collect_partial_results", fake_collect)

    state = {
        "tenant_id": "tenant-a",
        "case_id": "case-b",
        "document_ids": list(valid_ids),
        "embedding_profile": "standard",
        "run_id": "run-timeout",
        "trace_id": "trace-timeout",
        "idempotency_key": "timeout-id",
    }
    response = ingestion.run_ingestion.run(
        state,
        _build_meta(state),
        timeout_seconds=3.0,
        dead_letter_queue="dlq-test",
    )

    assert response["status"] == "failed"
    assert "timed out" in response["error"]
    assert response["inserted"] == 1
    assert response["replaced"] == 0
    assert response["skipped"] == 0
    assert response["total_chunks"] == 2
    assert response["duration_ms"] == pytest.approx(400.0)

    assert len(collect_calls) == 1
    assert collect_calls[0] is async_result

    assert len(apply_async_calls) == 1
    dead_letter_payload = apply_async_calls[0]["kwargs"]["args"][0]
    assert dead_letter_payload["document_id"] == "doc-2"
    assert dead_letter_payload["run_id"] == "run-timeout"
    assert dead_letter_payload["trace_id"] == "trace-timeout"
    assert dead_letter_payload["process"] is None
    assert dead_letter_payload["workflow_id"] is None
    assert "doc_class" not in dead_letter_payload
    expected_resolution = _expected_ingestion_resolution()
    for key, value in expected_resolution.items():
        assert dead_letter_payload[key] == value

    assert child_pending.revoked == [{"terminate": True}]
    assert child_success.revoked == []

    assert start_calls[0]["doc_count"] == len(valid_ids)
    assert len(end_calls) == 1
    assert end_calls[0]["inserted"] == 1
    assert end_calls[0]["replaced"] == 0
    assert end_calls[0]["skipped"] == 0
    assert end_calls[0]["total_chunks"] == 2
    assert end_calls[0]["duration_ms"] == pytest.approx(400.0)
    assert schema_calls == [
        {
            "id": "rag/standard@v1",
            "schema": "rag",
            "backend": "pgvector",
            "dimension": 1536,
        }
    ]


@pytest.mark.django_db
def test_run_ingestion_base_exception_dispatches_dead_letters(monkeypatch):
    (
        dummy_process,
        start_calls,
        end_calls,
        apply_async_calls,
        schema_calls,
    ) = _setup_common_monkeypatches(monkeypatch)
    valid_ids = ["doc-1", "doc-2"]
    _patch_partition(monkeypatch, valid_ids, [])

    child_success = DummyChildResult(
        "doc-1",
        ready=True,
        partial={
            "document_id": "doc-1",
            "inserted": 1,
            "replaced": 0,
            "skipped": 0,
            "chunk_count": 2,
        },
    )
    child_pending = DummyChildResult(
        "doc-2",
        ready=False,
        partial=None,
        get_exception=CeleryTimeoutError("pending"),
    )
    async_result = DummyAsyncResult(results=[child_success, child_pending])
    async_result.should_raise = CeleryTimeoutError("timed out")

    _patch_group(monkeypatch, async_result)
    _patch_perf_counter(monkeypatch, 30.0, 30.6)

    original_determine = ingestion._determine_failed_documents
    call_counter = {"count": 0}

    def flaky_determine(document_ids: Iterable[str], results: Iterable[Dict[str, Any]]):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            raise RuntimeError("aggregation failed")
        return original_determine(document_ids, results)

    monkeypatch.setattr(ingestion, "_determine_failed_documents", flaky_determine)

    with pytest.raises(RuntimeError, match="aggregation failed"):
        state = {
            "tenant_id": "tenant-a",
            "case_id": "case-b",
            "document_ids": list(valid_ids),
            "embedding_profile": "standard",
            "run_id": "run-exc",
            "trace_id": "trace-exc",
            "idempotency_key": "exc-id",
        }
        ingestion.run_ingestion.run(
            state,
            _build_meta(state),
            timeout_seconds=2.5,
            dead_letter_queue="dlq-exc",
        )

    assert call_counter["count"] >= 2
    assert len(apply_async_calls) == 1
    dead_letter_payload = apply_async_calls[0]["kwargs"]["args"][0]
    assert dead_letter_payload["document_id"] == "doc-2"
    assert dead_letter_payload["run_id"] == "run-exc"
    assert dead_letter_payload["trace_id"] == "trace-exc"
    assert dead_letter_payload["process"] is None
    assert dead_letter_payload["workflow_id"] is None
    assert "doc_class" not in dead_letter_payload
    expected_resolution = _expected_ingestion_resolution()
    for key, value in expected_resolution.items():
        assert dead_letter_payload[key] == value

    assert child_pending.revoked == [{"terminate": True}]
    assert child_success.revoked == []

    assert start_calls[0]["doc_count"] == len(valid_ids)
    assert len(end_calls) == 1
    assert end_calls[0]["duration_ms"] == pytest.approx(600.0)
    assert schema_calls == [
        {
            "id": "rag/standard@v1",
            "schema": "rag",
            "backend": "pgvector",
            "dimension": 1536,
        }
    ]


@pytest.mark.django_db
def test_run_ingestion_contract_error_includes_context(monkeypatch):
    (
        dummy_process,
        start_calls,
        end_calls,
        apply_async_calls,
        schema_calls,
    ) = _setup_common_monkeypatches(monkeypatch)
    valid_ids = ["doc-1"]
    _patch_partition(monkeypatch, valid_ids, [])

    async_result = DummyAsyncResult(results=[])
    async_result.should_raise = InputError(
        IngestionContractErrorCode.VECTOR_DIMENSION_MISMATCH,
        "dimension mismatch",
        context={
            "process": "review",
            "workflow_id": "flow-legal",
            "expected_dimension": 2,
            "observed_dimension": 1,
            "chunk_index": 0,
        },
    )

    _patch_group(monkeypatch, async_result)
    _patch_perf_counter(monkeypatch, 40.0, 40.4)

    state = {
        "tenant_id": "tenant-a",
        "case_id": "case-b",
        "document_ids": list(valid_ids),
        "embedding_profile": "standard",
        "run_id": "run-contract",
        "trace_id": "trace-contract",
    }
    response = ingestion.run_ingestion.run(
        state,
        _build_meta(state),
        dead_letter_queue="dlq-contract",
    )

    assert response["status"] == "failed"
    assert "dimension mismatch" in response["error"]
    assert len(apply_async_calls) == 1
    dead_letter_payload = apply_async_calls[0]["kwargs"]["args"][0]
    assert dead_letter_payload["process"] == "review"
    assert dead_letter_payload["workflow_id"] == "flow-legal"
    assert "doc_class" not in dead_letter_payload
    assert dead_letter_payload["expected_dimension"] == 2
    assert dead_letter_payload["observed_dimension"] == 1
    assert dead_letter_payload["chunk_index"] == 0

    expected_resolution = _expected_ingestion_resolution()
    for key, value in expected_resolution.items():
        assert dead_letter_payload[key] == value

    assert start_calls[0]["doc_count"] == len(valid_ids)
    assert len(end_calls) == 1
    assert end_calls[0]["inserted"] == 0
    assert end_calls[0]["replaced"] == 0
    assert end_calls[0]["skipped"] == 0
    assert end_calls[0]["total_chunks"] == 0
    assert end_calls[0]["duration_ms"] == pytest.approx(400.0)
    assert schema_calls == [
        {
            "id": "rag/standard@v1",
            "schema": "rag",
            "backend": "pgvector",
            "dimension": 1536,
        }
    ]
