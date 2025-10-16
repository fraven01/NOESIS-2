from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from uuid import UUID

from celery import group, shared_task
from celery.exceptions import TimeoutError as CeleryTimeoutError
from common.celery import ScopedTask
from common.logging import get_logger
from django.utils import timezone
from django.conf import settings

from . import tasks as pipe
from .infra import object_store
from .ingestion_utils import make_fallback_external_id
from .ingestion_status import (
    mark_ingestion_run_completed,
    mark_ingestion_run_running,
)
from ai_core.tools import InputError

from .rag.ingestion_contracts import resolve_ingestion_profile
from .rag.vector_schema import ensure_vector_space_schema
from .rag.selector_utils import normalise_selector_value

log = get_logger(__name__)


def _upload_dir(tenant: str, case: str) -> Path:
    return (
        object_store.BASE_PATH
        / object_store.sanitize_identifier(tenant)
        / object_store.sanitize_identifier(case)
        / "uploads"
    )


def _meta_store_path(tenant: str, case: str, document_id: str) -> str:
    return "/".join(
        (
            object_store.sanitize_identifier(tenant),
            object_store.sanitize_identifier(case),
            "uploads",
            f"{document_id}.meta.json",
        )
    )


def _resolve_upload(
    tenant: str, case: str, document_id: str
) -> tuple[Path, Dict[str, object]]:
    updir = _upload_dir(tenant, case)

    meta_path = updir / f"{document_id}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata for {document_id} not found")

    metadata_raw = json.loads(meta_path.read_text())
    if not isinstance(metadata_raw, dict):
        metadata = {}
    else:
        metadata = dict(metadata_raw)

    matches = list(updir.glob(f"{document_id}_*"))
    if not matches:
        raise FileNotFoundError(f"file for {document_id} not found")

    file_path = matches[0]
    external_id = metadata.get("external_id")
    if not (isinstance(external_id, str) and external_id.strip()):
        try:
            stat = file_path.stat()
            with file_path.open("rb") as handle:
                prefix = handle.read(64 * 1024)
        except FileNotFoundError:
            raise
        fallback = make_fallback_external_id(file_path.name, stat.st_size, prefix)
        metadata["external_id"] = fallback
        sanitized_metadata = dict(metadata)
        sanitized_metadata.pop("tenant", None)
        sanitized_metadata.pop("case", None)
        object_store.write_json(
            _meta_store_path(tenant, case, document_id), sanitized_metadata
        )
        metadata = sanitized_metadata

    metadata.pop("tenant", None)
    metadata.pop("case", None)

    return file_path, metadata


def _status_store_path(tenant: str, case: str, document_id: str) -> str:
    return "/".join(
        (
            object_store.sanitize_identifier(tenant),
            object_store.sanitize_identifier(case),
            "uploads",
            f"{document_id}.status.json",
        )
    )


def _load_pipeline_state(tenant: str, case: str, document_id: str) -> Dict[str, object]:
    status_path = _status_store_path(tenant, case, document_id)
    try:
        raw_state = object_store.read_json(status_path)
    except FileNotFoundError:
        return {
            "steps": {},
            "attempts": 0,
        }
    if not isinstance(raw_state, dict):
        return {"steps": {}, "attempts": 0}
    raw_state.setdefault("steps", {})
    if not isinstance(raw_state["steps"], dict):
        raw_state["steps"] = {}
    return raw_state


def _write_pipeline_state(
    tenant: str, case: str, document_id: str, state: Dict[str, object]
) -> None:
    status_path = _status_store_path(tenant, case, document_id)
    serialized: Dict[str, object] = {}
    for key, value in state.items():
        if key == "steps" and isinstance(value, dict):
            normalized_steps: Dict[str, object] = {}
            for step_name, step_data in value.items():
                if not isinstance(step_data, dict):
                    continue
                normalized: Dict[str, object] = {}
                for field, field_value in step_data.items():
                    if field == "path" and field_value:
                        normalized[field] = str(field_value)
                    elif isinstance(field_value, Path):
                        normalized[field] = str(field_value)
                    else:
                        normalized[field] = field_value
                normalized_steps[step_name] = normalized
            serialized[key] = normalized_steps
        else:
            serialized[key] = value
    object_store.write_json(status_path, serialized)


def _normalize_step_result(result: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in result.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def _ensure_step(
    tenant: str,
    case: str,
    document_id: str,
    state: Dict[str, object],
    step_name: str,
    run_step: Callable[[], Dict[str, object]],
) -> tuple[Dict[str, object], bool]:
    steps = state.setdefault("steps", {})
    cached = steps.get(step_name)
    if isinstance(cached, dict):
        cached_path = cached.get("path")
        if cached_path:
            if (object_store.BASE_PATH / str(cached_path)).exists():
                cached_result = {
                    key: cached[key]
                    for key in cached
                    if key not in {"completed_at", "cleaned"}
                }
                return cached_result, True
        elif "path" not in cached:
            cached_result = {
                key: cached[key]
                for key in cached
                if key not in {"completed_at", "cleaned"}
            }
            return cached_result, True

    result = _normalize_step_result(run_step())
    steps[step_name] = {
        **result,
        "completed_at": time.time(),
        "cleaned": False,
    }
    _write_pipeline_state(tenant, case, document_id, state)
    return result, False


def _cleanup_artifacts(paths: Iterable[Optional[str]]) -> List[str]:
    removed: List[str] = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        target = object_store.BASE_PATH / path
        try:
            if target.exists():
                target.unlink()
                removed.append(path)
        except Exception:
            log.exception("Failed to cleanup artifact", extra={"path": path})
    return removed


def _mark_cleaned(
    tenant: str,
    case: str,
    document_id: str,
    state: Dict[str, object],
    removed_paths: Iterable[str],
) -> None:
    removed_set = {str(path) for path in removed_paths}
    steps = state.setdefault("steps", {})
    updated = False
    for step_name, step_data in list(steps.items()):
        if not isinstance(step_data, dict):
            continue
        if step_data.get("path") in removed_set:
            step_data["path"] = None
            step_data["cleaned"] = True
            updated = True
    if updated:
        _write_pipeline_state(tenant, case, document_id, state)


def partition_document_ids(
    tenant: str, case: str, document_ids: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """Split document identifiers into existing and missing uploads."""

    existing: List[str] = []
    missing: List[str] = []

    for document_id in document_ids:
        try:
            _resolve_upload(tenant, case, document_id)
        except FileNotFoundError:
            missing.append(document_id)
        else:
            existing.append(document_id)

    return existing, missing


@shared_task(
    base=ScopedTask,
    queue="ingestion",
    bind=True,
    max_retries=3,
    accepts_scope=True,
)
def process_document(
    self,
    tenant: str,
    case: str,
    document_id: str,
    embedding_profile: str,
    tenant_schema: Optional[str] = None,
) -> Dict[str, object]:
    started = time.perf_counter()
    state = _load_pipeline_state(tenant, case, document_id)
    state["attempts"] = int(state.get("attempts", 0)) + 1
    state["last_attempt_started_at"] = time.time()
    _write_pipeline_state(tenant, case, document_id, state)
    current_step: Optional[str] = None
    created_artifacts: List[Tuple[str, str]] = []
    resolved_profile_id: Optional[str] = None
    vector_space_id: Optional[str] = None
    vector_space_schema: Optional[str] = None
    vector_space_backend: Optional[str] = None
    vector_space_dimension: Optional[int] = None
    try:
        current_step = "resolve_profile"
        profile_binding = resolve_ingestion_profile(embedding_profile)
        resolved_profile_id = profile_binding.profile_id
        vector_space = profile_binding.resolution.vector_space
        vector_space_id = vector_space.id
        vector_space_schema = vector_space.schema
        vector_space_backend = vector_space.backend
        vector_space_dimension = vector_space.dimension

        fpath, meta_json = _resolve_upload(tenant, case, document_id)
        file_bytes = fpath.read_bytes()
        external_id = meta_json.get("external_id")
        if not (isinstance(external_id, str) and external_id.strip()):
            external_id = make_fallback_external_id(
                fpath.name,
                len(file_bytes),
                file_bytes,
            )
            meta_json["external_id"] = external_id
            object_store.write_json(
                _meta_store_path(tenant, case, document_id), meta_json
            )

        meta_json["embedding_profile"] = resolved_profile_id
        if vector_space_id:
            meta_json["vector_space_id"] = vector_space_id
        sanitized_meta_json = dict(meta_json)
        sanitized_meta_json.pop("tenant", None)
        sanitized_meta_json.pop("case", None)
        raw_collection_id = meta_json.get("collection_id")
        normalized_collection_id: str | None = None
        if raw_collection_id is not None:
            try:
                normalized_collection_id = str(UUID(str(raw_collection_id).strip()))
            except (TypeError, ValueError, AttributeError):
                normalized_collection_id = None
        normalized_process = normalise_selector_value(meta_json.get("process"))
        normalized_doc_class = normalise_selector_value(meta_json.get("doc_class"))
        if normalized_process is not None:
            sanitized_meta_json["process"] = normalized_process
        elif "process" in sanitized_meta_json:
            sanitized_meta_json.pop("process", None)
        if normalized_doc_class is not None:
            sanitized_meta_json["doc_class"] = normalized_doc_class
        elif "doc_class" in sanitized_meta_json:
            sanitized_meta_json.pop("doc_class", None)
        if normalized_collection_id is not None:
            sanitized_meta_json["collection_id"] = normalized_collection_id
        elif "collection_id" in sanitized_meta_json:
            sanitized_meta_json.pop("collection_id", None)
        # Normalize meta keys to the new contract: tenant_id/case_id
        meta = {**sanitized_meta_json, "tenant_id": tenant, "case_id": case}
        if tenant_schema:
            meta["tenant_schema"] = tenant_schema
        if resolved_profile_id:
            meta["embedding_profile"] = resolved_profile_id
        if vector_space_id:
            meta["vector_space_id"] = vector_space_id
        if vector_space_schema:
            meta["vector_space_schema"] = vector_space_schema
        if vector_space_backend:
            meta["vector_space_backend"] = vector_space_backend
        if vector_space_dimension is not None:
            meta["vector_space_dimension"] = vector_space_dimension
        if normalized_process is not None:
            meta["process"] = normalized_process
        if normalized_doc_class is not None:
            meta["doc_class"] = normalized_doc_class
        if normalized_collection_id is not None:
            meta["collection_id"] = normalized_collection_id
        state["meta"] = {
            "external_id": meta.get("external_id"),
            "file": fpath.name,
            "embedding_profile": resolved_profile_id,
            "vector_space_id": vector_space_id,
        }
        if normalized_process is not None:
            state["meta"]["process"] = normalized_process
        if normalized_doc_class is not None:
            state["meta"]["doc_class"] = normalized_doc_class
        if normalized_collection_id is not None:
            state["meta"]["collection_id"] = normalized_collection_id
        object_store.write_json(
            _meta_store_path(tenant, case, document_id), sanitized_meta_json
        )
        _write_pipeline_state(tenant, case, document_id, state)

        current_step = "ingest_raw"
        raw, reused = _ensure_step(
            tenant,
            case,
            document_id,
            state,
            current_step,
            lambda: pipe.ingest_raw(meta, fpath.name, file_bytes),
        )
        if not reused and raw.get("path"):
            created_artifacts.append((current_step, str(raw["path"])))
        if "content_hash" in raw:
            meta["content_hash"] = raw["content_hash"]
            state.setdefault("meta", {})["content_hash"] = raw["content_hash"]
            _write_pipeline_state(tenant, case, document_id, state)

        current_step = "extract_text"
        text, reused = _ensure_step(
            tenant,
            case,
            document_id,
            state,
            current_step,
            lambda: pipe.extract_text(meta, raw["path"]),
        )
        if not reused and text.get("path"):
            created_artifacts.append((current_step, str(text["path"])))

        current_step = "pii_mask"
        if not getattr(settings, "INGESTION_PII_MASK_ENABLED", True):
            masked = {"path": text["path"]}
            reused = True
            # Mark step as skipped in state for traceability
            state.setdefault("steps", {})[current_step] = {
                "path": text.get("path"),
                "skipped": True,
                "completed_at": time.time(),
                "cleaned": True,
            }
            _write_pipeline_state(tenant, case, document_id, state)
        else:
            masked, reused = _ensure_step(
                tenant,
                case,
                document_id,
                state,
                current_step,
                lambda: pipe.pii_mask(meta, text["path"]),
            )
            if not reused and masked.get("path"):
                created_artifacts.append((current_step, str(masked["path"])))

        current_step = "chunk"
        chunks, reused = _ensure_step(
            tenant,
            case,
            document_id,
            state,
            current_step,
            lambda: pipe.chunk(meta, masked["path"]),
        )
        if not reused and chunks.get("path"):
            created_artifacts.append((current_step, str(chunks["path"])))

        current_step = "embed"
        emb, reused = _ensure_step(
            tenant,
            case,
            document_id,
            state,
            current_step,
            lambda: pipe.embed(meta, chunks["path"]),
        )
        if not reused and emb.get("path"):
            created_artifacts.append((current_step, str(emb["path"])))

        current_step = "upsert"
        upsert_result = pipe.upsert(meta, emb["path"], tenant_schema=tenant_schema)
        state.setdefault("steps", {})["upsert"] = {
            "completed_at": time.time(),
            "cleaned": True,
        }
        state["last_error"] = None
        state["completed_at"] = time.time()
        _write_pipeline_state(tenant, case, document_id, state)
    except InputError as exc:
        state["last_error"] = {
            "step": current_step,
            "message": str(exc),
            "retry": getattr(self.request, "retries", 0),
            "failed_at": time.time(),
        }
        _write_pipeline_state(tenant, case, document_id, state)
        raise
    except Exception as exc:  # pragma: no cover - defensive retry path
        retries = getattr(self.request, "retries", 0)
        countdown = min(300, 5 * (2**retries or 1))
        state["last_error"] = {
            "step": current_step,
            "message": str(exc),
            "retry": retries,
            "failed_at": time.time(),
        }
        _write_pipeline_state(tenant, case, document_id, state)
        cleanup_targets = [
            path for step, path in created_artifacts if step == current_step
        ]
        removed = _cleanup_artifacts(cleanup_targets)
        if removed:
            _mark_cleaned(tenant, case, document_id, state, removed)
        log.warning(
            "Retrying ingestion document task after failure",
            extra={
                "tenant": tenant,
                "case": case,
                "document_id": document_id,
                "retries": retries,
            },
        )
        raise self.retry(exc=exc, countdown=countdown)

    all_paths: List[str] = []
    for step_data in state.get("steps", {}).values():
        if isinstance(step_data, dict) and step_data.get("path"):
            all_paths.append(str(step_data["path"]))
    removed_after_success = _cleanup_artifacts(
        [path for _, path in created_artifacts] + all_paths
    )
    if removed_after_success:
        _mark_cleaned(tenant, case, document_id, state, removed_after_success)

    written = int(upsert_result)
    documents = getattr(upsert_result, "documents", [])
    if documents:
        inserted_count = sum(
            1 for info in documents if info.get("action") == "inserted"
        )
        replaced_count = sum(
            1 for info in documents if info.get("action") == "replaced"
        )
        skipped_count = sum(1 for info in documents if info.get("action") == "skipped")
        chunk_count = sum(int(info.get("chunk_count", 0)) for info in documents)
        if len(documents) == 1:
            action = str(documents[0].get("action", "unknown"))
        elif inserted_count and not (replaced_count or skipped_count):
            action = "inserted"
        elif replaced_count and not (inserted_count or skipped_count):
            action = "replaced"
        elif skipped_count and not (inserted_count or replaced_count):
            action = "skipped"
        else:
            action = "mixed"
    else:
        inserted_count = 1 if written else 0
        replaced_count = 0
        skipped_count = 1 if written == 0 else 0
        chunk_count = written
        action = "skipped" if written == 0 else "inserted"
    duration_ms = (time.perf_counter() - started) * 1000

    log.info(
        "Ingested document",
        extra={
            "tenant": tenant,
            "case": case,
            "document_id": document_id,
            "file": fpath.name,
            "written_chunks": written,
            "action": action,
            "chunk_count": chunk_count,
            "duration_ms": duration_ms,
            "embedding_profile": resolved_profile_id,
            "vector_space_id": vector_space_id,
        },
    )
    return {
        "document_id": document_id,
        "written": written,
        "action": action,
        "chunk_count": chunk_count,
        "duration_ms": duration_ms,
        "external_id": meta.get("external_id"),
        "content_hash": meta.get("content_hash"),
        "inserted": inserted_count,
        "replaced": replaced_count,
        "skipped": skipped_count,
        "embedding_profile": resolved_profile_id,
        "vector_space_id": vector_space_id,
    }


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
def run_ingestion(
    tenant: str,
    case: str,
    document_ids: List[str],
    embedding_profile: str,
    *,
    run_id: str,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    tenant_schema: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    dead_letter_queue: Optional[str] = None,
    session_salt: Optional[str] = None,  # noqa: ARG001 - propagated by ScopedTask
    session_scope: Optional[Tuple[str, str, str]] = None,  # noqa: ARG001 - unused hook
) -> Dict[str, object]:
    valid_ids, invalid_ids = partition_document_ids(tenant, case, document_ids)
    dispatch_ids = list(valid_ids if valid_ids else document_ids)
    doc_count = len(valid_ids)
    try:
        binding = resolve_ingestion_profile(embedding_profile)
    except InputError as exc:
        log.error(
            "Failed to resolve ingestion profile",
            extra={
                "tenant": tenant,
                "case": case,
                "run_id": run_id,
                "trace_id": trace_id,
                "embedding_profile": embedding_profile,
                "error": str(exc),
            },
        )
        raise

    resolved_profile_id = binding.profile_id
    resolved_space = binding.resolution.vector_space
    vector_space_id = resolved_space.id
    vector_space_schema = resolved_space.schema
    vector_space_backend = resolved_space.backend
    vector_space_dimension = resolved_space.dimension

    schema_initialised = ensure_vector_space_schema(resolved_space)
    if schema_initialised:
        log.info(
            "Initialised vector schema for ingestion run",
            extra={
                "tenant": tenant,
                "case": case,
                "run_id": run_id,
                "embedding_profile": resolved_profile_id,
                "vector_space_id": vector_space_id,
                "vector_space_schema": vector_space_schema,
            },
        )

    pipe.log_ingestion_run_start(
        tenant=tenant,
        case=case,
        run_id=run_id,
        doc_count=doc_count,
        trace_id=trace_id,
        idempotency_key=idempotency_key,
        embedding_profile=resolved_profile_id,
        vector_space_id=vector_space_id,
    )
    mark_ingestion_run_running(
        tenant,
        case,
        run_id,
        started_at=timezone.now().isoformat(),
        document_ids=dispatch_ids,
    )
    started = time.perf_counter()

    async_result = None
    failure: Optional[BaseException] = None
    results: List[Dict[str, object]] = []
    failed_ids: List[str] = []
    duration_ms = 0.0
    inserted = replaced = skipped = total_chunks = 0

    try:
        if valid_ids:
            job_group = group(
                process_document.s(
                    tenant,
                    case,
                    doc_id,
                    resolved_profile_id,
                    tenant_schema,
                )
                for doc_id in valid_ids
            )
            try:
                async_result = job_group.apply_async()
            except Exception as exc:  # pragma: no cover - defensive path
                failure = exc
                log.exception(
                    "Failed to dispatch ingestion group",
                    extra={
                        "tenant": tenant,
                        "case": case,
                        "run_id": run_id,
                        "trace_id": trace_id,
                    },
                )
            else:
                try:
                    results = async_result.get(
                        timeout=timeout_seconds, disable_sync_subtasks=False
                    )
                except CeleryTimeoutError as exc:
                    failure = exc
                    log.error(
                        "Timed out waiting for ingestion group",
                        extra={
                            "tenant": tenant,
                            "case": case,
                            "run_id": run_id,
                            "trace_id": trace_id,
                            "timeout_seconds": timeout_seconds,
                        },
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    failure = exc
                    log.exception(
                        "Ingestion group failed",
                        extra={
                            "tenant": tenant,
                            "case": case,
                            "run_id": run_id,
                            "trace_id": trace_id,
                        },
                    )

        if failure is not None:
            if async_result:
                results = _collect_partial_results(async_result)
                failed_ids = _determine_failed_documents(valid_ids, results)
                _revoke_pending(async_result)
            elif valid_ids:
                failed_ids = list(valid_ids)
            _safe_dispatch_dead_letters(
                dead_letter_queue,
                tenant,
                case,
                run_id,
                trace_id,
                failed_ids,
                failure,
                embedding_profile=resolved_profile_id,
                vector_space_id=vector_space_id,
                vector_space_schema=vector_space_schema,
                vector_space_backend=vector_space_backend,
                vector_space_dimension=vector_space_dimension,
            )

    except BaseException as exc:
        if failure is None:
            failure = exc
        if async_result and not failed_ids:
            results = _collect_partial_results(async_result)
            failed_ids = _determine_failed_documents(valid_ids, results)
            _revoke_pending(async_result)
            _safe_dispatch_dead_letters(
                dead_letter_queue,
                tenant,
                case,
                run_id,
                trace_id,
                failed_ids,
                failure,
                embedding_profile=resolved_profile_id,
                vector_space_id=vector_space_id,
                vector_space_schema=vector_space_schema,
                vector_space_backend=vector_space_backend,
                vector_space_dimension=vector_space_dimension,
            )
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        inserted = sum(int(result.get("inserted", 0)) for result in results)
        replaced = sum(int(result.get("replaced", 0)) for result in results)
        skipped = sum(int(result.get("skipped", 0)) for result in results)
        total_chunks = sum(
            int(result.get("chunk_count", result.get("written", 0)))
            for result in results
        )

        pipe.log_ingestion_run_end(
            tenant=tenant,
            case=case,
            run_id=run_id,
            doc_count=doc_count,
            inserted=inserted,
            replaced=replaced,
            skipped=skipped,
            total_chunks=total_chunks,
            duration_ms=duration_ms,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
            embedding_profile=resolved_profile_id,
            vector_space_id=vector_space_id,
        )

    log.info(
        "Dispatched ingestion group",
        extra={
            "tenant": tenant,
            "case": case,
            "trace_id": trace_id,
            "count": doc_count,
            "invalid_ids": invalid_ids,
            "embedding_profile": resolved_profile_id,
            "vector_space_id": vector_space_id,
            "failed_ids": (
                failed_ids
                if failed_ids
                else (
                    _determine_failed_documents(valid_ids, results) if valid_ids else []
                )
            ),
        },
    )
    finished_at = timezone.now().isoformat()

    response: Dict[str, object] = {
        "status": "dispatched",
        "count": doc_count,
        "invalid_ids": invalid_ids,
        "inserted": inserted,
        "replaced": replaced,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "duration_ms": duration_ms,
    }
    if failure is not None:
        response["status"] = "failed"
        response["error"] = str(failure)

    mark_ingestion_run_completed(
        tenant,
        case,
        run_id,
        finished_at=finished_at,
        duration_ms=duration_ms,
        inserted_documents=inserted,
        replaced_documents=replaced,
        skipped_documents=skipped,
        inserted_chunks=total_chunks,
        invalid_document_ids=invalid_ids,
        document_ids=dispatch_ids,
        error=response.get("error"),
    )

    return response


def _collect_partial_results(async_result) -> List[Dict[str, object]]:
    collected: List[Dict[str, object]] = []
    for result in getattr(async_result, "results", []) or []:
        try:
            partial = result.get(timeout=0, propagate=False)
        except CeleryTimeoutError:
            continue
        except Exception:
            continue
        if isinstance(partial, dict):
            collected.append(partial)
    return collected


def _revoke_pending(async_result) -> None:
    for result in getattr(async_result, "results", []) or []:
        if not result.ready():
            try:
                result.revoke(terminate=True)
            except Exception:
                continue


def _determine_failed_documents(
    document_ids: Iterable[str], results: Iterable[Dict[str, object]]
) -> List[str]:
    processed_ids = {
        str(result.get("document_id"))
        for result in results
        if isinstance(result, dict) and result.get("document_id")
    }
    return [doc_id for doc_id in document_ids if doc_id not in processed_ids]


def _dispatch_dead_letters(
    dead_letter_queue: Optional[str],
    tenant: str,
    case: str,
    run_id: str,
    trace_id: Optional[str],
    failed_ids: Iterable[str],
    failure: BaseException,
    *,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
    vector_space_schema: Optional[str] = None,
    vector_space_backend: Optional[str] = None,
    vector_space_dimension: Optional[int] = None,
) -> None:
    failed_list = list(failed_ids)
    if not failed_list:
        return
    payload = {
        "tenant": tenant,
        "case": case,
        "run_id": run_id,
        "trace_id": trace_id,
        "error": str(failure),
    }
    if embedding_profile:
        payload["embedding_profile"] = embedding_profile
    if vector_space_id:
        payload["vector_space_id"] = vector_space_id
    if vector_space_schema:
        payload["vector_space_schema"] = vector_space_schema
    if vector_space_backend:
        payload["vector_space_backend"] = vector_space_backend
    if vector_space_dimension is not None:
        payload["vector_space_dimension"] = vector_space_dimension
    context = getattr(failure, "context", None)
    if isinstance(context, dict):
        for key, value in context.items():
            if key in payload and key not in {"process", "doc_class"}:
                continue
            if value is not None or key in {"process", "doc_class"}:
                payload[key] = value
    payload.setdefault("process", None)
    payload.setdefault("doc_class", None)

    for document_id in failed_list:
        message = {**payload, "document_id": document_id}
        record_dead_letter.apply_async(
            args=[message],
            queue=dead_letter_queue if dead_letter_queue else "ingestion_dead_letter",
        )


def _safe_dispatch_dead_letters(
    dead_letter_queue: Optional[str],
    tenant: str,
    case: str,
    run_id: str,
    trace_id: Optional[str],
    failed_ids: Iterable[str],
    failure: BaseException,
    *,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
    vector_space_schema: Optional[str] = None,
    vector_space_backend: Optional[str] = None,
    vector_space_dimension: Optional[int] = None,
) -> None:
    try:
        _dispatch_dead_letters(
            dead_letter_queue,
            tenant,
            case,
            run_id,
            trace_id,
            failed_ids,
            failure,
            embedding_profile=embedding_profile,
            vector_space_id=vector_space_id,
            vector_space_schema=vector_space_schema,
            vector_space_backend=vector_space_backend,
            vector_space_dimension=vector_space_dimension,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.exception(
            "Failed to dispatch ingestion dead letters",
            extra={
                "tenant": tenant,
                "case": case,
                "run_id": run_id,
                "trace_id": trace_id,
                "failed_ids": list(failed_ids),
                "embedding_profile": embedding_profile,
                "vector_space_id": vector_space_id,
            },
        )


@shared_task(
    base=ScopedTask,
    queue="ingestion_dead_letter",
    name="ai_core.ingestion.dead_letter",
    accepts_scope=True,
)
def record_dead_letter(payload: Dict[str, object]) -> None:
    log.error("Ingestion dead letter", extra=payload)
