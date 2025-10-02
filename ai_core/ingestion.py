from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from celery import group, shared_task
from celery.exceptions import TimeoutError as CeleryTimeoutError
from common.celery import ScopedTask
from common.logging import get_logger

from . import tasks as pipe
from .infra import object_store
from .ingestion_utils import make_fallback_external_id

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


@shared_task(base=ScopedTask, queue="ingestion", bind=True, max_retries=3)
def process_document(
    self, tenant: str, case: str, document_id: str
) -> Dict[str, object]:
    started = time.perf_counter()
    try:
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

        sanitized_meta_json = dict(meta_json)
        sanitized_meta_json.pop("tenant", None)
        sanitized_meta_json.pop("case", None)
        meta = {**sanitized_meta_json, "tenant": tenant, "case": case}

        raw = pipe.ingest_raw(meta, fpath.name, file_bytes)
        text = pipe.extract_text(meta, raw["path"])
        masked = pipe.pii_mask(meta, text["path"])
        chunks = pipe.chunk(meta, masked["path"])
        emb = pipe.embed(meta, chunks["path"])
        upsert_result = pipe.upsert(meta, emb["path"])
    except Exception as exc:  # pragma: no cover - defensive retry path
        retries = getattr(self.request, "retries", 0)
        countdown = min(300, 5 * (2**retries or 1))
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
    }


@shared_task(base=ScopedTask, queue="ingestion")
def run_ingestion(
    tenant: str,
    case: str,
    document_ids: List[str],
    *,
    run_id: str,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    dead_letter_queue: Optional[str] = None,
) -> Dict[str, object]:
    valid_ids, invalid_ids = partition_document_ids(tenant, case, document_ids)
    doc_count = len(valid_ids)
    pipe.log_ingestion_run_start(
        tenant=tenant,
        case=case,
        run_id=run_id,
        doc_count=doc_count,
        trace_id=trace_id,
        idempotency_key=idempotency_key,
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
                process_document.s(tenant, case, doc_id) for doc_id in valid_ids
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
        )

    log.info(
        "Dispatched ingestion group",
        extra={
            "tenant": tenant,
            "case": case,
            "trace_id": trace_id,
            "count": doc_count,
            "invalid_ids": invalid_ids,
            "failed_ids": failed_ids
            if failed_ids
            else _determine_failed_documents(valid_ids, results)
            if valid_ids
            else [],
        },
    )
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
            },
        )


@shared_task(
    base=ScopedTask,
    queue="ingestion_dead_letter",
    name="ai_core.ingestion.dead_letter",
)
def record_dead_letter(payload: Dict[str, object]) -> None:
    log.error("Ingestion dead letter", extra=payload)
