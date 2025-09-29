from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from celery import group, shared_task
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


@shared_task(base=ScopedTask, queue="ingestion")
def process_document(tenant: str, case: str, document_id: str) -> Dict[str, object]:
    started = time.perf_counter()
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
        object_store.write_json(_meta_store_path(tenant, case, document_id), meta_json)

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

    results: List[Dict[str, object]] = []
    if valid_ids:
        job_group = group(
            process_document.s(tenant, case, doc_id) for doc_id in valid_ids
        )
        async_result = job_group.apply_async()
        results = async_result.get(disable_sync_subtasks=False)

    duration_ms = (time.perf_counter() - started) * 1000
    inserted = sum(int(result.get("inserted", 0)) for result in results)
    replaced = sum(int(result.get("replaced", 0)) for result in results)
    skipped = sum(int(result.get("skipped", 0)) for result in results)
    total_chunks = sum(
        int(result.get("chunk_count", result.get("written", 0))) for result in results
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
        },
    )
    return {
        "status": "dispatched",
        "count": doc_count,
        "invalid_ids": invalid_ids,
        "inserted": inserted,
        "replaced": replaced,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "duration_ms": duration_ms,
    }
