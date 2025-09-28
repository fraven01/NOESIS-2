from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from celery import shared_task
from common.celery import ScopedTask
from common.logging import get_logger
from .infra import object_store, pii
from django.utils import timezone
from .rag.schemas import Chunk
from .rag.vector_client import EMBEDDING_DIM
from .rag.vector_store import get_default_router

logger = get_logger(__name__)


def _build_path(meta: Dict[str, str], *parts: str) -> str:
    tenant = object_store.sanitize_identifier(meta["tenant"])
    case = object_store.sanitize_identifier(meta["case"])
    return "/".join([tenant, case, *parts])


@shared_task(base=ScopedTask)
def ingest_raw(meta: Dict[str, str], name: str, data: bytes) -> Dict[str, str]:
    """Persist raw document bytes."""
    path = _build_path(meta, "raw", name)
    object_store.put_bytes(path, data)
    return {"path": path}


@shared_task(base=ScopedTask)
def extract_text(meta: Dict[str, str], raw_path: str) -> Dict[str, str]:
    """Decode bytes to text and store."""
    full = object_store.BASE_PATH / raw_path
    text = full.read_bytes().decode("utf-8")
    out_path = _build_path(meta, "text", f"{Path(raw_path).stem}.txt")
    object_store.put_bytes(out_path, text.encode("utf-8"))
    return {"path": out_path}


@shared_task(base=ScopedTask)
def pii_mask(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
    """Mask PII in text."""
    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    masked = pii.mask(text)
    out_path = _build_path(meta, "text", f"{Path(text_path).stem}.masked.txt")
    object_store.put_bytes(out_path, masked.encode("utf-8"))
    return {"path": out_path}


@shared_task(base=ScopedTask)
def chunk(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
    """Split text into chunks; stubbed as a single chunk."""
    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    hash_val = hashlib.sha1(text.encode("utf-8")).hexdigest()
    chunks: List[Dict[str, object]] = [
        {
            "content": text,
            "meta": {
                "tenant": meta["tenant"],
                "case": meta["case"],
                "source": text_path,
                "hash": hash_val,
            },
        }
    ]
    out_path = _build_path(meta, "embeddings", "chunks.json")
    object_store.write_json(out_path, chunks)
    return {"path": out_path}


@shared_task(base=ScopedTask)
def embed(meta: Dict[str, str], chunks_path: str) -> Dict[str, str]:
    """Attach dummy embedding vectors to chunks."""
    chunks = object_store.read_json(chunks_path)
    embeddings = [{**ch, "embedding": [0.0] * EMBEDDING_DIM} for ch in chunks]
    out_path = _build_path(meta, "embeddings", "vectors.json")
    object_store.write_json(out_path, embeddings)
    return {"path": out_path}


@shared_task(base=ScopedTask)
def upsert(meta: Dict[str, str], embeddings_path: str) -> int:
    """Upsert embedded chunks into the vector client."""
    data = object_store.read_json(embeddings_path)
    chunk_objs = []
    for ch in data:
        vector = ch.get("embedding")
        embedding = [float(v) for v in vector] if vector is not None else None
        chunk_objs.append(
            Chunk(content=ch["content"], meta=ch["meta"], embedding=embedding)
        )

    tenant_id: Optional[str] = meta.get("tenant") if meta else None
    if not tenant_id:
        tenant_id = next(
            (
                str(chunk.meta.get("tenant"))
                for chunk in chunk_objs
                if chunk.meta and chunk.meta.get("tenant")
            ),
            None,
        )
    if not tenant_id:
        raise ValueError("tenant_id required for upsert")

    for chunk in chunk_objs:
        chunk_tenant = chunk.meta.get("tenant") if chunk.meta else None
        if chunk_tenant and str(chunk_tenant) != tenant_id:
            raise ValueError("chunk tenant mismatch")

    router = get_default_router()
    tenant_client = router
    for_tenant = getattr(router, "for_tenant", None)
    if callable(for_tenant):
        tenant_client = for_tenant(tenant_id)
    written = tenant_client.upsert_chunks(chunk_objs)
    return written


@shared_task(base=ScopedTask, queue="ingestion")
def ingestion_run(
    tenant_id: str,
    case_id: str,
    document_ids: List[str],
    priority: str = "normal",
    trace_id: str | None = None,
) -> Dict[str, object]:
    """Placeholder ingestion dispatcher used by the ingestion run endpoint."""

    # Keep relying on django.utils.timezone.now so call sites and tests can
    # monkeypatch the module-level helper consistently.
    queued_at = timezone.now().isoformat()
    logger.info(
        "Queued ingestion run",
        extra={
            "tenant": tenant_id,
            "case_id": case_id,
            "document_ids": document_ids,
            "priority": priority,
            "queued_at": queued_at,
            "trace_id": trace_id,
        },
    )
    return {"status": "queued", "queued_at": queued_at}
