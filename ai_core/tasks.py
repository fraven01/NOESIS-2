from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from celery import shared_task
from common.celery import ScopedTask
from common.logging import get_logger
from django.conf import settings
from django.utils import timezone

from .infra import object_store, pii
from .rag import metrics
from .rag.schemas import Chunk
from .rag.normalization import normalise_text
from .rag.vector_client import EMBEDDING_DIM
from .rag.vector_store import get_default_router

logger = get_logger(__name__)


def _build_path(meta: Dict[str, str], *parts: str) -> str:
    tenant = object_store.sanitize_identifier(meta["tenant"])
    case = object_store.sanitize_identifier(meta["case"])
    return "/".join([tenant, case, *parts])


def log_ingestion_run_start(
    *,
    tenant: str,
    case: str,
    run_id: str,
    doc_count: int,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> None:
    extra = {
        "tenant": tenant,
        "case": case,
        "run_id": run_id,
        "doc_count": doc_count,
    }
    if trace_id:
        extra["trace_id"] = trace_id
    if idempotency_key:
        extra["idempotency_key"] = idempotency_key
    logger.info("ingestion.start", extra=extra)


def log_ingestion_run_end(
    *,
    tenant: str,
    case: str,
    run_id: str,
    doc_count: int,
    inserted: int,
    replaced: int,
    skipped: int,
    total_chunks: int,
    duration_ms: float,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> None:
    extra = {
        "tenant": tenant,
        "case": case,
        "run_id": run_id,
        "doc_count": doc_count,
        "inserted": inserted,
        "replaced": replaced,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "duration_ms": duration_ms,
    }
    if trace_id:
        extra["trace_id"] = trace_id
    if idempotency_key:
        extra["idempotency_key"] = idempotency_key
    logger.info("ingestion.end", extra=extra)
    metrics.INGESTION_RUN_MS.observe(float(duration_ms))


@shared_task(base=ScopedTask)
def ingest_raw(meta: Dict[str, str], name: str, data: bytes) -> Dict[str, str]:
    """Persist raw document bytes."""
    external_id = meta.get("external_id")
    if not external_id:
        raise ValueError("external_id required for ingest_raw")

    path = _build_path(meta, "raw", name)
    object_store.put_bytes(path, data)
    content_hash = hashlib.sha256(data).hexdigest()
    meta["content_hash"] = content_hash
    return {"path": path, "content_hash": content_hash}


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
def _split_sentences(text: str) -> List[str]:
    """Best-effort sentence segmentation that retains punctuation."""

    pattern = re.compile(r"[^.!?]+(?:[.!?]+|\Z)")
    sentences = [segment.strip() for segment in pattern.findall(text)]
    sentences = [s for s in sentences if s]
    if sentences:
        if len(sentences) == 1 and not re.search(r"[.!?]", text):
            sentences = []
        else:
            return sentences
    # Fallback: use paragraphs or lines if no sentence boundary detected
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    return parts or [text.strip()]


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _chunkify(
    sentences: Sequence[str],
    *,
    target_tokens: int,
    overlap_tokens: int,
    hard_limit: int,
) -> List[str]:
    chunks: List[str] = []
    current: List[Tuple[str, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunk_text = " ".join(sentence for sentence, _ in current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if overlap_tokens > 0:
            retained: List[Tuple[str, int]] = []
            retained_tokens = 0
            for sentence, tokens in reversed(current):
                retained.insert(0, (sentence, tokens))
                retained_tokens += tokens
                if retained_tokens >= overlap_tokens:
                    break
            current = retained
            current_tokens = retained_tokens
        else:
            current = []
            current_tokens = 0

    for sentence in sentences:
        tokens = _token_count(sentence)
        if tokens > hard_limit:
            words = sentence.split()
            start = 0
            while start < len(words):
                end = min(start + hard_limit, len(words))
                sub_sentence = " ".join(words[start:end])
                start = end
                if current_tokens + _token_count(sub_sentence) > hard_limit:
                    flush()
                current.append((sub_sentence, _token_count(sub_sentence)))
                current_tokens += _token_count(sub_sentence)
                if current_tokens >= target_tokens:
                    flush()
            continue

        if current_tokens + tokens > hard_limit and current:
            flush()

        current.append((sentence, tokens))
        current_tokens += tokens

        if current_tokens >= target_tokens:
            flush()

    flush()
    return chunks


def _build_chunk_prefix(meta: Dict[str, str]) -> str:
    parts: List[str] = []
    breadcrumbs = meta.get("breadcrumbs")
    if isinstance(breadcrumbs, Iterable) and not isinstance(breadcrumbs, (str, bytes)):
        crumb_parts = [str(item).strip() for item in breadcrumbs if str(item).strip()]
        if crumb_parts:
            parts.append(" / ".join(crumb_parts))
    title = meta.get("title")
    if title:
        parts.append(str(title).strip())
    return " — ".join(part for part in parts if part)


def chunk(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
    """Split text into overlapping chunks for embeddings."""

    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    content_hash = meta.get("content_hash")
    if not content_hash:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        meta["content_hash"] = content_hash
    external_id = meta.get("external_id")
    if not external_id:
        raise ValueError("external_id required for chunk")

    target_tokens = int(getattr(settings, "RAG_CHUNK_TARGET_TOKENS", 450))
    overlap_tokens = int(getattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", 80))
    hard_limit = max(target_tokens, 512)
    sentences = _split_sentences(text)
    prefix = _build_chunk_prefix(meta)
    chunk_bodies = _chunkify(
        sentences,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        hard_limit=hard_limit,
    )

    chunks: List[Dict[str, object]] = []
    for body in chunk_bodies:
        chunk_text = body
        if prefix:
            chunk_text = f"{prefix}\n\n{body}" if body else prefix
        normalised = normalise_text(chunk_text)
        chunks.append(
            {
                "content": chunk_text,
                "normalized": normalised,
                "meta": {
                    "tenant": meta["tenant"],
                    "case": meta.get("case"),
                    "source": text_path,
                    "hash": content_hash,
                    "external_id": external_id,
                    "content_hash": content_hash,
                },
            }
        )

    if not chunks:
        normalised = normalise_text(text)
        chunks.append(
            {
                "content": text,
                "normalized": normalised,
                "meta": {
                    "tenant": meta["tenant"],
                    "case": meta.get("case"),
                    "source": text_path,
                    "hash": content_hash,
                    "external_id": external_id,
                    "content_hash": content_hash,
                },
            }
        )

    out_path = _build_path(meta, "embeddings", "chunks.json")
    object_store.write_json(out_path, chunks)
    return {"path": out_path}


@shared_task(base=ScopedTask)
def embed(meta: Dict[str, str], chunks_path: str) -> Dict[str, str]:
    """Attach dummy embedding vectors to chunks."""

    chunks = object_store.read_json(chunks_path)
    embeddings = []
    for ch in chunks:
        normalised = ch.get("normalized") or normalise_text(ch.get("content", ""))
        magnitude = float(len((normalised or "").split()) or 1)
        embedding = [0.0] * EMBEDDING_DIM
        embedding[0] = magnitude
        embeddings.append({**ch, "embedding": embedding, "normalized": normalised})
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
