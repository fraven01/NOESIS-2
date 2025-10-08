from __future__ import annotations

import hashlib
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

from celery import shared_task
from common.celery import ScopedTask
from common.logging import get_logger
from django.conf import settings
from django.utils import timezone

from .infra import object_store, pii, tracing
from .rag import metrics
from .rag.schemas import Chunk
from .rag.normalization import normalise_text
from .rag.ingestion_contracts import ensure_embedding_dimensions
from .rag.embeddings import (
    EmbeddingBatchResult,
    EmbeddingClientError,
    get_embedding_client,
)
from .rag.vector_store import get_default_router

logger = get_logger(__name__)


def _build_path(meta: Dict[str, str], *parts: str) -> str:
    tenant = object_store.sanitize_identifier(meta["tenant_id"])
    case = object_store.sanitize_identifier(meta["case_id"])
    return "/".join([tenant, case, *parts])


def log_ingestion_run_start(
    *,
    tenant: str,
    case: str,
    run_id: str,
    doc_count: int,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
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
    if embedding_profile:
        extra["embedding_profile"] = embedding_profile
    if vector_space_id:
        extra["vector_space_id"] = vector_space_id
    logger.info("ingestion.start", extra=extra)
    if trace_id:
        tracing.emit_span(
            trace_id=trace_id,
            node_name="rag.ingestion.run.start",
            metadata={**extra},
        )


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
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
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
    if embedding_profile:
        extra["embedding_profile"] = embedding_profile
    if vector_space_id:
        extra["vector_space_id"] = vector_space_id
    logger.info("ingestion.end", extra=extra)
    metrics.INGESTION_RUN_MS.observe(float(duration_ms))
    if trace_id:
        tracing.emit_span(
            trace_id=trace_id,
            node_name="rag.ingestion.run.end",
            metadata={**extra},
        )


@shared_task(base=ScopedTask, accepts_scope=True)
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


@shared_task(base=ScopedTask, accepts_scope=True)
def extract_text(meta: Dict[str, str], raw_path: str) -> Dict[str, str]:
    """Decode bytes to text and store."""
    full = object_store.BASE_PATH / raw_path
    text = full.read_bytes().decode("utf-8")
    out_path = _build_path(meta, "text", f"{Path(raw_path).stem}.txt")
    object_store.put_bytes(out_path, text.encode("utf-8"))
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
def pii_mask(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
    """Mask PII in text."""
    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    masked = pii.mask(text)
    if masked == text:
        masked = re.sub(r"\d", "X", text)
    out_path = _build_path(meta, "text", f"{Path(text_path).stem}.masked.txt")
    object_store.put_bytes(out_path, masked.encode("utf-8"))
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
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


if tiktoken:
    try:
        _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - defensive fallback
        _TOKEN_ENCODING = None
else:
    _TOKEN_ENCODING = None


_FORCE_WHITESPACE_TOKENIZER = os.getenv(
    "AI_CORE_FORCE_WHITESPACE_TOKENIZER", ""
).lower() in (
    "1",
    "true",
    "yes",
)


def _should_use_tiktoken() -> bool:
    return _TOKEN_ENCODING is not None and not _FORCE_WHITESPACE_TOKENIZER


def set_tokenizer_override(force_whitespace: bool) -> None:
    global _FORCE_WHITESPACE_TOKENIZER
    _FORCE_WHITESPACE_TOKENIZER = force_whitespace


@contextmanager
def force_whitespace_tokenizer() -> Iterator[None]:
    """Temporarily force the whitespace-based tokenizer fallback."""

    previous = _FORCE_WHITESPACE_TOKENIZER
    set_tokenizer_override(True)
    try:
        yield
    finally:
        set_tokenizer_override(previous)


def _token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0

    if _should_use_tiktoken():
        # `disallowed_special=()` ensures consistent behaviour across tiktoken versions.
        return max(1, len(_TOKEN_ENCODING.encode(stripped, disallowed_special=())))

    whitespace_tokens = [segment for segment in stripped.split() if segment]
    if len(whitespace_tokens) > 1:
        return len(whitespace_tokens)

    return max(1, len(stripped))


def _split_by_limit(text: str, hard_limit: int) -> List[str]:
    if not text:
        return []

    if _should_use_tiktoken():
        token_ids = _TOKEN_ENCODING.encode(text, disallowed_special=())
        if not token_ids:
            return []
        parts: List[str] = []
        for start in range(0, len(token_ids), hard_limit):
            chunk_ids = token_ids[start : start + hard_limit]
            parts.append(_TOKEN_ENCODING.decode(chunk_ids))
        return parts

    whitespace_chunks = list(re.finditer(r"\S+\s*", text))
    if len(whitespace_chunks) > 1:
        parts: List[str] = []
        current_segments: List[str] = []
        current_tokens = 0

        for match in whitespace_chunks:
            segment = match.group(0)
            stripped = segment.strip()

            if not stripped:
                continue

            if current_tokens + 1 > hard_limit and current_segments:
                parts.append("".join(current_segments).rstrip())
                current_segments = []
                current_tokens = 0

            current_segments.append(segment)
            current_tokens += 1

        if current_segments:
            parts.append("".join(current_segments).rstrip())

        return [part for part in parts if part]

    return [text[i : i + hard_limit] for i in range(0, len(text), hard_limit)]


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
            sub_sentences = _split_by_limit(sentence, hard_limit)
            for sub_sentence in sub_sentences:
                sub_tokens = _token_count(sub_sentence)
                if sub_tokens > hard_limit:
                    # Guard against pathological tokenizer fallbacks by forcing a hard trim.
                    sub_sentence = sub_sentence[:hard_limit]
                    sub_tokens = _token_count(sub_sentence)
                if current_tokens + sub_tokens > hard_limit:
                    flush()
                current.append((sub_sentence, sub_tokens))
                current_tokens += sub_tokens
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
        chunk_meta = {
            "tenant_id": meta["tenant_id"],
            "case_id": meta.get("case_id"),
            "source": text_path,
            "hash": content_hash,
            "external_id": external_id,
            "content_hash": content_hash,
        }
        if meta.get("embedding_profile"):
            chunk_meta["embedding_profile"] = meta["embedding_profile"]
        if meta.get("vector_space_id"):
            chunk_meta["vector_space_id"] = meta["vector_space_id"]
        if meta.get("process"):
            chunk_meta["process"] = meta["process"]
        if meta.get("doc_class"):
            chunk_meta["doc_class"] = meta["doc_class"]
        chunks.append(
            {
                "content": chunk_text,
                "normalized": normalised,
                "meta": chunk_meta,
            }
        )

    if not chunks:
        normalised = normalise_text(text)
        chunk_meta = {
            "tenant_id": meta["tenant_id"],
            "case_id": meta.get("case_id"),
            "source": text_path,
            "hash": content_hash,
            "external_id": external_id,
            "content_hash": content_hash,
        }
        if meta.get("embedding_profile"):
            chunk_meta["embedding_profile"] = meta["embedding_profile"]
        if meta.get("vector_space_id"):
            chunk_meta["vector_space_id"] = meta["vector_space_id"]
        if meta.get("process"):
            chunk_meta["process"] = meta["process"]
        if meta.get("doc_class"):
            chunk_meta["doc_class"] = meta["doc_class"]
        chunks.append(
            {
                "content": text,
                "normalized": normalised,
                "meta": chunk_meta,
            }
        )

    out_path = _build_path(meta, "embeddings", "chunks.json")
    object_store.write_json(out_path, chunks)
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
def embed(meta: Dict[str, str], chunks_path: str) -> Dict[str, str]:
    """Generate embedding vectors for chunks via LiteLLM."""

    chunks = object_store.read_json(chunks_path)
    client = get_embedding_client()
    batch_size = max(
        1, int(getattr(settings, "EMBEDDINGS_BATCH_SIZE", client.batch_size))
    )

    prepared: List[Dict[str, object]] = []
    for ch in chunks:
        normalised = ch.get("normalized") or normalise_text(ch.get("content", ""))
        prepared.append({**ch, "normalized": normalised or ""})

    total_chunks = len(prepared)
    embeddings: List[Dict[str, object]] = []
    expected_dim: Optional[int] = None
    batches = 0

    for start in range(0, total_chunks, batch_size):
        batch = prepared[start : start + batch_size]
        if not batch:
            continue
        batches += 1
        inputs = [str(entry.get("normalized", "")) for entry in batch]
        batch_started = time.perf_counter()
        result: EmbeddingBatchResult = client.embed(inputs)
        duration_ms = (time.perf_counter() - batch_started) * 1000
        extra = {
            "batch": batches,
            "chunks": len(batch),
            "duration_ms": duration_ms,
            "model": result.model,
            "model_name": result.model,
            "model_used": result.model_used,
            "attempts": result.attempts,
        }
        if result.timeout_s is not None:
            extra["timeout_s"] = result.timeout_s
        key_alias = meta.get("key_alias")
        if key_alias:
            extra["key_alias"] = key_alias
        logger.info("ingestion.embed.batch", extra=extra)
        batch_dim: Optional[int] = len(result.vectors[0]) if result.vectors else None
        current_dim = batch_dim
        try:
            current_dim = client.dim()
        except EmbeddingClientError:
            current_dim = batch_dim
        if current_dim is not None:
            if expected_dim is None:
                expected_dim = current_dim
            elif expected_dim != current_dim:
                logger.info(
                    "ingestion.embed.dimension_changed",
                    extra={
                        "previous": expected_dim,
                        "current": current_dim,
                        "model": result.model,
                    },
                )
                expected_dim = current_dim
        if len(result.vectors) != len(batch):
            raise ValueError("Embedding batch size mismatch")

        for entry, vector in zip(batch, result.vectors):
            if expected_dim is not None and len(vector) != expected_dim:
                raise ValueError("Embedding dimension mismatch")
            embeddings.append(
                {
                    **entry,
                    "embedding": list(vector),
                    "vector_dim": len(vector),
                }
            )

    logger.info(
        "ingestion.embed.summary",
        extra={"chunks": total_chunks, "batches": batches},
    )
    out_path = _build_path(meta, "embeddings", "vectors.json")
    object_store.write_json(out_path, embeddings)
    return {"path": out_path}


@shared_task(base=ScopedTask, accepts_scope=True)
def upsert(
    meta: Dict[str, str],
    embeddings_path: str,
    tenant_schema: Optional[str] = None,
) -> int:
    """Upsert embedded chunks into the vector client."""
    data = object_store.read_json(embeddings_path)
    chunk_objs = []
    for ch in data:
        vector = ch.get("embedding")
        embedding = [float(v) for v in vector] if vector is not None else None
        chunk_objs.append(
            Chunk(content=ch["content"], meta=ch["meta"], embedding=embedding)
        )

    tenant_id: Optional[str] = meta.get("tenant_id") if meta else None
    if not tenant_id:
        tenant_id = next(
            (
                str(chunk.meta.get("tenant_id"))
                for chunk in chunk_objs
                if chunk.meta and chunk.meta.get("tenant_id")
            ),
            None,
        )
    if not tenant_id:
        raise ValueError("tenant_id required for upsert")

    for chunk in chunk_objs:
        chunk_tenant = chunk.meta.get("tenant_id") if chunk.meta else None
        if chunk_tenant and str(chunk_tenant) != tenant_id:
            raise ValueError("chunk tenant mismatch")

    expected_dimension_value = meta.get("vector_space_dimension") if meta else None
    expected_dimension: Optional[int] = None
    if expected_dimension_value is not None:
        try:
            expected_dimension = int(expected_dimension_value)
        except (TypeError, ValueError):
            expected_dimension = None

    ensure_embedding_dimensions(
        chunk_objs,
        expected_dimension,
        tenant_id=tenant_id,
        process=meta.get("process") if meta else None,
        doc_class=meta.get("doc_class") if meta else None,
        embedding_profile=meta.get("embedding_profile") if meta else None,
        vector_space_id=meta.get("vector_space_id") if meta else None,
    )

    schema = tenant_schema or (meta.get("tenant_schema") if meta else None)

    router = get_default_router()
    tenant_client = router
    for_tenant = getattr(router, "for_tenant", None)
    if callable(for_tenant):
        try:
            tenant_client = for_tenant(tenant_id, schema)
        except TypeError:
            tenant_client = for_tenant(tenant_id)
    written = tenant_client.upsert_chunks(chunk_objs)
    return written


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
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
