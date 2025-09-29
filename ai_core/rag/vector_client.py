from __future__ import annotations

import atexit
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from psycopg2 import sql
from psycopg2.extras import Json, register_default_jsonb
from psycopg2.pool import SimpleConnectionPool

from common.logging import get_logger
from ai_core.rag.vector_store import VectorStore

from . import metrics
from .filters import strict_match
from .schemas import Chunk

# Ensure jsonb columns are decoded into Python dictionaries
register_default_jsonb(loads=json.loads, globally=True)

logger = get_logger(__name__)


# Welche Filter-Schlüssel sind erlaubt und worauf mappen sie?
# - "chunk_meta": JSONB c.metadata ->> '<key>'
# - "document_hash": Spalte d.hash
# - "document_id":  Spalte d.id::text
SUPPORTED_METADATA_FILTERS = {
    "case": "chunk_meta",
    "source": "chunk_meta",
    "doctype": "chunk_meta",
    "published": "chunk_meta",
    "hash": "document_hash",
    "id": "document_id",
    "external_id": "document_external_id",
}


FALLBACK_STATEMENT_TIMEOUT_MS = 15000
FALLBACK_RETRY_ATTEMPTS = 3
FALLBACK_RETRY_BASE_DELAY_MS = 50
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1536"))


DocumentKey = Tuple[str, str]
GroupedDocuments = Dict[DocumentKey, Dict[str, object]]
T = TypeVar("T")


class UpsertResult(int):
    """Integer-compatible return type carrying per-document ingestion metadata."""

    def __new__(
        cls, written: int, documents: List[Dict[str, object]]
    ) -> "UpsertResult":
        obj = int.__new__(cls, written)
        obj._documents = documents
        return obj

    @property
    def documents(self) -> List[Dict[str, object]]:
        return list(self._documents)


class PgVectorClient:
    """pgvector-backed client for chunk storage and retrieval."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "rag",
        minconn: int = 1,
        maxconn: int = 5,
        statement_timeout_ms: Optional[int] = None,
        retries: Optional[int] = None,
        retry_base_delay_ms: Optional[int] = None,
    ) -> None:
        if minconn < 1 or maxconn < minconn:
            raise ValueError("Invalid connection pool configuration")
        self._schema = schema
        env_timeout = int(
            os.getenv("RAG_STATEMENT_TIMEOUT_MS", str(FALLBACK_STATEMENT_TIMEOUT_MS))
        )
        env_retries = int(os.getenv("RAG_RETRY_ATTEMPTS", str(FALLBACK_RETRY_ATTEMPTS)))
        env_retry_delay = int(
            os.getenv("RAG_RETRY_BASE_DELAY_MS", str(FALLBACK_RETRY_BASE_DELAY_MS))
        )
        timeout_value = statement_timeout_ms if statement_timeout_ms is not None else env_timeout
        retries_value = retries if retries is not None else env_retries
        retry_delay_value = (
            retry_base_delay_ms if retry_base_delay_ms is not None else env_retry_delay
        )
        self._statement_timeout_ms = timeout_value
        self._pool = SimpleConnectionPool(minconn, maxconn, dsn)
        self._prepare_lock = threading.Lock()
        self._indexes_ready = False
        self._retries = max(1, retries_value)
        self._retry_base_delay = max(0, retry_delay_value) / 1000.0

    @classmethod
    def from_env(
        cls,
        *,
        env_var: str = "RAG_DATABASE_URL",
        fallback_env_var: str = "DATABASE_URL",
        **kwargs: object,
    ) -> "PgVectorClient":
        dsn = os.getenv(env_var) or os.getenv(fallback_env_var)
        if not dsn:
            raise RuntimeError(
                f"Neither {env_var} nor {fallback_env_var} is set; cannot initialise PgVectorClient"
            )
        return cls(dsn, **kwargs)

    def close(self) -> None:
        """Close all pooled connections."""

        self._pool.closeall()

    @contextmanager
    def _connection(self):  # type: ignore[no-untyped-def]
        conn = self._pool.getconn()
        try:
            self._prepare_connection(conn)
            yield conn
        finally:
            self._pool.putconn(conn)

    def _prepare_connection(self, conn) -> None:  # type: ignore[no-untyped-def]
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(self._schema)
                )
            )
        if self._indexes_ready:
            return
        with self._prepare_lock:
            if self._indexes_ready:
                return
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw
                    ON embeddings USING hnsw (embedding vector_l2_ops)
                    """
                )
                cur.execute("ANALYZE embeddings")
            conn.commit()
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SET search_path TO {}, public").format(
                        sql.Identifier(self._schema)
                    )
                )
            self._indexes_ready = True

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            logger.info("Skipping vector upsert because no chunks were provided")
            return 0

        grouped = self._group_by_document(chunk_list)
        tenants = sorted({key[0] for key in grouped})
        inserted_chunks = 0
        doc_actions: Dict[DocumentKey, str] = {}
        per_doc_timings: Dict[DocumentKey, Dict[str, float]] = {}

        def _operation() -> float:
            started = time.perf_counter()
            nonlocal inserted_chunks, doc_actions, per_doc_timings
            with self._connection() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SET LOCAL statement_timeout = %s",
                            (str(self._statement_timeout_ms),),
                        )
                        document_ids, doc_actions = self._ensure_documents(
                            cur, grouped
                        )
                        inserted_chunks, per_doc_timings = self._replace_chunks(
                            cur, grouped, document_ids, doc_actions
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return (time.perf_counter() - started) * 1000

        duration_ms = self._run_with_retries(_operation, op_name="upsert_chunks")

        skipped_documents = sum(1 for action in doc_actions.values() if action == "skipped")
        metrics.RAG_UPSERT_CHUNKS.inc(inserted_chunks)
        documents_info: List[Dict[str, object]] = []
        for key, doc in grouped.items():
            tenant_id, external_id = key
            action = doc_actions.get(key, "inserted")
            stats = per_doc_timings.get(
                key,
                {
                    "chunk_count": len(doc.get("chunks", [])),
                    "duration_ms": 0.0,
                },
            )
            chunk_count = int(stats.get("chunk_count", 0))
            duration = float(stats.get("duration_ms", 0.0))
            doc_payload = {
                "tenant": tenant_id,
                "external_id": external_id,
                "content_hash": doc.get("hash"),
                "action": action,
                "chunk_count": chunk_count,
                "duration_ms": duration,
            }
            documents_info.append(doc_payload)
            logger.info("ingestion.doc.result", extra=doc_payload)
            if action == "inserted":
                metrics.INGESTION_DOCS_INSERTED.inc()
            elif action == "replaced":
                metrics.INGESTION_DOCS_REPLACED.inc()
            else:
                metrics.INGESTION_DOCS_SKIPPED.inc()
            if action in {"inserted", "replaced"} and chunk_count:
                metrics.INGESTION_CHUNKS_WRITTEN.inc(float(chunk_count))
        logger.info(
            "RAG upsert completed: chunks=%d documents=%d tenants=%s skipped=%d duration_ms=%.2f",
            inserted_chunks,
            len(grouped),
            tenants,
            skipped_documents,
            duration_ms,
        )
        return UpsertResult(inserted_chunks, documents_info)

    def search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
    ) -> List[Chunk]:
        top_k = min(max(1, top_k), 10)
        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        tenant = str(tenant_uuid)
        normalized_filters: Dict[str, object | None] = {}
        if filters:
            normalized_filters = {
                key: (
                    value
                    if not (isinstance(value, str) and value == "")
                    and value is not None
                    else None
                )
                for key, value in filters.items()
            }
        case_value: Optional[str]
        if case_id not in {None, ""}:
            case_value = case_id
        else:
            case_value = normalized_filters.get("case")
        if case_value is not None:
            case_value = str(case_value)
        normalized_filters["tenant"] = tenant
        normalized_filters["case"] = case_value
        metadata_filters = [
            (key, value)
            for key, value in normalized_filters.items()
            if key not in {"tenant"}
            and value is not None
            and key in SUPPORTED_METADATA_FILTERS
        ]
        filter_debug: Dict[str, object | None] = {"tenant": "<set>"}
        for key, value in normalized_filters.items():
            if key == "tenant":
                continue
            filter_debug[key] = (
                "<set>"
                if value is not None and key in SUPPORTED_METADATA_FILTERS
                else None
            )
        logger.debug(
            "RAG search normalised inputs: tenant=%s top_k=%d filters=%s",
            tenant,
            top_k,
            filter_debug,
        )
        query_vec = self._format_vector(self._embed_query(query))

        def _operation() -> Tuple[
            List[Tuple[str, Mapping[str, object], Optional[str], object, float]],
            float,
        ]:
            started = time.perf_counter()
            with self._connection() as conn:
                with conn.cursor() as cur:
                    where_clauses = ["d.tenant_id::text = %s"]
                    where_params: List[object] = [tenant]
                    for key, value in metadata_filters:

                        kind = SUPPORTED_METADATA_FILTERS[key]
                        normalised = self._normalise_filter_value(value)
                        if kind == "chunk_meta":
                            where_clauses.append("c.metadata ->> %s = %s")
                            where_params.extend([key, normalised])
                        elif kind == "document_hash":
                            where_clauses.append("d.hash = %s")
                            where_params.append(normalised)
                        elif kind == "document_id":
                            where_clauses.append("d.id::text = %s")
                            where_params.append(normalised)
                        elif kind == "document_external_id":
                            where_clauses.append("d.external_id = %s")
                            where_params.append(normalised)
                    where_sql = "\n          AND ".join(where_clauses)
                    query_sql = f"""
                        SELECT
                            c.text,
                            c.metadata,
                            d.hash,
                            d.id,
                            e.embedding <-> %s::vector AS distance
                        FROM embeddings e
                        JOIN chunks c ON e.chunk_id = c.id
                        JOIN documents d ON c.document_id = d.id
                        WHERE {where_sql}
                        ORDER BY distance
                        LIMIT %s
                    """
                    cur.execute(query_sql, (query_vec, *where_params, top_k))
                    rows = cur.fetchall()
            return rows, (time.perf_counter() - started) * 1000

        rows, duration_ms = self._run_with_retries(_operation, op_name="search")

        results: List[Chunk] = []
        for text, metadata, doc_hash, doc_id, distance in rows:
            meta = dict(metadata or {})
            if doc_hash and not meta.get("hash"):
                meta["hash"] = doc_hash
            if doc_id is not None and "id" not in meta:
                meta["id"] = str(doc_id)
            if not strict_match(meta, tenant, case_value):
                continue
            meta["score"] = self._distance_to_score(distance)
            results.append(Chunk(content=text, meta=meta))
        metrics.RAG_SEARCH_MS.observe(duration_ms)
        logger.info(
            "RAG search executed: tenant=%s case=%s query_chars=%d results=%d duration_ms=%.2f",
            tenant,
            case_value,
            len(query),
            len(results),
            duration_ms,
        )
        return results

    def health_check(self) -> bool:
        """Run a lightweight query to assert connectivity."""

        def _operation() -> bool:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return True

        return bool(self._run_with_retries(_operation, op_name="health_check"))

    def _group_by_document(self, chunks: Sequence[Chunk]) -> GroupedDocuments:
        grouped: GroupedDocuments = {}
        for chunk in chunks:
            tenant_value = chunk.meta.get("tenant")
            doc_hash = str(chunk.meta.get("hash"))
            source = chunk.meta.get("source", "")
            external_id = chunk.meta.get("external_id")
            if tenant_value in {None, "", "None"}:
                raise ValueError("Chunk metadata must include tenant")
            if not doc_hash or doc_hash == "None":
                raise ValueError("Chunk metadata must include hash")
            if external_id in {None, "", "None"}:
                raise ValueError("Chunk metadata must include external_id")
            tenant_uuid = self._coerce_tenant_uuid(tenant_value)
            tenant = str(tenant_uuid)
            external_id_str = str(external_id)
            key = (tenant, external_id_str)
            if key not in grouped:
                grouped[key] = {
                    "id": uuid.uuid4(),
                    "tenant_id": tenant,
                    "external_id": external_id_str,
                    "hash": doc_hash,
                    "source": source,
                    "metadata": {
                        k: v
                        for k, v in chunk.meta.items()
                        if k not in {"tenant", "hash", "source"}
                    },
                    "chunks": [],
                }
            chunk_meta = dict(chunk.meta)
            chunk_meta["tenant"] = tenant
            chunk_meta["external_id"] = external_id_str
            grouped[key]["chunks"].append(
                Chunk(content=chunk.content, meta=chunk_meta, embedding=chunk.embedding)
            )
        return grouped

    def _ensure_documents(
        self,
        cur,
        grouped: GroupedDocuments,
    ) -> Tuple[Dict[DocumentKey, uuid.UUID], Dict[DocumentKey, str]]:  # type: ignore[no-untyped-def]
        document_ids: Dict[DocumentKey, uuid.UUID] = {}
        actions: Dict[DocumentKey, str] = {}
        for key, doc in grouped.items():
            tenant_uuid = self._coerce_tenant_uuid(doc["tenant_id"])
            metadata = Json(doc["metadata"])
            external_id = doc["external_id"]
            cur.execute(
                """
                SELECT id, hash
                FROM documents
                WHERE tenant_id = %s AND external_id = %s
                FOR UPDATE
                """,
                (str(tenant_uuid), external_id),
            )
            existing = cur.fetchone()
            if existing:
                document_id, stored_hash = existing
                if stored_hash == doc["hash"]:
                    document_ids[key] = document_id
                    actions[key] = "skipped"
                    logger.info(
                        "Skipping unchanged document during upsert",
                        extra={
                            "tenant": doc["tenant_id"],
                            "external_id": external_id,
                        },
                    )
                    continue
                cur.execute(
                    """
                    UPDATE documents
                    SET hash = %s,
                        source = %s,
                        metadata = %s,
                        deleted_at = NULL
                    WHERE id = %s
                    """,
                    (doc["hash"], doc["source"], metadata, document_id),
                )
                document_ids[key] = document_id
                actions[key] = "replaced"
                continue

            document_id = doc["id"]
            cur.execute(
                """
                INSERT INTO documents (id, tenant_id, external_id, source, hash, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    str(tenant_uuid),
                    external_id,
                    doc["source"],
                    doc["hash"],
                    metadata,
                ),
            )
            document_ids[key] = document_id
            actions[key] = "inserted"
        return document_ids, actions

    def _coerce_tenant_uuid(self, tenant_id: object) -> uuid.UUID:
        try:
            return uuid.UUID(str(tenant_id))
        except (TypeError, ValueError):
            if tenant_id in {None, "", "None"}:
                raise ValueError("Chunk metadata must include a tenant identifier")
            derived = uuid.uuid5(uuid.NAMESPACE_URL, f"tenant:{tenant_id}")
            logger.warning(
                "Mapped legacy tenant identifier to deterministic UUID",
                extra={"tenant": tenant_id, "derived_tenant_uuid": str(derived)},
            )
            return derived

    def _replace_chunks(
        self,
        cur,
        grouped: GroupedDocuments,
        document_ids: Dict[DocumentKey, uuid.UUID],
        doc_actions: Dict[DocumentKey, str],
    ) -> Tuple[int, Dict[DocumentKey, Dict[str, float]]]:  # type: ignore[no-untyped-def]
        chunk_insert_sql = """
            INSERT INTO chunks (id, document_id, ord, text, tokens, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        embedding_insert_sql = """
            INSERT INTO embeddings (id, chunk_id, embedding)
            VALUES (%s, %s, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """

        inserted = 0
        per_doc_stats: Dict[DocumentKey, Dict[str, float]] = {}
        for key, doc in grouped.items():
            action = doc_actions.get(key, "inserted")
            if action == "skipped":
                per_doc_stats[key] = {"chunk_count": 0, "duration_ms": 0.0}
                continue
            document_id = document_ids[key]
            started = time.perf_counter()
            cur.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = %s)",
                (document_id,),
            )
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

            chunk_rows = []
            embedding_rows = []
            chunk_count = 0
            for index, chunk in enumerate(doc["chunks"]):
                chunk_id = uuid.uuid4()
                tokens = self._estimate_tokens(chunk.content)
                chunk_rows.append(
                    (
                        chunk_id,
                        document_id,
                        index,
                        chunk.content,
                        tokens,
                        Json(dict(chunk.meta)),
                    )
                )
                chunk_count += 1
                if chunk.embedding is not None:
                    vector_value = self._format_vector(chunk.embedding)
                    embedding_rows.append((uuid.uuid4(), chunk_id, vector_value))

            if chunk_rows:
                cur.executemany(chunk_insert_sql, chunk_rows)
            if embedding_rows:
                cur.executemany(embedding_insert_sql, embedding_rows)
            inserted += chunk_count
            per_doc_stats[key] = {
                "chunk_count": float(chunk_count),
                "duration_ms": (time.perf_counter() - started) * 1000,
            }
        return inserted, per_doc_stats

    def _estimate_tokens(self, content: str) -> int:
        return max(1, len(content.split()))

    def _format_vector(self, values: Sequence[float]) -> str:
        if len(values) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(values)}"
            )
        return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"

    def _embed_query(self, query: str) -> List[float]:
        base = float(len(query.strip()) or 1)
        return [base] + [0.0] * (EMBEDDING_DIM - 1)

    def _distance_to_score(self, distance: float) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(value) or math.isinf(value):
            return 0.0
        if value < 0:
            value = 0.0
        return 1.0 / (1.0 + value)

    def _normalise_filter_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(value)

    def _run_with_retries(self, fn: Callable[[], T], *, op_name: str) -> T:
        """Execute ``fn`` with retry semantics and return its ``TypeVar`` result.

        The callable ``fn`` is invoked until it succeeds or the configured
        number of attempts is exhausted. Each retry waits ``attempt *
        self._retry_base_delay`` seconds, providing a linear backoff. Using the
        ``Callable[[], T]`` signature together with ``TypeVar('T')`` preserves
        the original return type (float, tuple, bool, ...), so callers receive
        the exact value produced by ``fn`` once it finally succeeds.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - requires failure injection
                last_exc = exc
                logger.warning(
                    "pgvector operation failed, retrying",
                    extra={"operation": op_name, "attempt": attempt},
                )
                metrics.RAG_RETRY_ATTEMPTS.labels(operation=op_name).inc()
                if attempt == self._retries:
                    raise
                time.sleep(self._retry_base_delay * attempt)
        if last_exc is not None:  # pragma: no cover - defensive
            raise last_exc
        raise RuntimeError("retry loop exited without result")


_DEFAULT_CLIENT: Optional[VectorStore] = None


def get_default_client() -> PgVectorClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = PgVectorClient.from_env()
    return cast(PgVectorClient, _DEFAULT_CLIENT)


def reset_default_client() -> None:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        _DEFAULT_CLIENT.close()
    _DEFAULT_CLIENT = None


atexit.register(reset_default_client)

try:  # pragma: no cover - celery optional
    from celery.signals import worker_shutdown
except Exception:  # pragma: no cover
    worker_shutdown = None
else:  # pragma: no cover - requires celery runtime

    @worker_shutdown.connect  # type: ignore[attr-defined]
    def _close_pgvector_pool(**_kwargs):
        reset_default_client()
