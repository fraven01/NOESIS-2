from __future__ import annotations

import atexit
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, cast

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

DEFAULT_STATEMENT_TIMEOUT_MS = int(os.getenv("RAG_STATEMENT_TIMEOUT_MS", "15000"))
DEFAULT_RETRY_ATTEMPTS = int(os.getenv("RAG_RETRY_ATTEMPTS", "3"))
DEFAULT_RETRY_BASE_DELAY_MS = int(os.getenv("RAG_RETRY_BASE_DELAY_MS", "50"))
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1536"))


DocumentKey = Tuple[str, str]
GroupedDocuments = Dict[DocumentKey, Dict[str, object]]
T = TypeVar("T")


class PgVectorClient:
    """pgvector-backed client for chunk storage and retrieval."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "rag",
        minconn: int = 1,
        maxconn: int = 5,
        statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
        retries: int = DEFAULT_RETRY_ATTEMPTS,
        retry_base_delay_ms: int = DEFAULT_RETRY_BASE_DELAY_MS,
    ) -> None:
        if minconn < 1 or maxconn < minconn:
            raise ValueError("Invalid connection pool configuration")
        self._schema = schema
        self._statement_timeout_ms = statement_timeout_ms
        self._pool = SimpleConnectionPool(minconn, maxconn, dsn)
        self._prepare_lock = threading.Lock()
        self._indexes_ready = False
        self._retries = max(1, retries)
        self._retry_base_delay = max(0, retry_base_delay_ms) / 1000.0

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

        def _operation() -> float:
            started = time.perf_counter()
            with self._connection() as conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SET LOCAL statement_timeout = %s",
                            (str(self._statement_timeout_ms),),
                        )
                        document_ids = self._ensure_documents(cur, grouped)
                        self._replace_chunks(cur, grouped, document_ids)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return (time.perf_counter() - started) * 1000

        duration_ms = self._run_with_retries(_operation, op_name="upsert_chunks")

        metrics.RAG_UPSERT_CHUNKS.inc(len(chunk_list))
        logger.info(
            "RAG upsert completed: chunks=%d documents=%d tenants=%s duration_ms=%.2f",
            len(chunk_list),
            len(grouped),
            tenants,
            duration_ms,
        )
        return len(chunk_list)

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
                    if not (isinstance(value, str) and value == "") and value is not None
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
            if key not in {"tenant"} and value is not None
        ]
        filter_debug = {
            key: ("<set>" if value is not None else None)
            for key, value in normalized_filters.items()
        }
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
                        where_clauses.append("c.metadata @> %s::jsonb")
                        where_params.append(
                            Json({key: self._normalise_filter_json_value(value)})
                        )
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
            if tenant_value in {None, "", "None"}:
                raise ValueError("Chunk metadata must include tenant")
            if not doc_hash or doc_hash == "None":
                raise ValueError("Chunk metadata must include hash")
            tenant_uuid = self._coerce_tenant_uuid(tenant_value)
            tenant = str(tenant_uuid)
            key = (tenant, doc_hash)
            if key not in grouped:
                grouped[key] = {
                    "id": uuid.uuid4(),
                    "tenant_id": tenant,
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
            grouped[key]["chunks"].append(
                Chunk(content=chunk.content, meta=chunk_meta, embedding=chunk.embedding)
            )
        return grouped

    def _ensure_documents(
        self,
        cur,
        grouped: GroupedDocuments,
    ) -> Dict[DocumentKey, uuid.UUID]:  # type: ignore[no-untyped-def]
        document_ids: Dict[DocumentKey, uuid.UUID] = {}
        for key, doc in grouped.items():
            tenant_uuid = self._coerce_tenant_uuid(doc["tenant_id"])
            metadata = Json(doc["metadata"])
            cur.execute(
                """
                INSERT INTO documents (id, tenant_id, source, hash, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, hash)
                DO UPDATE SET
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata,
                    deleted_at = NULL
                RETURNING id
                """,
                (doc["id"], str(tenant_uuid), doc["source"], doc["hash"], metadata),
            )
            returned = cur.fetchone()
            if returned is None:
                raise RuntimeError("Failed to upsert document")
            document_ids[key] = returned[0]
        return document_ids

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
    ) -> None:  # type: ignore[no-untyped-def]
        chunk_insert_sql = """
            INSERT INTO chunks (id, document_id, ord, text, tokens, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        embedding_insert_sql = """
            INSERT INTO embeddings (id, chunk_id, embedding)
            VALUES (%s, %s, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """

        for key, doc in grouped.items():
            document_id = document_ids[key]
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

            chunk_rows = []
            embedding_rows = []
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
                if chunk.embedding is not None:
                    vector_value = self._format_vector(chunk.embedding)
                    embedding_rows.append((uuid.uuid4(), chunk_id, vector_value))

            if chunk_rows:
                cur.executemany(chunk_insert_sql, chunk_rows)
            if embedding_rows:
                cur.executemany(embedding_insert_sql, embedding_rows)

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

    def _normalise_filter_json_value(self, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (str, bool, int, float)):
            return value
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
