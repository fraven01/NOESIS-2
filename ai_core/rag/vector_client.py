from __future__ import annotations

import atexit
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from psycopg2 import sql
from psycopg2.extras import Json, register_default_jsonb
from psycopg2.pool import SimpleConnectionPool

from common.logging import get_logger

from . import metrics
from .filters import strict_match
from .schemas import Chunk

# Ensure jsonb columns are decoded into Python dictionaries
register_default_jsonb(loads=json.loads, globally=True)

logger = get_logger(__name__)

DEFAULT_STATEMENT_TIMEOUT_MS = 15000
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1536"))


DocumentKey = Tuple[str, str]
GroupedDocuments = Dict[DocumentKey, Dict[str, object]]


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
    ) -> None:
        if minconn < 1 or maxconn < minconn:
            raise ValueError("Invalid connection pool configuration")
        self._schema = schema
        self._statement_timeout_ms = statement_timeout_ms
        self._pool = SimpleConnectionPool(minconn, maxconn, dsn)
        self._prepare_lock = threading.Lock()
        self._indexes_ready = False

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

        duration_ms = (time.perf_counter() - started) * 1000
        tenants = sorted({key[0] for key in grouped})
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
        filters: Dict[str, Optional[str]],
        top_k: int = 5,
    ) -> List[Chunk]:
        tenant = filters.get("tenant")
        case = filters.get("case")
        query_vec = self._format_vector(self._embed_query(query))
        started = time.perf_counter()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.text, c.metadata
                    FROM embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    JOIN documents d ON c.document_id = d.id
                    WHERE (%s IS NULL OR d.tenant_id::text = %s)
                      AND (%s IS NULL OR c.metadata ->> 'case' = %s)
                    ORDER BY e.embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (tenant, tenant, case, case, query_vec, top_k),
                )
                rows = cur.fetchall()
        results: List[Chunk] = []
        for text, metadata in rows:
            meta = dict(metadata or {})
            if not strict_match(meta, tenant, case):
                continue
            results.append(Chunk(content=text, meta=meta))
        duration_ms = (time.perf_counter() - started) * 1000
        metrics.RAG_SEARCH_MS.observe(duration_ms)
        logger.info(
            "RAG search executed: tenant=%s case=%s query_chars=%d results=%d duration_ms=%.2f",
            tenant,
            case,
            len(query),
            len(results),
            duration_ms,
        )
        return results

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


_DEFAULT_CLIENT: Optional[PgVectorClient] = None


def get_default_client() -> PgVectorClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = PgVectorClient.from_env()
    return _DEFAULT_CLIENT


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
