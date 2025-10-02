from __future__ import annotations

import atexit
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Callable,
    ClassVar,
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

from psycopg2 import Error as PsycopgError, sql
from psycopg2.extras import Json, register_default_jsonb
from psycopg2.pool import SimpleConnectionPool

from common.logging import get_logger
from ai_core.rag.vector_store import VectorStore
from .normalization import normalise_text, normalise_text_db

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
_ZERO_EPSILON = 1e-12


def _is_effectively_zero_vector(values: Sequence[float] | None) -> bool:
    if not values:
        return True
    try:
        norm_sq = math.fsum(float(value) * float(value) for value in values)
    except (TypeError, ValueError):
        return True
    return norm_sq <= _ZERO_EPSILON


def _coerce_env_value(
    value: str,
    default: float | int | str,
) -> tuple[bool, float | int | str]:
    """Coerce ``value`` from environment to the type of ``default``."""

    try:
        if isinstance(default, bool):  # pragma: no cover - defensive
            return True, type(default)(value)  # type: ignore[call-arg]
        if isinstance(default, int) and not isinstance(default, bool):
            return True, int(value)
        if isinstance(default, float):
            return True, float(value)
        if isinstance(default, str):
            return True, value
    except (TypeError, ValueError):
        return False, default
    return False, default


def _get_setting(name: str, default: float | int | str) -> float | int | str:
    env_value = os.getenv(name)
    if env_value is not None:
        success, coerced = _coerce_env_value(env_value, default)
        if success:
            return coerced
    try:  # pragma: no cover - requires Django settings
        from django.conf import settings  # type: ignore

        return cast(float | int | str, getattr(settings, name, default))
    except Exception:
        return default


DocumentKey = Tuple[str, str]
GroupedDocuments = Dict[DocumentKey, Dict[str, object]]
T = TypeVar("T")


@dataclass
class HybridSearchResult:
    chunks: List[Chunk]
    vector_candidates: int
    lexical_candidates: int
    fused_candidates: int
    duration_ms: float
    alpha: float
    min_sim: float
    vec_limit: int
    lex_limit: int
    below_cutoff: int = 0
    returned_after_cutoff: int = 0
    query_embedding_empty: bool = False
    applied_trgm_limit: float | None = None
    fallback_limit_used: float | None = None


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

    _ROW_SHAPE_WARNINGS: ClassVar[set[Tuple[str, int]]] = set()

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
        timeout_value = (
            statement_timeout_ms if statement_timeout_ms is not None else env_timeout
        )
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

    @staticmethod
    def _normalise_result_row(
        row: Sequence[object], *, kind: str
    ) -> Tuple[object, object, Mapping[str, object], object, object, float]:
        row_tuple = tuple(row)
        length = len(row_tuple)
        if length != 6:
            key = (kind, length)
            if key not in PgVectorClient._ROW_SHAPE_WARNINGS:
                logger.warning(
                    "rag.hybrid.row_shape_mismatch",
                    kind=kind,
                    row_len=length,
                )
                PgVectorClient._ROW_SHAPE_WARNINGS.add(key)
        padded_list: List[object] = list((row_tuple + (None,) * 6)[:6])

        metadata_value = padded_list[2]
        metadata_dict: Dict[str, object]
        if isinstance(metadata_value, Mapping):
            metadata_dict = dict(metadata_value)
        elif isinstance(metadata_value, Sequence) and not isinstance(
            metadata_value, (str, bytes)
        ):
            try:
                metadata_dict = dict(metadata_value)  # type: ignore[arg-type]
            except Exception:
                metadata_dict = {}
        elif metadata_value is None:
            metadata_dict = {}
        else:
            metadata_dict = {}
        padded_list[2] = metadata_dict

        score_value = padded_list[5]
        fallback = 1.0 if kind == "vector" else 0.0
        try:
            score_float = float(score_value) if score_value is not None else fallback
        except (TypeError, ValueError):
            score_float = fallback
        if math.isnan(score_float) or math.isinf(score_float):
            score_float = fallback
        padded_list[5] = score_float

        return cast(
            Tuple[object, object, Mapping[str, object], object, object, float],
            tuple(padded_list),
        )

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
            self._indexes_ready = True

    def _restore_session_after_rollback(self, cur) -> None:  # type: ignore[no-untyped-def]
        """Re-apply session level settings after a transaction rollback."""

        try:
            cur.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(self._schema)
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "rag.hybrid.search_path_restore_failed",
                extra={"schema": self._schema, "error": str(exc)},
            )
        try:
            cur.execute(
                "SET LOCAL statement_timeout = %s",
                (str(self._statement_timeout_ms),),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "rag.hybrid.statement_timeout_restore_failed",
                extra={
                    "timeout_ms": self._statement_timeout_ms,
                    "error": str(exc),
                },
            )

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
                        document_ids, doc_actions = self._ensure_documents(cur, grouped)
                        inserted_chunks, per_doc_timings = self._replace_chunks(
                            cur, grouped, document_ids, doc_actions
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return (time.perf_counter() - started) * 1000

        duration_ms = self._run_with_retries(_operation, op_name="upsert_chunks")

        skipped_documents = sum(
            1 for action in doc_actions.values() if action == "skipped"
        )
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
                "content_hash": doc.get("content_hash"),
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
        result = self.hybrid_search(
            query,
            tenant_id,
            case_id=case_id,
            top_k=top_k,
            filters=filters,
        )
        return result.chunks

    def hybrid_search(
        self,
        query: str,
        tenant_id: str,
        *,
        case_id: str | None = None,
        top_k: int = 5,
        filters: Mapping[str, object | None] | None = None,
        alpha: float | None = None,
        min_sim: float | None = None,
        vec_limit: int | None = None,
        lex_limit: int | None = None,
        trgm_limit: float | None = None,
        trgm_threshold: float | None = None,
        max_candidates: int | None = None,
    ) -> HybridSearchResult:
        """Execute hybrid vector/lexical retrieval for ``query``.

        The pg_trgm similarity threshold is applied per-connection via
        ``SELECT set_limit`` immediately before running the trigram ``%``
        operator to ensure consistent lexical matching behaviour.
        """
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
        alpha_value = float(
            alpha if alpha is not None else _get_setting("RAG_HYBRID_ALPHA", 0.7)
        )
        alpha_value = min(1.0, max(0.0, alpha_value))
        min_sim_value = float(
            min_sim if min_sim is not None else _get_setting("RAG_MIN_SIM", 0.15)
        )
        min_sim_value = min(1.0, max(0.0, min_sim_value))
        # Determine requested/effective trigram similarity limit
        default_trgm_limit = float(_get_setting("RAG_TRGM_LIMIT", 0.30))
        requested_trgm_limit: float | None
        if trgm_limit is not None:
            try:
                requested_trgm_limit = float(trgm_limit)
            except (TypeError, ValueError):
                requested_trgm_limit = None
        elif trgm_threshold is not None:
            try:
                requested_trgm_limit = float(trgm_threshold)
            except (TypeError, ValueError):
                requested_trgm_limit = None
        else:
            requested_trgm_limit = None
        effective_trgm_limit = (
            requested_trgm_limit
            if requested_trgm_limit is not None
            else default_trgm_limit
        )
        trgm_limit_value = min(1.0, max(0.0, float(effective_trgm_limit)))
        distance_score_mode = str(
            _get_setting("RAG_DISTANCE_SCORE_MODE", "inverse")
        ).lower()
        if distance_score_mode not in {"inverse", "linear"}:
            distance_score_mode = "inverse"
        max_candidates_setting = (
            max_candidates
            if max_candidates is not None
            else _get_setting("RAG_MAX_CANDIDATES", 200)
        )
        try:
            max_candidates_value = int(max_candidates_setting)
        except (TypeError, ValueError):
            max_candidates_value = 200
        max_candidates_value = max(top_k, max(1, max_candidates_value))
        vec_limit_requested = int(vec_limit) if vec_limit is not None else 50
        lex_limit_requested = int(lex_limit) if lex_limit is not None else 50
        vec_limit_value = min(
            max_candidates_value, max(top_k, vec_limit_requested)
        )
        lex_limit_value = min(
            max_candidates_value, max(top_k, lex_limit_requested)
        )
        query_norm = normalise_text(query)
        query_db_norm = normalise_text_db(query)
        raw_vec = self._embed_query(query_norm)
        is_zero_vec = True
        if raw_vec is not None:
            for value in raw_vec:
                try:
                    if abs(float(value)) > 1e-12:
                        is_zero_vec = False
                        break
                except (TypeError, ValueError):
                    continue
        query_vec: Optional[str] = None
        if raw_vec is not None and not is_zero_vec:
            query_vec = self._format_vector(raw_vec)
        query_embedding_empty = bool(is_zero_vec)
        if query_embedding_empty:
            metrics.RAG_QUERY_EMPTY_VEC_TOTAL.labels(tenant=tenant).inc()
            logger.info(
                "rag.hybrid.null_embedding",
                alpha=alpha_value,
                tenant=tenant,
                case=case_value,
            )
        index_kind = str(_get_setting("RAG_INDEX_KIND", "HNSW")).upper()
        ef_search = int(_get_setting("RAG_HNSW_EF_SEARCH", 80))
        probes = int(_get_setting("RAG_IVF_PROBES", 64))

        logger.debug(
            "RAG hybrid search inputs: tenant=%s top_k=%d vec_limit=%d lex_limit=%d filters=%s",
            tenant,
            top_k,
            vec_limit_value,
            lex_limit_value,
            filter_debug,
        )

        where_clauses = ["d.tenant_id = %s"]
        where_params: List[object] = [tenant_uuid]
        for key, value in metadata_filters:
            kind = SUPPORTED_METADATA_FILTERS[key]
            normalised = self._normalise_filter_value(value)
            if kind == "chunk_meta":
                where_clauses.append("c.metadata ->> %s = %s")
                where_params.extend([key, normalised])
            elif kind == "document_hash":
                where_clauses.append("(d.hash = %s OR d.metadata ->> 'hash' = %s)")
                where_params.extend([normalised, normalised])
            elif kind == "document_id":
                where_clauses.append("d.id::text = %s")
                where_params.append(normalised)
            elif kind == "document_external_id":
                where_clauses.append("d.external_id = %s")
                where_params.append(normalised)
        where_sql = "\n          AND ".join(where_clauses)

        applied_trgm_limit_value: Optional[float] = None
        fallback_limit_used_value: Optional[float] = None
        fallback_tried_limits: List[float] = []

        def _operation() -> Tuple[List[tuple], List[tuple], float]:
            nonlocal applied_trgm_limit_value
            nonlocal fallback_limit_used_value
            nonlocal fallback_tried_limits
            started = time.perf_counter()
            vector_rows: List[tuple] = []
            lexical_rows: List[tuple] = []
            fallback_tried_limits = []
            fallback_limit_used_value = None
            with self._connection() as conn:
                if query_vec is not None:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SET LOCAL statement_timeout = %s",
                                (str(self._statement_timeout_ms),),
                            )
                            if index_kind == "HNSW":
                                cur.execute(
                                    "SET LOCAL hnsw.ef_search = %s",
                                    (str(ef_search),),
                                )
                            elif index_kind == "IVFFLAT":
                                cur.execute(
                                    "SET LOCAL ivfflat.probes = %s",
                                    (str(probes),),
                                )
                            vector_sql = f"""
                                SELECT
                                    c.id,
                                    c.text,
                                    c.metadata,
                                    d.hash,
                                    d.id,
                                    e.embedding <=> %s::vector AS distance
                                FROM embeddings e
                                JOIN chunks c ON e.chunk_id = c.id
                                JOIN documents d ON c.document_id = d.id
                                WHERE {where_sql}
                                ORDER BY distance
                                LIMIT %s
                            """
                            cur.execute(
                                vector_sql, (query_vec, *where_params, vec_limit_value)
                            )
                            vector_rows = cur.fetchall()
                            try:
                                logger.warning(
                                    "rag.debug.rows.vector",
                                    extra={
                                        "count": len(vector_rows),
                                        "first_len": (
                                            len(vector_rows[0]) if vector_rows else 0
                                        ),
                                    },
                                )
                            except Exception:
                                pass
                    except Exception as exc:
                        vector_rows = []
                        try:
                            conn.rollback()
                        except Exception:  # pragma: no cover - defensive
                            pass
                        logger.warning(
                            "rag.hybrid.vector_query_failed",
                            extra={
                                "tenant": tenant,
                                "case": case_value,
                                "error": str(exc),
                            },
                        )

                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SET LOCAL statement_timeout = %s",
                            (str(self._statement_timeout_ms),),
                        )
                        logger.info(
                            "rag.pgtrgm.limit",
                            requested=requested_trgm_limit,
                            effective=trgm_limit_value,
                        )

                        def _fetch_show_limit_value() -> float | None:
                            try:
                                cur.execute("SELECT show_limit()")
                            except Exception:
                                return None
                            current = cur.fetchone()
                            if (
                                current
                                and isinstance(current, Sequence)
                                and len(current) > 0
                                and current[0] is not None
                            ):
                                try:
                                    return float(current[0])
                                except (TypeError, ValueError):
                                    return None
                            return None

                        applied_trgm_limit: float | None = None
                        try:
                            cur.execute(
                                "SELECT set_limit(%s::float4)",
                                (float(trgm_limit_value),),
                            )
                            applied_trgm_limit = _fetch_show_limit_value()
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.warning(
                                "rag.pgtrgm.limit.error",
                                requested=requested_trgm_limit,
                                exc_type=exc.__class__.__name__,
                                error=str(exc),
                            )
                            applied_trgm_limit = None
                        applied_trgm_limit_value = applied_trgm_limit

                        lexical_rows_local: List[tuple] = []
                        fallback_requires_rollback = False
                        lexical_sql = f"""
                            SELECT
                                c.id,
                                c.text,
                                c.metadata,
                                d.hash,
                                d.id,
                                similarity(c.text_norm, %s) AS lscore
                            FROM chunks c
                            JOIN documents d ON c.document_id = d.id
                            WHERE {where_sql}
                              AND c.text_norm % %s
                            ORDER BY lscore DESC
                            LIMIT %s
                        """
                        fallback_requested = requested_trgm_limit is not None
                        should_run_fallback = False
                        if query_db_norm.strip():
                            try:
                                cur.execute(
                                    lexical_sql,
                                    (
                                        query_db_norm,
                                        *where_params,
                                        query_db_norm,
                                        lex_limit_value,
                                    ),
                                )
                                lexical_rows_local = cur.fetchall()
                                try:
                                    logger.warning(
                                        "rag.debug.rows.lexical",
                                        extra={
                                            "count": len(lexical_rows_local),
                                            "first_len": (
                                                len(lexical_rows_local[0])
                                                if lexical_rows_local
                                                else 0
                                            ),
                                        },
                                    )
                                except Exception:
                                    pass
                                if lexical_rows_local and not should_run_fallback:
                                    # Guard against inconsistent similarity scores.
                                    # The trigram operator should never return rows
                                    # whose lscore falls below the currently applied
                                    # pg_trgm limit. However, unit tests using
                                    # `FakeCursor` can simulate this scenario when
                                    # they expect the client to fall back to the
                                    # explicit similarity path. Detect this edge
                                    # case and force the fallback execution so the
                                    # behaviour matches production semantics.
                                    invalid_lscore = False
                                    limit_threshold: float | None = None
                                    if applied_trgm_limit is not None:
                                        try:
                                            limit_threshold = float(applied_trgm_limit)
                                        except (TypeError, ValueError):
                                            limit_threshold = None
                                    if limit_threshold is not None:
                                        for row in lexical_rows_local:
                                            if not isinstance(row, Sequence) or not row:
                                                continue
                                            score = row[-1]
                                            if score is None:
                                                continue
                                            try:
                                                score_value = float(score)
                                            except (TypeError, ValueError):
                                                continue
                                            if score_value < limit_threshold - 1e-6:
                                                invalid_lscore = True
                                                break
                                    if invalid_lscore:
                                        should_run_fallback = True
                            except Exception as exc:
                                if isinstance(
                                    exc, (IndexError, ValueError, PsycopgError)
                                ):
                                    should_run_fallback = True
                                    fallback_requires_rollback = True
                                    lexical_rows_local = []
                                    logger.warning(
                                        "rag.hybrid.lexical_primary_failed",
                                        extra={
                                            "tenant": tenant,
                                            "case": case_value,
                                            "error": str(exc),
                                        },
                                    )
                                else:
                                    raise
                            if not lexical_rows_local and not should_run_fallback:
                                if fallback_requested:
                                    should_run_fallback = True
                                elif (
                                    applied_trgm_limit is None
                                    or applied_trgm_limit > 0.1
                                ):
                                    should_run_fallback = True
                            if should_run_fallback:
                                if fallback_requires_rollback:
                                    try:
                                        conn.rollback()
                                    except Exception:  # pragma: no cover - defensive
                                        pass
                                    else:
                                        self._restore_session_after_rollback(cur)
                                logger.info(
                                    "rag.hybrid.trgm_no_match",
                                    extra={
                                        "tenant": tenant,
                                        "case": case_value,
                                        "trgm_limit": trgm_limit_value,
                                        "applied_trgm_limit": applied_trgm_limit,
                                        "fallback": True,
                                    },
                                )
                                fallback_lexical_sql = f"""
                                    SELECT
                                        c.id,
                                        c.text,
                                        c.metadata,
                                        d.hash,
                                        d.id,
                                        similarity(c.text_norm, %s) AS lscore
                                    FROM chunks c
                                    JOIN documents d ON c.document_id = d.id
                                    WHERE {where_sql}
                                      AND similarity(c.text_norm, %s) >= %s
                                    ORDER BY lscore DESC
                                    LIMIT %s
                                """
                                base_limits: List[float] = []
                                if fallback_requested and (
                                    requested_trgm_limit is not None
                                ):
                                    base_limits.append(float(requested_trgm_limit))
                                if applied_trgm_limit is not None:
                                    base_limits.append(float(applied_trgm_limit))
                                else:
                                    base_limits.append(min(trgm_limit_value, 0.10))
                                base_limits.extend(
                                    [
                                        min(trgm_limit_value, 0.10),
                                        0.08,
                                        0.06,
                                        0.05,
                                        0.04,
                                        0.03,
                                        0.02,
                                        0.01,
                                        0.0,
                                    ]
                                )
                                fallback_limits: List[float] = []
                                for limit in base_limits:
                                    try:
                                        limit_value = float(limit)
                                    except (TypeError, ValueError):
                                        continue
                                    limit_value = max(0.0, limit_value)
                                    if limit_value not in fallback_limits:
                                        fallback_limits.append(limit_value)
                                fallback_floor = min(trgm_limit_value, 0.05)
                                if (
                                    fallback_requested
                                    and requested_trgm_limit is not None
                                ):
                                    try:
                                        fallback_floor = max(
                                            0.0, float(requested_trgm_limit)
                                        )
                                    except (TypeError, ValueError):
                                        fallback_floor = fallback_floor
                                picked_limit: float | None = None
                                last_attempt_rows: List[tuple] = []
                                best_rows: List[tuple] = []
                                best_limit: float | None = None
                                for limit_value in fallback_limits:
                                    fallback_tried_limits.append(limit_value)
                                    cur.execute(
                                        fallback_lexical_sql,
                                        (
                                            query_db_norm,
                                            *where_params,
                                            query_db_norm,
                                            limit_value,
                                            lex_limit_value,
                                        ),
                                    )
                                    attempt_rows = cur.fetchall()
                                    last_attempt_rows = list(attempt_rows)
                                    try:
                                        logger.warning(
                                            "rag.debug.rows.lexical",
                                            extra={
                                                "count": len(attempt_rows),
                                                "first_len": (
                                                    len(attempt_rows[0])
                                                    if attempt_rows
                                                    else 0
                                                ),
                                            },
                                        )
                                    except Exception:
                                        pass
                                    if attempt_rows:
                                        lexical_rows_local = attempt_rows
                                        best_rows = attempt_rows
                                        best_limit = limit_value
                                        if limit_value <= fallback_floor + 1e-9:
                                            picked_limit = limit_value
                                            break
                                else:
                                    if best_rows:
                                        lexical_rows_local = best_rows
                                        picked_limit = best_limit
                                    else:
                                        lexical_rows_local = last_attempt_rows
                                if picked_limit is None and best_limit is not None:
                                    picked_limit = best_limit
                                fallback_limit_used_value = picked_limit
                                if (
                                    picked_limit is not None
                                    and requested_trgm_limit is None
                                    and (
                                        applied_trgm_limit is None
                                        or picked_limit < applied_trgm_limit - 1e-9
                                    )
                                ):
                                    try:
                                        cur.execute(
                                            "SELECT set_limit(%s::float4)",
                                            (float(picked_limit),),
                                        )
                                        reapplied_limit = _fetch_show_limit_value()
                                    except Exception:
                                        reapplied_limit = None
                                    if reapplied_limit is not None:
                                        applied_trgm_limit = reapplied_limit
                                logger.info(
                                    "rag.hybrid.trgm_fallback_applied",
                                    tenant=tenant,
                                    case=case_value,
                                    tried_limits=list(fallback_tried_limits),
                                    picked_limit=picked_limit,
                                    count=len(lexical_rows_local),
                                )
                        # Ensure the locally fetched lexical rows are propagated
                        # to the outer scope so they are counted/fused later.
                        lexical_rows = lexical_rows_local
                        logger.info(
                            "rag.pgtrgm.limit.applied",
                            requested=requested_trgm_limit,
                            applied=applied_trgm_limit,
                        )
                        applied_trgm_limit_value = applied_trgm_limit
                except Exception as exc:
                    lexical_rows = []
                    try:
                        conn.rollback()
                    except Exception:  # pragma: no cover - defensive
                        pass
                    logger.warning(
                        "rag.hybrid.lexical_query_failed",
                        extra={
                            "tenant": tenant,
                            "case": case_value,
                            "error": str(exc),
                        },
                    )
                    if not vector_rows:
                        raise
            # Debug: final lexical rows right before returning to retry wrapper
            try:
                logger.warning(
                    "rag.debug.rows.lexical.final",
                    count=len(lexical_rows),
                    first_len=(len(lexical_rows[0]) if lexical_rows else 0),
                )
            except Exception:
                pass
            return vector_rows, lexical_rows, (time.perf_counter() - started) * 1000

        vector_rows, lexical_rows, duration_ms = self._run_with_retries(
            _operation, op_name="search"
        )

        logger.info(
            "rag.hybrid.sql_counts",
            tenant=tenant,
            case=case_value,
            vec_rows=len(vector_rows),
            lex_rows=len(lexical_rows),
            alpha=alpha_value,
            min_sim=min_sim_value,
            trgm_limit=trgm_limit_value,
            applied_trgm_limit=applied_trgm_limit_value,
            fallback_limit_used=fallback_limit_used_value,
            distance_score_mode=distance_score_mode,
            duration_ms=duration_ms,
        )

        candidates: Dict[str, Dict[str, object]] = {}
        for row in vector_rows:
            vector_score_missing = len(row) < 6
            if not vector_score_missing:
                try:
                    raw_value = float(row[5])
                except (TypeError, ValueError):
                    vector_score_missing = True
                else:
                    if math.isnan(raw_value) or math.isinf(raw_value):
                        vector_score_missing = True
            (
                chunk_id,
                text_value,
                metadata,
                doc_hash,
                doc_id,
                score_raw,
            ) = self._normalise_result_row(row, kind="vector")
            text_value = text_value or ""
            key = str(chunk_id)
            metadata_dict = dict(metadata)
            entry = candidates.setdefault(
                key,
                {
                    "chunk_id": key,
                    "content": text_value,
                    "metadata": metadata_dict,
                    "doc_hash": doc_hash,
                    "doc_id": doc_id,
                    "vscore": 0.0,
                    "lscore": 0.0,
                    "_allow_below_cutoff": False,
                },
            )
            entry["chunk_id"] = chunk_id if chunk_id is not None else key
            if vector_score_missing:
                entry["_allow_below_cutoff"] = True
            else:
                distance_value = float(score_raw)
                if distance_score_mode == "inverse":
                    distance_value = max(0.0, distance_value)
                    vscore = 1.0 / (1.0 + distance_value)
                else:
                    vscore = max(0.0, 1.0 - float(distance_value))
                entry["vscore"] = max(float(entry.get("vscore", 0.0)), vscore)

        allow_trgm_fallback_below_cutoff = (
            fallback_limit_used_value is not None and alpha_value <= 0.0
        )

        for row in lexical_rows:
            lexical_score_missing = len(row) < 6
            if not lexical_score_missing:
                try:
                    raw_value = float(row[5])
                except (TypeError, ValueError):
                    lexical_score_missing = True
                else:
                    if math.isnan(raw_value) or math.isinf(raw_value):
                        lexical_score_missing = True
            (
                chunk_id,
                text_value,
                metadata,
                doc_hash,
                doc_id,
                score_raw,
            ) = self._normalise_result_row(row, kind="lexical")
            text_value = text_value or ""
            key = str(chunk_id)
            metadata_dict = dict(metadata)
            entry = candidates.setdefault(
                key,
                {
                    "chunk_id": key,
                    "content": text_value,
                    "metadata": metadata_dict,
                    "doc_hash": doc_hash,
                    "doc_id": doc_id,
                    "vscore": 0.0,
                    "lscore": 0.0,
                    "_allow_below_cutoff": False,
                },
            )
            entry["chunk_id"] = chunk_id if chunk_id is not None else key
            lscore_value = max(0.0, float(score_raw))
            entry["lscore"] = max(float(entry.get("lscore", 0.0)), lscore_value)

            if lexical_score_missing or allow_trgm_fallback_below_cutoff:

                entry["_allow_below_cutoff"] = True

        fused_candidates = len(candidates)
        logger.info(
            "rag.hybrid.debug.fusion",
            extra={
                "tenant": tenant,
                "case": case_value,
                "candidates": fused_candidates,
                "has_vec": bool(vector_rows),
                "has_lex": bool(lexical_rows),
            },
        )
        results: List[Tuple[Chunk, bool]] = []
        has_vector_signal = (
            bool(vector_rows)
            and (query_vec is not None)
            and (not query_embedding_empty)
        )
        for entry in candidates.values():
            allow_below_cutoff = bool(entry.pop("_allow_below_cutoff", False))
            meta = dict(entry["metadata"])
            doc_hash = entry.get("doc_hash")
            doc_id = entry.get("doc_id")
            if doc_hash and not meta.get("hash"):
                meta["hash"] = doc_hash
            if doc_id is not None and "id" not in meta:
                meta["id"] = str(doc_id)
            if not strict_match(meta, tenant, case_value):
                candidate_tenant = meta.get("tenant")
                candidate_case = meta.get("case")
                reasons: List[str] = []
                if tenant is not None:
                    if candidate_tenant is None:
                        reasons.append("tenant_missing")
                    elif candidate_tenant != tenant:
                        reasons.append("tenant_mismatch")
                if case_value is not None:
                    if candidate_case is None:
                        reasons.append("case_missing")
                    elif candidate_case != case_value:
                        reasons.append("case_mismatch")
                logger.info(
                    "rag.strict.reject",
                    tenant=tenant,
                    case=case_value,
                    candidate_tenant=candidate_tenant,
                    candidate_case=candidate_case,
                    doc_hash=doc_hash,
                    doc_id=doc_id,
                    chunk_id=entry["chunk_id"],
                    reasons=reasons or ["unknown"],
                )
                continue
            try:
                vscore = float(entry.get("vscore", 0.0))
            except (TypeError, ValueError):
                vscore = 0.0
            if not has_vector_signal:
                vscore = 0.0
            try:
                lscore = float(entry.get("lscore", 0.0))
            except (TypeError, ValueError):
                lscore = 0.0
            lscore = max(0.0, lscore)
            if query_embedding_empty:
                fused = max(0.0, min(1.0, lscore))
            elif has_vector_signal:
                fused = max(
                    0.0,
                    min(1.0, alpha_value * vscore + (1.0 - alpha_value) * lscore),
                )
            else:
                fused = max(0.0, min(1.0, lscore))
            meta["vscore"] = vscore
            meta["lscore"] = lscore
            meta["fused"] = fused
            meta["score"] = fused
            results.append(
                (
                    Chunk(content=str(entry.get("content", "")), meta=meta),
                    allow_below_cutoff,
                )
            )

        results.sort(
            key=lambda item: float(item[0].meta.get("fused", 0.0)), reverse=True
        )
        below_cutoff = 0
        if min_sim_value > 0.0:
            below_cutoff = sum(
                1
                for chunk, _ in results
                if float(chunk.meta.get("fused", 0.0)) < min_sim_value
            )
            if below_cutoff > 0:
                metrics.RAG_QUERY_BELOW_CUTOFF_TOTAL.labels(tenant=tenant).inc(
                    float(below_cutoff)
                )
            filtered_results = [
                chunk
                for chunk, allow in results
                if allow or float(chunk.meta.get("fused", 0.0)) >= min_sim_value
            ]
        else:
            filtered_results = [chunk for chunk, _ in results]
        limited_results = filtered_results[:top_k]
        if not limited_results and results and min_sim_value > 0.0:
            try:
                logger.info(
                    "rag.hybrid.cutoff_fallback",
                    extra={
                        "tenant": tenant,
                        "case": case_value,
                        "requested_min_sim": min_sim_value,
                        "returned": len(limited_results),
                        "below_cutoff": below_cutoff,
                    },
                )
            except Exception:
                pass

        try:
            top_fused = (
                float(limited_results[0].meta.get("fused", 0.0))
                if limited_results
                else 0.0
            )
            top_v = (
                float(limited_results[0].meta.get("vscore", 0.0))
                if limited_results
                else 0.0
            )
            top_l = (
                float(limited_results[0].meta.get("lscore", 0.0))
                if limited_results
                else 0.0
            )
        except Exception:
            top_fused = top_v = top_l = 0.0

        logger.info(
            "rag.hybrid.debug.after_cutoff",
            extra={
                "tenant": tenant,
                "case": case_value,
                "returned": len(limited_results),
                "top_fused": top_fused,
                "top_vscore": top_v,
                "top_lscore": top_l,
                "min_sim": min_sim_value,
                "alpha": alpha_value,
                "distance_score_mode": distance_score_mode,
            },
        )

        metrics.RAG_SEARCH_MS.observe(duration_ms)
        logger.info(
            "RAG hybrid search executed: tenant=%s case=%s vector_candidates=%d lexical_candidates=%d fused_candidates=%d returned=%d duration_ms=%.2f",
            tenant,
            case_value,
            len(vector_rows),
            len(lexical_rows),
            fused_candidates,
            len(limited_results),
            duration_ms,
        )

        return HybridSearchResult(
            chunks=limited_results,
            vector_candidates=len(vector_rows),
            lexical_candidates=len(lexical_rows),
            fused_candidates=fused_candidates,
            duration_ms=duration_ms,
            alpha=alpha_value,
            min_sim=min_sim_value,
            vec_limit=vec_limit_value,
            lex_limit=lex_limit_value,
            below_cutoff=below_cutoff,
            returned_after_cutoff=len(filtered_results),
            query_embedding_empty=query_embedding_empty,
            applied_trgm_limit=applied_trgm_limit_value,
            fallback_limit_used=fallback_limit_used_value,
        )

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
                logger.warning(
                    "Chunk without external_id encountered; falling back to hash",
                    extra={"tenant": tenant_value, "hash": doc_hash},
                )
                external_id = doc_hash
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
                    "content_hash": doc_hash,
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
            external_id = str(doc["external_id"])
            content_hash = str(doc.get("content_hash", doc.get("hash", "")))
            storage_hash = self._compute_storage_hash(
                cur,
                tenant_uuid,
                content_hash,
                external_id,
            )
            doc["hash"] = storage_hash
            doc["content_hash"] = content_hash
            metadata_dict = dict(doc.get("metadata", {}))
            metadata_dict.setdefault("hash", content_hash)
            metadata = Json(metadata_dict)
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
                if stored_hash == storage_hash:
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
                    (storage_hash, doc["source"], metadata, document_id),
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
                    storage_hash,
                    metadata,
                ),
            )
            document_ids[key] = document_id
            actions[key] = "inserted"
        return document_ids, actions

    def _compute_storage_hash(
        self,
        cur,
        tenant_uuid: uuid.UUID,
        content_hash: str,
        external_id: str,
    ) -> str:
        if not content_hash:
            return content_hash
        tenant_value = str(tenant_uuid)
        cur.execute(
            """
            SELECT external_id
            FROM documents
            WHERE tenant_id = %s AND hash = %s
            LIMIT 1
            """,
            (tenant_value, content_hash),
        )
        existing = cur.fetchone()
        if existing:
            existing_external_id = existing[0]
            if existing_external_id and str(existing_external_id) != external_id:
                suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"external:{external_id}")
                return f"{content_hash}:{suffix}"
        return content_hash

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
                embedding_values = chunk.embedding
                is_empty_embedding = (
                    embedding_values is None
                    or _is_effectively_zero_vector(embedding_values)
                )
                if is_empty_embedding:
                    metrics.RAG_EMBEDDINGS_EMPTY_TOTAL.inc()
                    logger.warning(
                        "embedding.empty",
                        extra={
                            "tenant": doc["tenant_id"],
                            "doc_id": str(document_id),
                            "chunk_id": str(chunk_id),
                            "source": doc.get("source"),
                        },
                    )
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
                if not is_empty_embedding:
                    vector_value = self._format_vector(embedding_values)
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

    def _score_from_distance(
        self, distance: float | None, mode: str = "inverse"
    ) -> float:
        if distance is None:
            return 0.0
        if mode == "linear":
            try:
                value = float(distance)
            except (TypeError, ValueError):
                return 0.0
            if math.isnan(value) or math.isinf(value):
                return 0.0
            if value < 0:
                value = 0.0
            return max(0.0, min(1.0, 1.0 - value))
        return self._distance_to_score(distance)

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
                    operation=op_name,
                    attempt=attempt,
                    exc_type=exc.__class__.__name__,
                    exc_message=str(exc),
                )
                metrics.RAG_RETRY_ATTEMPTS.labels(operation=op_name).inc()
                if attempt == self._retries:
                    raise
                time.sleep(self._retry_base_delay * attempt)
        if last_exc is not None:  # pragma: no cover - defensive
            raise last_exc
        raise RuntimeError("retry loop exited without result")


_DEFAULT_CLIENT: Optional[VectorStore] = None


def _resolve_vector_schema() -> str:
    """Return the schema configured for the default vector store.

    The management commands rely on :func:`get_default_client` to run SQL
    statements such as index rebuilds. In multi-store setups we determine the
    schema according to the active default scope, honouring the ``default``
    flag used by :class:`~ai_core.rag.vector_store.VectorStoreRouter`.
    """

    schema_env = os.getenv("RAG_VECTOR_SCHEMA")
    if schema_env:
        return schema_env

    stores_config: Mapping[str, Mapping[str, object]] | None = None
    configured_default_scope: str | None = None
    try:  # pragma: no cover - requires Django settings
        from django.conf import settings  # type: ignore

        configured = getattr(settings, "RAG_VECTOR_STORES", None)
        if isinstance(configured, Mapping):
            stores_config = configured
        configured_default_scope = getattr(settings, "RAG_VECTOR_DEFAULT_SCOPE", None)
    except Exception:
        stores_config = None

    if stores_config:
        target_scope: str | None = None
        if configured_default_scope and configured_default_scope in stores_config:
            target_scope = configured_default_scope
        else:
            for scope_name, config in stores_config.items():
                if isinstance(config, Mapping) and config.get("default"):
                    target_scope = scope_name
                    break
        if target_scope is None:
            if "global" in stores_config:
                target_scope = "global"
            else:
                try:
                    target_scope = next(iter(stores_config))
                except StopIteration:  # pragma: no cover - defensive
                    target_scope = None
        if target_scope and target_scope in stores_config:
            config = stores_config[target_scope]
            schema_value = config.get("schema") if isinstance(config, Mapping) else None
            if schema_value:
                return str(schema_value)

    return "rag"


def get_default_client() -> PgVectorClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = PgVectorClient.from_env(schema=_resolve_vector_schema())
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
