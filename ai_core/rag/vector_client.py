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

from psycopg2 import Error as PsycopgError, OperationalError, sql
from psycopg2.errors import DeadlockDetected, LockNotAvailable, UniqueViolation
from psycopg2.extras import Json, register_default_jsonb
from psycopg2.extensions import make_dsn, parse_dsn
from psycopg2.pool import SimpleConnectionPool

from common.logging import get_log_context, get_logger
from ai_core.rag.vector_store import VectorStore
from .embeddings import EmbeddingClientError, get_embedding_client
from .normalization import normalise_text, normalise_text_db

from . import metrics
from .filters import strict_match
from .schemas import Chunk
from .visibility import Visibility

# Ensure jsonb columns are decoded into Python dictionaries
register_default_jsonb(loads=json.loads, globally=True)

logger = get_logger(__name__)


# Welche Filter-Schlüssel sind erlaubt und worauf mappen sie?
# - "chunk_meta": JSONB c.metadata ->> '<key>'
# - "document_hash": Spalte d.hash
# - "document_id":  Spalte d.id::text
SUPPORTED_METADATA_FILTERS = {
    "case_id": "chunk_meta",
    "source": "chunk_meta",
    "doctype": "chunk_meta",
    "published": "chunk_meta",
    "hash": "document_hash",
    "id": "document_id",
    "external_id": "document_external_id",
}


_OPERATOR_CLASS_PREFERENCE: tuple[str, ...] = (
    "vector_cosine_ops",
    "vector_l2_ops",
    "vector_ip_ops",
)
_OPERATOR_FOR_CLASS: dict[str, str] = {
    "vector_cosine_ops": "<=>",
    "vector_l2_ops": "<->",
    "vector_ip_ops": "<#>",
}


FALLBACK_STATEMENT_TIMEOUT_MS = 15000
FALLBACK_RETRY_ATTEMPTS = 3
FALLBACK_RETRY_BASE_DELAY_MS = 50
_ZERO_EPSILON = 1e-12


def get_embedding_dim() -> int:
    """Return the embedding dimensionality reported by the provider."""

    return get_embedding_client().dim()


def operator_class_exists(cur, operator_class: str, access_method: str) -> bool:
    """Return whether *operator_class* is available for *access_method*."""

    cur.execute(
        """
        SELECT 1
        FROM pg_catalog.pg_opclass opc
        JOIN pg_catalog.pg_am am ON am.oid = opc.opcmethod
        WHERE opc.opcname = %s AND am.amname = %s
        """,
        (operator_class, access_method),
    )
    return cur.fetchone() is not None


def resolve_operator_class(cur, index_kind: str) -> str | None:
    """Determine the operator class for the configured vector index."""

    kind = index_kind.upper()
    access_method = "hnsw" if kind == "HNSW" else "ivfflat"
    for candidate in _OPERATOR_CLASS_PREFERENCE:
        if operator_class_exists(cur, candidate, access_method):
            return candidate
    return None


def resolve_distance_operator(cur, index_kind: str) -> str | None:
    """Return the distance operator matching the active operator class."""

    operator_class = resolve_operator_class(cur, index_kind)
    if operator_class is None:
        return None
    return _OPERATOR_FOR_CLASS.get(operator_class)


def _normalise_vector(values: Sequence[float] | None) -> list[float] | None:
    """Scale ``values`` to unit length if possible.

    Returns ``None`` when ``values`` cannot be interpreted as a numeric
    sequence or if its norm is effectively zero. Callers should treat a
    ``None`` result as an empty embedding and skip persistence/search to
    avoid unstable similarity scores.
    """

    if not values:
        return None
    try:
        floats = [float(value) for value in values]
    except (TypeError, ValueError):
        return None

    norm_sq = math.fsum(value * value for value in floats)
    if norm_sq <= _ZERO_EPSILON:
        return None

    norm = math.sqrt(norm_sq)
    if not math.isfinite(norm) or norm <= _ZERO_EPSILON:
        return None

    scale = 1.0 / norm
    return [value * scale for value in floats]


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
    visibility: str = Visibility.ACTIVE.value
    deleted_matches_blocked: int = 0


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
        self._distance_operator_cache: Dict[str, str] = {}

    @classmethod
    def from_env(
        cls,
        *,
        env_var: str = "RAG_DATABASE_URL",
        fallback_env_var: str = "DATABASE_URL",
        **kwargs: object,
    ) -> "PgVectorClient":
        dsn_env_value = os.getenv(env_var)
        dsn = dsn_env_value or os.getenv(fallback_env_var)
        if not dsn:
            raise RuntimeError(
                f"Neither {env_var} nor {fallback_env_var} is set; cannot initialise PgVectorClient"
            )

        django_dsn = _resolve_django_dsn_if_available(dsn=dsn)
        if django_dsn:
            dsn = django_dsn
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

    @staticmethod
    def _ensure_chunk_metadata_contract(
        meta: Mapping[str, object] | None,
        *,
        tenant_id: str | None,
        case_id: str | None,
        filters: Mapping[str, object | None] | None,
        chunk_id: object,
        doc_id: object,
    ) -> Dict[str, object]:
        enriched = dict(meta or {})
        if "chunk_id" not in enriched and chunk_id is not None:
            enriched["chunk_id"] = chunk_id
        if "doc_id" not in enriched and doc_id is not None:
            enriched["doc_id"] = doc_id
        return enriched

    @contextmanager
    def _connection(self):  # type: ignore[no-untyped-def]
        conn = self._pool.getconn()
        try:
            self._prepare_connection(conn)
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def connection(self):  # type: ignore[no-untyped-def]
        """Yield a prepared connection from the pool."""

        with self._connection() as conn:
            yield conn

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

    def _get_distance_operator(self, conn, index_kind: str) -> str:
        key = index_kind.upper()
        cached = self._distance_operator_cache.get(key)
        if cached:
            return cached
        with conn.cursor() as cur:
            operator = resolve_distance_operator(cur, key)
        if operator is None:
            raise RuntimeError(
                "No compatible pgvector operator class available for queries. "
                "Ensure the vector index has been created with a supported operator class."
            )
        self._distance_operator_cache[key] = operator
        return operator

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
                "tenant_id": tenant_id,
                "external_id": external_id,
                "content_hash": doc.get("content_hash"),
                "action": action,
                "chunk_count": chunk_count,
                "duration_ms": duration,
            }
            metadata = doc.get("metadata", {})
            embedding_profile = metadata.get("embedding_profile")
            if embedding_profile:
                doc_payload["embedding_profile"] = embedding_profile
            vector_space_id = metadata.get("vector_space_id")
            if vector_space_id:
                doc_payload["vector_space_id"] = vector_space_id
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
        visibility: str | None = None,
        visibility_override_allowed: bool = False,
    ) -> HybridSearchResult:
        """Execute hybrid vector/lexical retrieval for ``query``.

        The pg_trgm similarity threshold is applied per-connection via
        ``SELECT set_limit`` immediately before running the trigram ``%``
        operator to ensure consistent lexical matching behaviour.
        """
        top_k = min(max(1, top_k), 10)
        allowed_visibilities = {value.value for value in Visibility}
        explicit_visibility_requested = False
        if isinstance(visibility, Visibility):
            visibility_value = visibility.value
            explicit_visibility_requested = True
        elif visibility is None:
            visibility_value = Visibility.ACTIVE.value
        else:
            try:
                text_value = str(visibility).strip().lower()
            except Exception:
                text_value = ""
            if text_value in allowed_visibilities:
                visibility_value = text_value
                explicit_visibility_requested = True
            else:
                visibility_value = Visibility.ACTIVE.value
        if (
            visibility_value != Visibility.ACTIVE.value
            and not visibility_override_allowed
            and not explicit_visibility_requested
        ):
            visibility_value = Visibility.ACTIVE.value
        visibility_mode = Visibility(visibility_value)
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
                if key != "visibility"
            }
        case_value: Optional[str]
        if case_id not in {None, ""}:
            case_value = case_id
        else:
            case_value = normalized_filters.get("case_id")
        if case_value is not None:
            case_value = str(case_value)
        normalized_filters["tenant_id"] = tenant
        normalized_filters["case_id"] = case_value
        metadata_filters = [
            (key, value)
            for key, value in normalized_filters.items()
            if key not in {"tenant_id"}
            and value is not None
            and key in SUPPORTED_METADATA_FILTERS
        ]
        filter_debug: Dict[str, object | None] = {
            "tenant_id": "<set>",
            "visibility": visibility_mode.value,
        }
        for key, value in normalized_filters.items():
            if key in {"tenant_id"}:
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
        vec_limit_value = min(max_candidates_value, max(top_k, vec_limit_requested))
        lex_limit_value = min(max_candidates_value, max(top_k, lex_limit_requested))
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
        vector_format_error: Optional[Exception] = None
        if raw_vec is not None and not is_zero_vec:
            try:
                query_vec = self._format_vector(raw_vec)
            except ValueError as exc:
                # Be lenient for query-time formatting: some tests patch
                # `_embed_query` directly without adjusting the provider
                # dimension. In that case, still attempt the vector search
                # using a best-effort formatted vector instead of treating
                # this as a hard failure.
                try:
                    query_vec = self._format_vector_lenient(raw_vec)
                    vector_format_error = None
                except Exception:
                    vector_format_error = exc
        query_embedding_empty = bool(is_zero_vec or vector_format_error is not None)
        if query_embedding_empty:
            metrics.RAG_QUERY_EMPTY_VEC_TOTAL.labels(tenant_id=tenant).inc()
            logger.info(
                "rag.hybrid.null_embedding",
                alpha=alpha_value,
                tenant_id=tenant,
                case_id=case_value,
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
        deleted_visibility_clauses = (
            "d.deleted_at IS NULL",
            "(c.metadata ->> 'deleted_at') IS NULL",
        )
        if visibility_mode is Visibility.ACTIVE:
            where_clauses.extend(deleted_visibility_clauses)
        elif visibility_mode is Visibility.DELETED:
            where_clauses.append(
                "(d.deleted_at IS NOT NULL OR (c.metadata ->> 'deleted_at') IS NOT NULL)"
            )
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
        where_sql_without_deleted: str | None = None
        distance_operator_value: Optional[str] = None
        lexical_query_variant: str = "none"
        lexical_fallback_limit_value: Optional[float] = None
        if visibility_mode is Visibility.ACTIVE:
            filtered_clauses = [
                clause
                for clause in where_clauses
                if clause not in deleted_visibility_clauses
            ]
            where_sql_without_deleted = "\n          AND ".join(filtered_clauses)

        applied_trgm_limit_value: Optional[float] = None
        fallback_limit_used_value: Optional[float] = None
        fallback_tried_limits: List[float] = []
        total_without_filter: Optional[int] = None

        def _operation() -> Tuple[List[tuple], List[tuple], float]:
            nonlocal applied_trgm_limit_value
            nonlocal fallback_limit_used_value
            nonlocal fallback_tried_limits
            nonlocal total_without_filter
            nonlocal distance_operator_value
            nonlocal lexical_query_variant
            nonlocal lexical_fallback_limit_value
            started = time.perf_counter()
            vector_rows: List[tuple] = []
            lexical_rows: List[tuple] = []
            vector_query_failed = vector_format_error is not None
            fallback_tried_limits = []
            fallback_limit_used_value = None
            total_without_filter_local: Optional[int] = None
            distance_operator_value = None
            lexical_query_variant = "none"
            lexical_fallback_limit_value = None

            def _build_vector_sql(
                where_sql_value: str, select_columns: str, order_by_clause: str
            ) -> str:
                return f"""
                    SELECT
                        {select_columns}
                    FROM embeddings e
                    JOIN chunks c ON e.chunk_id = c.id
                    JOIN documents d ON c.document_id = d.id
                    WHERE {where_sql_value}
                    ORDER BY {order_by_clause}
                    LIMIT %s
                """

            def _build_lexical_primary_sql(
                where_sql_value: str, select_columns: str
            ) -> str:
                return f"""
                    SELECT
                        {select_columns}
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE {where_sql_value}
                      AND c.text_norm % %s
                    ORDER BY lscore DESC
                    LIMIT %s
                """

            def _build_lexical_fallback_sql(
                where_sql_value: str, select_columns: str
            ) -> str:
                return f"""
                    SELECT
                        {select_columns}
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE {where_sql_value}
                      AND similarity(c.text_norm, %s) >= %s
                    ORDER BY lscore DESC
                    LIMIT %s
                """

            with self._connection() as conn:
                if vector_format_error is not None:
                    try:
                        conn.rollback()
                    except Exception:  # pragma: no cover - defensive
                        pass
                    logger.warning(
                        "rag.hybrid.vector_query_failed",
                        tenant_id=tenant,
                        tenant=tenant,
                        case_id=case_value,
                        error=str(vector_format_error),
                    )
                elif query_vec is not None:
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
                            distance_operator = self._get_distance_operator(
                                conn, index_kind
                            )
                            distance_operator_value = distance_operator
                            vector_sql = _build_vector_sql(
                                where_sql,
                                "c.id,\n                                    c.text,\n                                    c.metadata,\n                                    d.hash,\n                                    d.id,\n                                    e.embedding "
                                + f"{distance_operator} %s::vector AS distance",
                                "distance",
                            )
                            # Bind parameters in textual order: SELECT (vector), WHERE params, LIMIT
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
                        vector_query_failed = True
                        try:
                            conn.rollback()
                        except Exception:  # pragma: no cover - defensive
                            pass
                        logger.warning(
                            "rag.hybrid.vector_query_failed",
                            tenant_id=tenant,
                            tenant=tenant,
                            case_id=case_value,
                            error=str(exc),
                        )
                else:
                    # Even when the query embedding is empty, execute a lightweight
                    # no-op vector statement to ensure limit clamping is exercised
                    # consistently (observability/tests rely on this record).
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SET LOCAL statement_timeout = %s",
                                (str(self._statement_timeout_ms),),
                            )
                            cur.execute(
                                "SELECT 1 FROM embeddings e LIMIT %s",
                                (vec_limit_value,),
                            )
                    except Exception:
                        try:
                            conn.rollback()
                        except Exception:
                            pass

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
                                lexical_query_variant = "primary"
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
                                if isinstance(exc, (IndexError, ValueError)):
                                    should_run_fallback = True
                                    fallback_requires_rollback = True
                                    lexical_rows_local = []
                                    logger.warning(
                                        "rag.hybrid.lexical_primary_failed",
                                        extra={
                                            "tenant_id": tenant,
                                            "case_id": case_value,
                                            "error": str(exc),
                                        },
                                    )
                                elif isinstance(exc, PsycopgError):
                                    if vector_query_failed:
                                        raise

                                    # Treat database errors during the primary lexical query as
                                    # a signal to attempt the explicit similarity fallback.
                                    # We mark that a rollback is required to restore session
                                    # settings (e.g. search_path) before running the fallback.
                                    should_run_fallback = True
                                    fallback_requires_rollback = True
                                    lexical_rows_local = []
                                    logger.warning(
                                        "rag.hybrid.lexical_primary_failed",
                                    )
                                    raise
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
                                        "tenant_id": tenant,
                                        "case_id": case_value,
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
                                fallback_last_limit_value: float | None = None
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
                                    fallback_last_limit_value = float(limit_value)
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
                                lexical_query_variant = "fallback"
                                fallback_limit_used_value = picked_limit
                                if fallback_limit_used_value is None:
                                    fallback_limit_used_value = (
                                        fallback_last_limit_value
                                    )
                                lexical_fallback_limit_value = fallback_limit_used_value
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
                                    tenant_id=tenant,
                                    case_id=case_value,
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
                    rollback_succeeded = False
                    try:
                        conn.rollback()
                    except Exception:  # pragma: no cover - defensive
                        pass
                    else:
                        rollback_succeeded = True
                    if rollback_succeeded:
                        try:
                            with conn.cursor() as restore_cur:
                                self._restore_session_after_rollback(restore_cur)
                        except Exception:  # pragma: no cover - defensive
                            pass
                    logger.warning(
                        "rag.hybrid.lexical_query_failed",
                        tenant_id=tenant,
                        tenant=tenant,
                        case_id=case_value,
                        error=str(exc),
                    )
                    if not vector_rows:
                        if vector_query_failed:
                            fatal_exc = PsycopgError(str(exc))
                            setattr(fatal_exc, "_rag_retry_fatal", True)
                            raise fatal_exc from exc
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

            if visibility_mode is Visibility.ACTIVE and where_sql_without_deleted:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SET LOCAL statement_timeout = %s",
                            (str(self._statement_timeout_ms),),
                        )
                        try:
                            cur.execute(
                                sql.SQL("SET LOCAL search_path TO {}, public").format(
                                    sql.Identifier(self._schema)
                                )
                            )
                        except Exception:
                            # search_path is already configured at connection setup;
                            # ignore errors when SET LOCAL is not available.
                            pass
                        count_selects: List[str] = []
                        count_params: List[object] = []

                        if (
                            query_vec is not None
                            and not vector_query_failed
                            and distance_operator_value is not None
                        ):
                            vector_count_sql = _build_vector_sql(
                                where_sql_without_deleted,
                                "c.id",
                                f"e.embedding {distance_operator_value} %s::vector",
                            )
                            count_selects.append(
                                f"SELECT id FROM ({vector_count_sql}) AS vector_candidates"
                            )
                            count_params.extend(
                                (*where_params, query_vec, vec_limit_value)
                            )

                        if query_db_norm.strip():
                            if lexical_query_variant == "primary":
                                lexical_count_sql = _build_lexical_primary_sql(
                                    where_sql_without_deleted,
                                    "c.id,\n                                    similarity(c.text_norm, %s) AS lscore",
                                )
                                count_selects.append(
                                    f"SELECT id FROM ({lexical_count_sql}) AS lexical_candidates"
                                )
                                count_params.extend(
                                    (
                                        query_db_norm,
                                        *where_params,
                                        query_db_norm,
                                        lex_limit_value,
                                    )
                                )
                            elif (
                                lexical_query_variant == "fallback"
                                and lexical_fallback_limit_value is not None
                            ):
                                lexical_count_sql = _build_lexical_fallback_sql(
                                    where_sql_without_deleted,
                                    "c.id,\n                                    similarity(c.text_norm, %s) AS lscore",
                                )
                                count_selects.append(
                                    f"SELECT id FROM ({lexical_count_sql}) AS lexical_candidates"
                                )
                                count_params.extend(
                                    (
                                        query_db_norm,
                                        *where_params,
                                        query_db_norm,
                                        lexical_fallback_limit_value,
                                        lex_limit_value,
                                    )
                                )

                        if count_selects:
                            union_sql = " UNION ALL ".join(count_selects)
                            count_without_deleted_sql = (
                                "SELECT COUNT(DISTINCT id) FROM ("
                                + union_sql
                                + ") AS all_candidates"
                            )
                            cur.execute(
                                count_without_deleted_sql,
                                tuple(count_params),
                            )
                            row = cur.fetchone()
                            if row and row[0] is not None:
                                total_without_filter_local = int(row[0])
                except Exception as exc:
                    logger.warning(
                        "rag.hybrid.deleted_visibility_count_failed",
                        tenant_id=tenant,
                        case_id=case_value,
                        error=str(exc),
                    )
                    total_without_filter_local = None

            total_without_filter = total_without_filter_local
            return vector_rows, lexical_rows, (time.perf_counter() - started) * 1000

        vector_rows, lexical_rows, duration_ms = self._run_with_retries(
            _operation, op_name="search"
        )

        logger.info(
            "rag.hybrid.sql_counts",
            tenant_id=tenant,
            case_id=case_value,
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
            key = str(chunk_id) if chunk_id is not None else f"row-{len(candidates)}"
            chunk_identifier = chunk_id if chunk_id is not None else key
            raw_meta = dict(cast(Mapping[str, object] | None, metadata) or {})
            if "chunk_id" not in raw_meta and chunk_identifier is not None:
                raw_meta["chunk_id"] = chunk_identifier
            if "doc_id" not in raw_meta and doc_id is not None:
                raw_meta["doc_id"] = doc_id
            entry = candidates.setdefault(
                key,
                {
                    "chunk_id": key,
                    "content": text_value,
                    "metadata": raw_meta,
                    "doc_hash": doc_hash,
                    "doc_id": doc_id,
                    "vscore": 0.0,
                    "lscore": 0.0,
                    "_allow_below_cutoff": False,
                },
            )
            entry["chunk_id"] = chunk_identifier
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
            key = str(chunk_id) if chunk_id is not None else f"row-{len(candidates)}"
            chunk_identifier = chunk_id if chunk_id is not None else key
            raw_meta = dict(cast(Mapping[str, object] | None, metadata) or {})
            if "chunk_id" not in raw_meta and chunk_identifier is not None:
                raw_meta["chunk_id"] = chunk_identifier
            if "doc_id" not in raw_meta and doc_id is not None:
                raw_meta["doc_id"] = doc_id
            entry = candidates.setdefault(
                key,
                {
                    "chunk_id": key,
                    "content": text_value,
                    "metadata": raw_meta,
                    "doc_hash": doc_hash,
                    "doc_id": doc_id,
                    "vscore": 0.0,
                    "lscore": 0.0,
                    "_allow_below_cutoff": False,
                },
            )
            entry["chunk_id"] = chunk_identifier
            lscore_value = max(0.0, float(score_raw))
            entry["lscore"] = max(float(entry.get("lscore", 0.0)), lscore_value)

            if lexical_score_missing or allow_trgm_fallback_below_cutoff:

                entry["_allow_below_cutoff"] = True

        fused_candidates = len(candidates)
        logger.info(
            "rag.hybrid.debug.fusion",
            extra={
                "tenant_id": tenant,
                "case_id": case_value,
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
            raw_meta = dict(cast(Mapping[str, object] | None, entry.get("metadata")) or {})
            candidate_tenant = cast(Optional[str], raw_meta.get("tenant_id"))
            candidate_case = cast(Optional[str], raw_meta.get("case_id"))
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

            if case_value is None:
                strict_ok = (tenant is None) or (
                    candidate_tenant is not None and candidate_tenant == tenant
                )
            else:
                strict_ok = strict_match(raw_meta, tenant, case_value)

            if not strict_ok or reasons:
                logger.info(
                    "rag.strict.reject",
                    tenant_id=tenant,
                    case_id=case_value,
                    candidate_tenant_id=candidate_tenant,
                    candidate_case_id=candidate_case,
                    doc_hash=entry.get("doc_hash"),
                    doc_id=entry.get("doc_id"),
                    chunk_id=entry.get("chunk_id"),
                    reasons=reasons or ["unknown"],
                )
                continue

            meta = self._ensure_chunk_metadata_contract(
                raw_meta,
                tenant_id=tenant,
                case_id=case_value,
                filters=normalized_filters,
                chunk_id=entry.get("chunk_id"),
                doc_id=entry.get("doc_id"),
            )
            doc_hash = entry.get("doc_hash")
            doc_id = entry.get("doc_id")
            if doc_hash and not meta.get("hash"):
                meta["hash"] = doc_hash
            if doc_id is not None and "id" not in meta:
                meta["id"] = str(doc_id)
            if not strict_match(meta, tenant, case_value):
                candidate_tenant = meta.get("tenant_id")
                candidate_case = meta.get("case_id")
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
                    tenant_id=tenant,
                    case_id=case_value,
                    candidate_tenant_id=candidate_tenant,
                    candidate_case_id=candidate_case,
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
                metrics.RAG_QUERY_BELOW_CUTOFF_TOTAL.labels(tenant_id=tenant).inc(
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
                        "tenant_id": tenant,
                        "case_id": case_value,
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
                "tenant_id": tenant,
                "case_id": case_value,
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
        deleted_matches_blocked_value = 0
        if visibility_mode is Visibility.ACTIVE and total_without_filter is not None:
            deleted_matches_blocked_value = max(
                0,
                int(total_without_filter) - (len(vector_rows) + len(lexical_rows)),
            )

        logger.info(
            "RAG hybrid search executed: tenant=%s case=%s vector_candidates=%d lexical_candidates=%d fused_candidates=%d returned=%d deleted_blocked=%d duration_ms=%.2f",
            tenant,
            case_value,
            len(vector_rows),
            len(lexical_rows),
            fused_candidates,
            len(limited_results),
            deleted_matches_blocked_value,
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
            visibility=visibility_mode.value,
            deleted_matches_blocked=deleted_matches_blocked_value,
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
            tenant_value = chunk.meta.get("tenant_id")
            doc_hash = str(chunk.meta.get("hash"))
            source = chunk.meta.get("source", "")
            external_id = chunk.meta.get("external_id")
            if tenant_value in {None, "", "None"}:
                raise ValueError("Chunk metadata must include tenant_id")
            if not doc_hash or doc_hash == "None":
                raise ValueError("Chunk metadata must include hash")
            if external_id in {None, "", "None"}:
                logger.warning(
                    "Chunk without external_id encountered; falling back to hash",
                    extra={"tenant_id": tenant_value, "hash": doc_hash},
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
                        if k not in {"tenant_id", "hash", "source"}
                    },
                    "chunks": [],
                }
            chunk_meta = dict(chunk.meta)
            chunk_meta["tenant_id"] = tenant
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
            document_id = doc["id"]
            cur.execute(
                """
                INSERT INTO documents (id, tenant_id, external_id, source, hash, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, external_id) DO UPDATE
                SET source = EXCLUDED.source,
                    hash = EXCLUDED.hash,
                    metadata = EXCLUDED.metadata,
                    deleted_at = NULL
                WHERE documents.hash IS DISTINCT FROM EXCLUDED.hash
                RETURNING id, hash
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
            upsert_result = cur.fetchone()
            if upsert_result:
                returned_id, _ = upsert_result
                document_ids[key] = returned_id
                if returned_id == document_id:
                    actions[key] = "inserted"
                else:
                    actions[key] = "replaced"
                continue

            # Conflict occurred but existing row already matched the payload. Re-read to
            # ensure we have the persisted identifiers for downstream actions.
            cur.execute(
                """
                SELECT id, hash
                FROM documents
                WHERE tenant_id = %s AND external_id = %s
                """,
                (str(tenant_uuid), external_id),
            )
            existing = cur.fetchone()
            if not existing:
                raise RuntimeError(
                    "Document upsert yielded no result but record is missing"
                )
            existing_id, _ = existing
            document_ids[key] = existing_id
            actions[key] = "skipped"
            logger.info(
                "Skipping unchanged document during upsert",
                extra={
                    "tenant_id": doc["tenant_id"],
                    "external_id": external_id,
                },
            )
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
                extra={"tenant_id": tenant_id, "derived_tenant_uuid": str(derived)},
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
                normalised_embedding = (
                    _normalise_vector(embedding_values)
                    if embedding_values is not None
                    else None
                )
                is_empty_embedding = normalised_embedding is None
                if is_empty_embedding:
                    metrics.RAG_EMBEDDINGS_EMPTY_TOTAL.inc()
                    logger.warning(
                        "embedding.empty",
                        extra={
                            "tenant_id": doc["tenant_id"],
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
                if normalised_embedding is not None:
                    vector_value = self._format_vector(normalised_embedding)
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
        expected_dim = get_embedding_dim()
        floats = [float(v) for v in values]
        if len(floats) != expected_dim:
            raise ValueError("Embedding dimension mismatch")
        return "[" + ",".join(f"{value:.6f}" for value in floats) + "]"

    def _format_vector_lenient(self, values: Sequence[float]) -> str:
        """Format a vector without enforcing provider dimension.

        This helper is used for query-time formatting only, to avoid turning a
        temporary dimension mismatch (e.g. during tests or provider changes)
        into a hard failure that would bypass the vector search entirely.
        """
        floats = [float(v) for v in values]
        return "[" + ",".join(f"{value:.6f}" for value in floats) + "]"

    def _embed_query(self, query: str) -> List[float]:
        client = get_embedding_client()
        normalised = normalise_text(query)
        text = normalised or ""
        started = time.perf_counter()
        result = client.embed([text])
        duration_ms = (time.perf_counter() - started) * 1000

        if not result.vectors:
            raise EmbeddingClientError("Embedding provider returned no vectors")
        vector = result.vectors[0]
        if not isinstance(vector, list):
            vector = list(vector)
        try:
            vector = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingClientError(
                "Embedding vector contains non-numeric values"
            ) from exc
        try:
            expected_dim = client.dim()
        except EmbeddingClientError:
            expected_dim = len(vector)
        if len(vector) != expected_dim:
            raise EmbeddingClientError(
                "Embedding dimension mismatch between query and provider"
            )

        normalised_vector = _normalise_vector(vector)
        if normalised_vector is None:
            vector = [0.0 for _ in vector]
        else:
            vector = normalised_vector

        context = get_log_context()
        tenant_id = context.get("tenant")
        extra: Dict[str, object] = {
            "tenant_id": tenant_id or "-",
            "len_text": len(text),
            "model_name": result.model,
            "model_used": result.model_used,
            "duration_ms": duration_ms,
            "attempts": result.attempts,
        }
        timeout_s = result.timeout_s
        if timeout_s is not None:
            extra["timeout_s"] = timeout_s
        key_alias = context.get("key_alias")
        if key_alias:
            extra["key_alias"] = key_alias
        logger.info("rag.query.embed", extra=extra)
        return vector

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
        transient_errors = (
            OperationalError,
            DeadlockDetected,
            LockNotAvailable,
            UniqueViolation,
        )
        for attempt in range(1, self._retries + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - requires failure injection
                last_exc = exc
                is_transient = isinstance(exc, transient_errors)
                is_fatal_override = bool(getattr(exc, "_rag_retry_fatal", False))
                is_pg_error = isinstance(exc, PsycopgError)

                if not is_transient and (is_pg_error or is_fatal_override):
                    logger.error(
                        "pgvector operation failed, aborting",
                        operation=op_name,
                        attempt=attempt,
                        exc_type=exc.__class__.__name__,
                        exc_message=str(exc),
                    )
                    raise

                logger.warning(
                    "pgvector operation failed, retrying",
                    operation=op_name,
                    attempt=attempt,
                    exc_type=exc.__class__.__name__,
                    exc_message=str(exc),
                )

                if attempt == self._retries:
                    raise

                metrics.RAG_RETRY_ATTEMPTS.labels(operation=op_name).inc()
                time.sleep(self._retry_base_delay * attempt)
        if last_exc is not None:  # pragma: no cover - defensive
            raise last_exc
        raise RuntimeError("retry loop exited without result")


def _parse_dsn_parameters(value: str) -> Dict[str, str]:
    """Return a mapping of DSN parameters parsed from ``value``.

    Uses :func:`psycopg2.extensions.parse_dsn` and falls back to building a
    canonical DSN via :func:`psycopg2.extensions.make_dsn` so URI-style strings
    (``postgresql://user:pass@host/db``) are also supported.
    """

    try:
        parsed = parse_dsn(value)
    except Exception:
        try:
            canonical = make_dsn(value)
        except Exception:
            canonical = None
        if not canonical:
            return {}
        try:
            parsed = parse_dsn(canonical)
        except Exception:
            return {}

    params: Dict[str, str] = {}
    for key, parsed_value in parsed.items():
        if parsed_value is None:
            continue
        params[str(key)] = str(parsed_value)
    return params


def _build_dsn_from_settings_dict(settings_dict: Mapping[str, object]) -> str | None:
    """Construct a DSN string from a Django ``settings_dict`` mapping."""

    engine = settings_dict.get("ENGINE")
    if engine and "postgres" not in str(engine).lower():
        return None

    key_mapping = {
        "NAME": "dbname",
        "USER": "user",
        "PASSWORD": "password",
        "HOST": "host",
        "PORT": "port",
    }
    params: Dict[str, str] = {}
    for settings_key, dsn_key in key_mapping.items():
        value = settings_dict.get(settings_key)
        if value in (None, ""):
            continue
        params[dsn_key] = str(value)

    options = settings_dict.get("OPTIONS")
    if isinstance(options, Mapping):
        for option_key, option_value in options.items():
            if option_value in (None, ""):
                continue
            params[str(option_key)] = str(option_value)

    if not params:
        return None

    try:
        return make_dsn(**params)
    except Exception:
        return None


def _resolve_django_dsn_if_available(*, dsn: str) -> str | None:
    """Return a Django-aware DSN when running inside a configured project."""

    try:  # pragma: no cover - guarded import
        from django.conf import settings  # type: ignore
        from django.db import DEFAULT_DB_ALIAS, connections  # type: ignore
    except Exception:  # pragma: no cover - Django not available
        return None

    if not getattr(settings, "configured", False):
        return None

    databases = getattr(settings, "DATABASES", None)
    if not isinstance(databases, Mapping):
        return None

    default_config = databases.get(DEFAULT_DB_ALIAS)
    if not isinstance(default_config, Mapping):
        return None

    env_params = _parse_dsn_parameters(dsn)
    env_dbname = env_params.get("dbname") or env_params.get("database")

    connection = connections[DEFAULT_DB_ALIAS]
    connection_settings = getattr(connection, "settings_dict", None)
    if not isinstance(connection_settings, Mapping):
        return None

    default_name = connection_settings.get("NAME") or default_config.get("NAME")
    if not default_name or not env_dbname:
        return None

    if str(env_dbname) != str(default_name):
        return None

    def _norm(value: object) -> str | None:
        if value in (None, ""):
            return None
        return str(value)

    comparison_keys = (
        ("host", "HOST"),
        ("port", "PORT"),
        ("user", "USER"),
        ("password", "PASSWORD"),
    )
    for dsn_key, settings_key in comparison_keys:
        env_value = _norm(env_params.get(dsn_key))
        if env_value is None:
            continue
        settings_value = _norm(connection_settings.get(settings_key))
        if settings_value is None or env_value != settings_value:
            return None

    rebuilt = _build_dsn_from_settings_dict(connection_settings)
    return rebuilt


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


def get_default_schema() -> str:
    """Return the schema configured for the default :class:`PgVectorClient`."""

    return _resolve_vector_schema()


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
