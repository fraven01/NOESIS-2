from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
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
from .deduplication import compute_document_embedding, find_near_duplicate
from .embeddings import EmbeddingClientError, get_embedding_client
from .hashing import compute_storage_hash
from .metadata_handler import MetadataHandler
from .normalization import normalise_text, normalise_text_db
from .parents import limit_parent_payload
from .query_builder import (
    build_lexical_fallback_sql,
    build_lexical_primary_sql,
    build_vector_sql,
)
from .lexical_search import run_lexical_search
from .routing_rules import is_collection_routing_enabled
from .hybrid_fusion import fuse_candidates
from .vector_search import run_vector_search
from .vector_utils import (
    _normalise_vector,
    embed_query,
    format_vector,
    format_vector_lenient,
)

from . import metrics
from .schemas import Chunk
from .visibility import Visibility

# Ensure jsonb columns are decoded into Python dictionaries
register_default_jsonb(loads=json.loads, globally=True)

logger = get_logger(__name__)

logger.info(
    "module_loaded",
    extra={"module": __name__, "path": os.path.abspath(__file__)},
)


_HYDE_PROMPT_TEMPLATE = (
    "Write a short, factual passage that would answer the question. "
    "Keep it concise and information-dense.\n\n"
    "Question:\n{query}\n\n"
    "Passage:"
)

LIFECYCLE_ACTIVE = "active"
LIFECYCLE_RETIRED = "retired"
LIFECYCLE_DELETED = "deleted"


def _normalize_lifecycle_state(value: object | None) -> str:
    if isinstance(value, str):
        candidate = value.strip().lower()
    elif value is None:
        candidate = LIFECYCLE_ACTIVE
    else:
        candidate = str(value).strip().lower()
    if candidate in {LIFECYCLE_ACTIVE, LIFECYCLE_RETIRED, LIFECYCLE_DELETED}:
        return candidate
    return LIFECYCLE_ACTIVE


# Welche Filter-Schlüssel sind erlaubt und worauf mappen sie?
# - "chunk_meta": JSONB c.metadata ->> '<key>'
# - "document_hash": Spalte d.hash
# - "document_id":  Spalte d.id::text
SUPPORTED_METADATA_FILTERS = {
    "case_id": "chunk_meta",
    "source": "chunk_meta",
    "doctype": "chunk_meta",
    "published": "chunk_meta",
    "document_version_id": "chunk_meta",
    "is_latest": "chunk_meta",
    "hash": "document_hash",
    "id": "document_id",
    "external_id": "document_external_id",
    "collection_id": "chunk_collection",
}


@dataclass(frozen=True)
class HybridScopePlan:
    """Normalised scope configuration for a hybrid search request."""

    normalized_filters: Dict[str, object | None]
    metadata_filters: List[Tuple[str, object | None]]
    case_value: Optional[str]
    collection_ids_filter: Optional[List[str]]
    has_single_collection_filter: bool
    single_collection_value: Optional[str]
    collection_ids_count: int
    filter_debug: Dict[str, object | None]


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
_LEXICAL_RESULT_MIN_COLUMNS = 6
_ZERO_EPSILON = 1e-12


def get_embedding_dim() -> int:
    """Return the embedding dimensionality reported by the provider."""

    return get_embedding_client().dim()


def _prepare_scope_filters(
    *,
    tenant: str,
    case_id: str | None,
    raw_filters: Mapping[str, object | None] | None,
    visibility_value: str,
) -> HybridScopePlan:
    """Normalise hybrid search scope inputs for downstream processing."""

    normalized_filters: Dict[str, object | None] = {}
    if raw_filters:
        normalized_filters = {
            key: (
                value
                if not (isinstance(value, str) and value == "") and value is not None
                else None
            )
            for key, value in raw_filters.items()
            if key != "visibility"
        }

    raw_collection_ids = normalized_filters.pop("collection_ids", None)
    collection_ids_filter: list[str] | None = None
    if isinstance(raw_collection_ids, Sequence) and not isinstance(
        raw_collection_ids, (str, bytes, bytearray)
    ):
        seen: set[str] = set()
        collected: list[str] = []
        for item in raw_collection_ids:
            if isinstance(item, str):
                candidate = item.strip()
            elif isinstance(item, uuid.UUID):
                candidate = str(item)
            else:
                candidate = str(item).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            collected.append(candidate)
        if collected:
            collection_ids_filter = collected
        else:
            # Explicitly treat empty sequences as no collection filter to keep tenant-wide search.
            collection_ids_filter = None

    has_single_collection_filter = False
    if "collection_id" in normalized_filters:
        single_value = normalized_filters["collection_id"]
        if isinstance(single_value, str):
            trimmed = single_value.strip()
            normalized_filters["collection_id"] = trimmed or None
        elif isinstance(single_value, uuid.UUID):
            normalized_filters["collection_id"] = str(single_value)
        elif single_value is None:
            normalized_filters.pop("collection_id", None)
        else:
            candidate = str(single_value).strip()
            normalized_filters["collection_id"] = candidate or None
        if normalized_filters.get("collection_id") is None:
            normalized_filters.pop("collection_id", None)
        has_single_collection_filter = "collection_id" in normalized_filters

    if case_id not in {None, ""}:
        case_value: Optional[str] = str(case_id)
    else:
        raw_case_value = normalized_filters.get("case_id")
        if raw_case_value in {None, ""}:
            case_value = None
        else:
            case_value = str(raw_case_value)

    normalized_filters["tenant_id"] = tenant
    normalized_filters["case_id"] = case_value

    if "document_version_id" in normalized_filters:
        doc_version_value = normalized_filters.get("document_version_id")
        if doc_version_value in {None, ""}:
            normalized_filters.pop("document_version_id", None)
        elif isinstance(doc_version_value, uuid.UUID):
            normalized_filters["document_version_id"] = str(doc_version_value)
        else:
            normalized_filters["document_version_id"] = str(doc_version_value).strip()

    if "is_latest" in normalized_filters:
        latest_value = normalized_filters.get("is_latest")
        if isinstance(latest_value, bool):
            normalized_filters["is_latest"] = "true" if latest_value else "false"
        elif latest_value in {None, ""}:
            normalized_filters.pop("is_latest", None)
        else:
            normalized_filters["is_latest"] = str(latest_value).strip().lower()

    if (
        "document_version_id" not in normalized_filters
        and "is_latest" not in normalized_filters
    ):
        normalized_filters["is_latest"] = "true"

    metadata_filters = [
        (key, value)
        for key, value in normalized_filters.items()
        if key not in {"tenant_id", "case_id", "collection_id"}
        and value is not None
        and key in SUPPORTED_METADATA_FILTERS
    ]

    filter_debug: Dict[str, object | None] = {
        "tenant_id": "<set>",
        "visibility": visibility_value,
    }
    for key, value in normalized_filters.items():
        if key in {"tenant_id"}:
            continue
        filter_debug[key] = (
            "<set>" if value is not None and key in SUPPORTED_METADATA_FILTERS else None
        )

    collection_ids_count = 0
    if collection_ids_filter:
        filter_debug["collection_ids"] = "<set>"
        try:
            collection_ids_count = len(collection_ids_filter)
        except Exception:
            collection_ids_count = 0

    filter_debug["has_single_collection_filter"] = has_single_collection_filter

    single_collection_value = None
    if has_single_collection_filter:
        single_candidate = normalized_filters.get("collection_id")
        if isinstance(single_candidate, str):
            single_collection_value = single_candidate
        elif isinstance(single_candidate, uuid.UUID):
            single_collection_value = str(single_candidate)
        elif single_candidate is not None:
            single_collection_value = str(single_candidate).strip() or None

    return HybridScopePlan(
        normalized_filters=normalized_filters,
        metadata_filters=metadata_filters,
        case_value=case_value,
        collection_ids_filter=collection_ids_filter,
        has_single_collection_filter=has_single_collection_filter,
        single_collection_value=single_collection_value,
        collection_ids_count=collection_ids_count,
        filter_debug=filter_debug,
    )


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


def _get_bool_setting(name: str, default: bool) -> bool:
    env_value = os.getenv(name)
    if env_value is not None:
        lowered = env_value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    try:  # pragma: no cover - requires Django settings
        from django.conf import settings  # type: ignore

        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return value
    except Exception:
        return default
    return default


DocumentKey = Tuple[str, str | None, str, str | None]
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
    index_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    cached_total_candidates: int | None = None
    scores: List[Dict[str, float]] | None = None


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
    """pgvector-backed client for chunk storage and retrieval.

    Near-duplicate detection converts pgvector distances into a cosine-like
    similarity in ``[0, 1]``. The ``RAG_NEAR_DUPLICATE_THRESHOLD`` setting is
    therefore interpreted uniformly across cosine and L2 operators.
    """

    _ROW_SHAPE_WARNINGS: ClassVar[set[Tuple[str, int]]] = set()
    _NEAR_DUPLICATE_OPERATOR_WARNINGS: ClassVar[set[str]] = set()
    _LOGGED_STATEMENT_SCHEMAS: ClassVar[set[str]] = set()
    _CHUNK_INSERT_STATEMENT_NAME = "rag.chunks.bulk_insert"
    _EMBEDDING_UPSERT_STATEMENT_NAME = "rag.embeddings.bulk_upsert"
    _NEAR_DUPLICATE_PARSE_FALLBACK_LOGGED: ClassVar[bool] = False

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
        try:
            self._pool = SimpleConnectionPool(minconn, maxconn, dsn)
        except OperationalError:
            # Allow instantiation without DB for test collection/builds
            logger.warning(
                "Could not connect to vector DB; continuing in disconnected mode."
            )
            self._pool = None
        self._prepare_lock = threading.Lock()
        self._indexes_ready = False
        self._retries = max(1, retries_value)
        self._retry_base_delay = max(0, retry_delay_value) / 1000.0
        self._distance_operator_cache: Dict[str, str] = {}
        self._near_duplicate_operator_support: Dict[str, bool] = {}
        near_strategy = str(_get_setting("RAG_NEAR_DUPLICATE_STRATEGY", "skip")).lower()
        if near_strategy not in {"skip", "replace", "off"}:
            near_strategy = "skip"
        threshold_setting = _get_setting("RAG_NEAR_DUPLICATE_THRESHOLD", 0.97)
        try:
            near_threshold = float(threshold_setting)
        except (TypeError, ValueError):
            near_threshold = 0.97
        probe_setting = _get_setting("RAG_NEAR_DUPLICATE_PROBE_LIMIT", 8)
        try:
            probe_limit = int(probe_setting)
        except (TypeError, ValueError):
            probe_limit = 8
        self._near_duplicate_strategy = near_strategy
        self._near_duplicate_threshold = max(0.0, min(1.0, near_threshold))
        self._near_duplicate_probe_limit = max(1, probe_limit)
        self._near_duplicate_enabled = (
            near_strategy in {"skip", "replace"}
            and self._near_duplicate_threshold > 0.0
        )
        self._require_unit_norm_for_l2 = _get_bool_setting(
            "RAG_NEAR_DUPLICATE_REQUIRE_UNIT_NORM", False
        )
        self._near_duplicate_operator_supported: bool | None = None

        self._log_bulk_statement_templates()

    def _table(self, name: str) -> sql.Identifier:
        """Return a schema-qualified identifier for ``name``."""

        return sql.Identifier(self._schema, name)

    def _qualified_table_name(self, name: str) -> str:
        """Return the quoted identifier for ``schema.table`` for logging."""

        schema_part = self._schema.replace('"', '""')
        table_part = name.replace('"', '""')
        return f'"{schema_part}"."{table_part}"'

    def _log_bulk_statement_templates(self) -> None:
        """Emit debug logging for bulk insert statements once per schema."""

        if self._schema in self._LOGGED_STATEMENT_SCHEMAS:
            return

        chunk_table = self._qualified_table_name("chunks")
        embedding_table = self._qualified_table_name("embeddings")
        chunk_sql = (
            f"INSERT INTO {chunk_table} ("
            "id, document_id, ord, text, tokens, metadata, tenant_id, collection_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        embedding_sql = (
            f"INSERT INTO {embedding_table} (id, chunk_id, embedding, tenant_id, collection_id) "
            "VALUES (%s, %s, %s::vector, %s, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding, "
            "tenant_id = EXCLUDED.tenant_id, "
            "collection_id = EXCLUDED.collection_id"
        )
        logger.debug(
            "rag.vector.statements.registered",
            extra={
                "schema": self._schema,
                "statements": {
                    self._CHUNK_INSERT_STATEMENT_NAME: chunk_sql,
                    self._EMBEDDING_UPSERT_STATEMENT_NAME: embedding_sql,
                },
            },
        )
        self._LOGGED_STATEMENT_SCHEMAS.add(self._schema)

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

        if self._pool:
            self._pool.closeall()

    @staticmethod
    def _normalise_result_row(
        row: Sequence[object] | Mapping[str, object], *, kind: str
    ) -> tuple[
        object | None,
        object | None,
        Dict[str, object],
        object | None,
        object | None,
        object | None,
        float,
    ]:
        # Support both tuple/sequence-shaped rows and mapping-shaped rows
        # returned by different query execution paths in tests and adapters.
        # When a mapping is provided, extract fields by name.
        if isinstance(row, Mapping):
            # Expected keys: id, text, metadata, hash, document_id,
            # optional collection_id and either distance (vector) or lscore (lexical)
            chunk_id = row.get("id")
            text_value = row.get("text")
            metadata_value = row.get("metadata")
            doc_hash = row.get("hash")
            doc_id = row.get("document_id")
            collection_id = row.get("collection_id")
            score_candidate = PgVectorClient._extract_score_from_row(row, kind=kind)
            fallback = 1.0 if kind == "vector" else 0.0
            try:
                score_float = (
                    float(score_candidate) if score_candidate is not None else fallback
                )
            except (TypeError, ValueError):
                score_float = fallback
            if math.isnan(score_float) or math.isinf(score_float):
                score_float = fallback

            metadata_dict = MetadataHandler.coerce_map(metadata_value)
            return (
                chunk_id,
                text_value,
                metadata_dict,
                doc_hash,
                doc_id,
                collection_id,
                score_float,
            )

        row_tuple = tuple(row)
        length = len(row_tuple)
        if length not in {6, 7}:
            key = (kind, length)
            if key not in PgVectorClient._ROW_SHAPE_WARNINGS:
                logger.warning(
                    "rag.hybrid.row_shape_mismatch",
                    kind=kind,
                    row_len=length,
                )
                PgVectorClient._ROW_SHAPE_WARNINGS.add(key)
        padded_list: List[object] = list((row_tuple + (None,) * 7)[:7])

        metadata_dict = MetadataHandler.coerce_map(padded_list[2])
        padded_list[2] = metadata_dict

        score_value = padded_list[-1]
        fallback = 1.0 if kind == "vector" else 0.0
        try:
            score_float = float(score_value) if score_value is not None else fallback
        except (TypeError, ValueError):
            score_float = fallback
        if math.isnan(score_float) or math.isinf(score_float):
            score_float = fallback

        return (
            padded_list[0],
            padded_list[1],
            metadata_dict,
            padded_list[3],
            padded_list[4],
            padded_list[5],
            score_float,
        )

    @staticmethod
    def _extract_score_from_row(row: object, *, kind: str) -> object | None:
        if isinstance(row, Mapping):
            key = "distance" if kind == "vector" else "lscore"
            value = row.get(key)
            if value is not None:
                return value
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
            # Take the last element as score to support both 6- and 7-column shapes
            try:
                return row[-1]
            except Exception:
                return None
        return None

    @staticmethod
    def _ensure_chunk_metadata_contract(
        meta: Mapping[str, object] | None,
        *,
        tenant_id: str | None,
        case_id: str | None,
        filters: Mapping[str, object | None] | None,
        chunk_id: object,
        document_id: object,
        collection_id: object | None = None,
    ) -> Dict[str, object]:
        enriched = PgVectorClient._strip_collection_scope(meta, remove_collection=True)
        if "tenant_id" not in enriched and tenant_id:
            enriched["tenant_id"] = tenant_id
        filter_case_value = (filters or {}).get("case_id", case_id)
        if "case_id" not in enriched:
            enriched["case_id"] = filter_case_value
        enriched.setdefault("document_id", document_id)
        enriched.setdefault("chunk_id", chunk_id)
        if collection_id is not None and "collection_id" not in enriched:
            try:
                enriched["collection_id"] = (
                    collection_id
                    if isinstance(collection_id, (str, bytes, bytearray))
                    else str(collection_id)
                )
            except Exception:
                enriched["collection_id"] = str(collection_id)
        return enriched

    @staticmethod
    def _strip_collection_scope(
        metadata: object, *, remove_collection: bool = False
    ) -> Dict[str, object]:
        if isinstance(metadata, Mapping):
            sanitized = dict(metadata)
        elif isinstance(metadata, Sequence) and not isinstance(
            metadata, (str, bytes, bytearray)
        ):
            try:
                sanitized = dict(metadata)  # type: ignore[arg-type]
            except Exception:
                sanitized = {}
        else:
            sanitized = {}
        if remove_collection:
            sanitized.pop("collection_id", None)
        return sanitized

    @staticmethod
    def _workflow_predicate_clause(
        workflow_id: str | None,
    ) -> tuple[sql.SQL, tuple[str, ...]]:
        if workflow_id is None:
            return sql.SQL("workflow_id IS NULL"), ()
        return sql.SQL("workflow_id = %s"), (workflow_id,)

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

    def _disable_near_duplicate_for_operator(
        self,
        *,
        index_kind: str,
        operator: str,
        tenant_uuid: uuid.UUID,
    ) -> None:
        """Disable near-duplicate detection for unsupported operators."""

        self._near_duplicate_enabled = False
        self._near_duplicate_operator_supported = False
        warning_key = f"{index_kind}:{operator}"
        if warning_key in self._NEAR_DUPLICATE_OPERATOR_WARNINGS:
            return
        self._NEAR_DUPLICATE_OPERATOR_WARNINGS.add(warning_key)
        logger.warning(
            "ingestion.doc.near_duplicate_operator_unsupported",
            extra={
                "tenant_id": str(tenant_uuid),
                "index_kind": index_kind,
                "operator": operator,
            },
        )

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

    def _is_near_duplicate_operator_supported(
        self, index_kind: str, operator: str
    ) -> bool:
        key = index_kind.upper()
        cached = self._near_duplicate_operator_support.get(key)
        if cached is not None:
            return cached

        supported = False
        if operator == "<=>":
            supported = True
        elif operator == "<->":
            supported = True
            if self._require_unit_norm_for_l2:
                logger.info(
                    "ingestion.doc.near_duplicate_l2_enabled",
                    extra={
                        "index_kind": key,
                        "requires_unit_normalised": True,
                    },
                )
            else:
                logger.info(
                    "ingestion.doc.near_duplicate_l2_distance_mode",
                    extra={
                        "index_kind": key,
                        "requires_unit_normalised": False,
                    },
                )
        else:
            logger.warning(
                "ingestion.doc.near_duplicate_operator_unsupported",
                extra={"index_kind": key, "operator": operator},
            )

        self._near_duplicate_operator_support[key] = supported
        return supported

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
                f"SET LOCAL statement_timeout = {int(self._statement_timeout_ms)}"
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
                        try:
                            cur.execute(
                                f"SET LOCAL statement_timeout = {int(self._statement_timeout_ms)}"
                            )
                        except Exception:
                            pass
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
            1
            for action in doc_actions.values()
            if action in {"skipped", "near_duplicate_skipped"}
        )
        metrics.RAG_UPSERT_CHUNKS.inc(inserted_chunks)
        documents_info: List[Dict[str, object]] = []
        for key, doc in grouped.items():
            # DEBUG.1.initial_doc â€“ capture initial state for this document candidate
            try:
                logger.warning(
                    "vector_client.ensure_documents",
                    extra={
                        "event": "DEBUG.1.initial_doc",
                        "tenant_id": str(doc.get("tenant_id")),
                        "external_id": str(doc.get("external_id")),
                        "workflow_id": str(doc.get("workflow_id")),
                        "collection_id": str(doc.get("collection_id")),
                        "initial_metadata": json.dumps(
                            doc.get("metadata"), default=str, ensure_ascii=False
                        ),
                        "initial_parents": json.dumps(
                            doc.get("parents"), default=str, ensure_ascii=False
                        ),
                    },
                )
            except Exception:
                pass
            tenant_id, workflow_id, external_id, collection_id = key
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
                "workflow_id": workflow_id,
                "external_id": external_id,
                "content_hash": doc.get("content_hash"),
                "action": action,
                "chunk_count": chunk_count,
                "duration_ms": duration,
            }
            if workflow_id is None:
                doc_payload.pop("workflow_id", None)
            if collection_id is not None:
                doc_payload["collection_id"] = collection_id
            near_info = doc.get("near_duplicate_info")
            if isinstance(near_info, Mapping):
                matched_external = near_info.get("external_id")
                if matched_external is not None:
                    doc_payload["near_duplicate_of"] = str(matched_external)
                similarity_value = near_info.get("similarity")
                if similarity_value is not None:
                    try:
                        doc_payload["near_duplicate_similarity"] = float(
                            similarity_value
                        )
                    except (TypeError, ValueError):
                        pass
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
            elif action == "linked":
                metrics.INGESTION_DOCS_INSERTED.inc()
            elif action in {"replaced", "near_duplicate_replaced"}:
                metrics.INGESTION_DOCS_REPLACED.inc()
            elif action in {"skipped", "near_duplicate_skipped"}:
                metrics.INGESTION_DOCS_SKIPPED.inc()
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

    def list_chunks_by_version(
        self,
        *,
        tenant_id: str,
        document_id: uuid.UUID | str,
        document_version_id: uuid.UUID | str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, object]]:
        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        try:
            document_uuid = (
                document_id
                if isinstance(document_id, uuid.UUID)
                else uuid.UUID(str(document_id))
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid_document_id") from exc
        version_text = str(document_version_id)

        chunks_table = self._table("chunks")
        query = sql.SQL(
            "SELECT id, ord, text, metadata "
            "FROM {} "
            "WHERE document_id = %s "
            "AND tenant_id = %s "
            "AND metadata ->> 'document_version_id' = %s "
            "ORDER BY ord"
        ).format(chunks_table)
        params: list[object] = [document_uuid, tenant_uuid, version_text]
        if limit is not None:
            query += sql.SQL(" LIMIT %s")
            params.append(int(limit))
        if offset is not None:
            query += sql.SQL(" OFFSET %s")
            params.append(int(offset))

        rows: list[tuple[object, int, str, object]] = []
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = list(cur.fetchall())

        results: list[dict[str, object]] = []
        for chunk_id, ord_value, text_value, meta_value in rows:
            metadata = MetadataHandler.coerce_map(meta_value)
            results.append(
                {
                    "chunk_id": str(chunk_id),
                    "ord": ord_value,
                    "text": text_value,
                    "metadata": metadata,
                }
            )
        return results

    def hard_delete_document_versions(
        self,
        *,
        tenant_id: str,
        document_version_ids: Sequence[object],
    ) -> dict[str, int]:
        if not document_version_ids:
            return {"chunks": 0, "embeddings": 0}

        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        version_values: list[str] = []
        for version_id in document_version_ids:
            try:
                version_values.append(str(uuid.UUID(str(version_id))))
            except (TypeError, ValueError, AttributeError):
                version_values.append(str(version_id))

        chunks_table = self._table("chunks")
        embeddings_table = self._table("embeddings")
        embeddings_deleted = 0
        chunks_deleted = 0
        with self._connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {} WHERE chunk_id IN ("
                            "SELECT id FROM {} WHERE tenant_id = %s"
                            " AND metadata ->> 'document_version_id' = ANY(%s)"
                            ")"
                        ).format(embeddings_table, chunks_table),
                        (tenant_uuid, version_values),
                    )
                    embeddings_deleted = cur.rowcount or 0
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {} WHERE tenant_id = %s"
                            " AND metadata ->> 'document_version_id' = ANY(%s)"
                        ).format(chunks_table),
                        (tenant_uuid, version_values),
                    )
                    chunks_deleted = cur.rowcount or 0
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "chunks": int(chunks_deleted),
            "embeddings": int(embeddings_deleted),
        }

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
        # Compatibility with VectorStoreRouter keyword passthrough
        collection_id: str | None = None,
        workflow_id: str | None = None,
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
        scope_plan = _prepare_scope_filters(
            tenant=tenant,
            case_id=case_id,
            raw_filters=filters,
            visibility_value=visibility_mode.value,
        )
        normalized_filters = scope_plan.normalized_filters
        metadata_filters = scope_plan.metadata_filters
        case_value = scope_plan.case_value
        collection_ids_filter = scope_plan.collection_ids_filter
        has_single_collection_filter = scope_plan.has_single_collection_filter
        single_collection_value = scope_plan.single_collection_value
        collection_ids_count = scope_plan.collection_ids_count
        filter_debug = dict(scope_plan.filter_debug)
        case_filter_value: str | None = None
        if case_value:
            case_filter_value = self._normalise_filter_value(case_value)
        effective_collection_filter: list[str] | None = None
        legacy_doc_class_filter: list[str] | None = None
        if collection_ids_filter:
            effective_collection_filter = [
                self._normalise_filter_value(item) for item in collection_ids_filter
            ]
        elif has_single_collection_filter and single_collection_value is not None:
            effective_collection_filter = [
                self._normalise_filter_value(single_collection_value)
            ]
        if is_collection_routing_enabled() and effective_collection_filter:
            legacy_doc_class_filter = list(effective_collection_filter)
        collection_scope = "none"
        if effective_collection_filter:
            try:
                collection_scope = (
                    "single" if len(effective_collection_filter) == 1 else "list"
                )
            except Exception:
                collection_scope = "list"
        elif has_single_collection_filter:
            collection_scope = "single"
        filter_debug["collection_scope"] = collection_scope
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
        raw_vec: List[float] | None = None
        hyde_enabled = _get_bool_setting("RAG_HYDE_ENABLED", False)
        if hyde_enabled and query_norm:
            context = get_log_context()
            hyde_metadata = {
                "tenant_id": tenant,
                "case_id": case_value,
                "trace_id": context.get("trace_id"),
                "prompt_version": "hyde-v1",
            }
            key_alias = context.get("key_alias")
            if key_alias:
                hyde_metadata["key_alias"] = key_alias
            hyde_text = self._generate_hyde_document(query_norm, hyde_metadata)
            if hyde_text:
                try:
                    raw_vec = self._embed_query(hyde_text)
                except EmbeddingClientError as exc:
                    logger.warning(
                        "rag.hyde.embedding_failed",
                        extra={
                            "tenant_id": tenant,
                            "case_id": case_value,
                            "error": str(exc),
                        },
                    )
                    raw_vec = None
                else:
                    logger.info(
                        "rag.hyde.applied",
                        extra={
                            "tenant_id": tenant,
                            "case_id": case_value,
                            "hyde_chars": len(hyde_text),
                        },
                    )
        if raw_vec is None:
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

        debug_message = (
            "RAG hybrid search inputs: "
            f"tenant={tenant} "
            f"top_k={top_k} "
            f"vec_limit={vec_limit_value} "
            f"lex_limit={lex_limit_value} "
            f"filters={filter_debug} "
            f"collection_ids_count={collection_ids_count} "
            f"has_single_collection_filter={str(has_single_collection_filter).lower()} "
            f"collection_scope={collection_scope}"
        )
        logger.debug(debug_message)

        where_clauses = ["d.tenant_id = %s"]
        deleted_visibility_clauses = (
            "COALESCE(d.lifecycle, 'active') = 'active'",
            "COALESCE(c.metadata ->> 'lifecycle_state', 'active') = 'active'",
        )
        if visibility_mode is Visibility.ACTIVE:
            where_clauses.extend(deleted_visibility_clauses)
        elif visibility_mode is Visibility.DELETED:
            where_clauses.append(
                "(COALESCE(d.lifecycle, 'active') <> 'active'"
                " OR COALESCE(c.metadata ->> 'lifecycle_state', 'active') <> 'active')"
            )
        where_params: List[object] = [tenant_uuid]
        for key, value in metadata_filters:
            kind = SUPPORTED_METADATA_FILTERS[key]
            normalised = self._normalise_filter_value(value)
            if kind == "chunk_meta":
                if key == "is_latest":
                    if normalised == "true":
                        where_clauses.append(
                            "(NOT (c.metadata ? 'is_latest') OR c.metadata ->> 'is_latest' = %s)"
                        )
                        where_params.append(normalised)
                    else:
                        where_clauses.append("c.metadata ->> 'is_latest' = %s")
                        where_params.append(normalised)
                else:
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
        # We intentionally use OR logic (union) between case_id and collection filters to
        # allow retrieval from the current case and selected knowledge collections
        # simultaneously; intersection is not desired here.
        if case_filter_value is not None and effective_collection_filter:
            if legacy_doc_class_filter:
                where_clauses.append(
                    "((c.metadata ->> 'case_id' = %s) OR (c.collection_id = ANY(%s::uuid[])) OR (c.metadata ->> 'doc_class' = ANY(%s)))"
                )
                where_params.extend(
                    [
                        case_filter_value,
                        effective_collection_filter,
                        legacy_doc_class_filter,
                    ]
                )
            else:
                where_clauses.append(
                    "((c.metadata ->> 'case_id' = %s) OR (c.collection_id = ANY(%s::uuid[])))"
                )
                where_params.extend([case_filter_value, effective_collection_filter])
        elif case_filter_value is not None:
            where_clauses.append("c.metadata ->> 'case_id' = %s")
            where_params.append(case_filter_value)
        elif effective_collection_filter:
            if legacy_doc_class_filter:
                where_clauses.append(
                    "((c.collection_id = ANY(%s::uuid[])) OR (c.metadata ->> 'doc_class' = ANY(%s)))"
                )
                where_params.extend(
                    [effective_collection_filter, legacy_doc_class_filter]
                )
            else:
                where_clauses.append("c.collection_id = ANY(%s::uuid[])")
                where_params.append(effective_collection_filter)
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
        total_without_filter: Optional[int] = None

        def _safe_chunk_identifier(value: object | None) -> str | None:
            if value is None:
                return None
            try:
                text = str(value)
            except Exception:
                return None
            return text

        def _safe_float(value: object | None) -> float | None:
            if value is None:
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if math.isnan(number) or math.isinf(number):
                return None
            return number

        def _summarise_rows(
            rows: Iterable[object], *, kind: str
        ) -> List[Dict[str, object | None]]:
            summary: List[Dict[str, object | None]] = []
            for row in rows:
                chunk_ref: object | None = None
                if isinstance(row, Mapping):
                    chunk_ref = row.get("id")
                elif isinstance(row, Sequence) and not isinstance(
                    row, (str, bytes, bytearray)
                ):
                    try:
                        chunk_ref = row[0]
                    except Exception:
                        chunk_ref = None
                score_raw = self._extract_score_from_row(row, kind=kind)
                summary.append(
                    {
                        "chunk_id": _safe_chunk_identifier(chunk_ref),
                        "score": _safe_float(score_raw),
                    }
                )
            return summary

        def _operation() -> Tuple[List[tuple], List[tuple], float]:
            nonlocal applied_trgm_limit_value
            nonlocal fallback_limit_used_value
            nonlocal total_without_filter
            nonlocal distance_operator_value
            nonlocal lexical_query_variant
            nonlocal lexical_fallback_limit_value
            started = time.perf_counter()
            vector_rows: List[tuple] = []
            lexical_rows: List[tuple] = []
            vector_query_failed = vector_format_error is not None
            fallback_limit_used_value = None
            total_without_filter_local: Optional[int] = None
            distance_operator_value = None
            lexical_query_variant = "none"
            lexical_fallback_limit_value = None
            lexical_mode_raw = str(_get_setting("RAG_LEXICAL_MODE", "trgm")).strip()
            lexical_mode = (
                "bm25"
                if lexical_mode_raw.lower() in {"bm25", "tsvector", "fulltext"}
                else "trgm"
            )
            if lexical_mode == "bm25":
                lexical_score_sql = (
                    "ts_rank_cd(c.text_tsv, plainto_tsquery('simple', %s)) AS lscore"
                )
                lexical_match_clause = "c.text_tsv @@ plainto_tsquery('simple', %s)"
            else:
                lexical_score_sql = "similarity(c.text_norm, %s) AS lscore"
                lexical_match_clause = "c.text_norm %% %s"

            with self._connection() as conn:
                vector_outcome = run_vector_search(
                    client=self,
                    conn=conn,
                    query_vec=query_vec,
                    vector_format_error=vector_format_error,
                    index_kind=index_kind,
                    ef_search=ef_search,
                    probes=probes,
                    where_sql=where_sql,
                    where_params=where_params,
                    vec_limit=vec_limit_value,
                    tenant=tenant,
                    case_id=case_value,
                    statement_timeout_ms=self._statement_timeout_ms,
                    summarise_rows=_summarise_rows,
                    logger=logger,
                )
                vector_rows = vector_outcome.rows
                vector_query_failed = vector_outcome.vector_query_failed
                distance_operator_value = vector_outcome.distance_operator

                lexical_outcome = run_lexical_search(
                    client=self,
                    conn=conn,
                    query_db_norm=query_db_norm,
                    where_sql=where_sql,
                    where_params=where_params,
                    lex_limit=lex_limit_value,
                    trgm_limit_value=trgm_limit_value,
                    requested_trgm_limit=requested_trgm_limit,
                    tenant=tenant,
                    case_id=case_value,
                    statement_timeout_ms=self._statement_timeout_ms,
                    schema=self._schema,
                    lexical_score_sql=lexical_score_sql,
                    lexical_match_clause=lexical_match_clause,
                    vector_rows=vector_rows,
                    vector_query_failed=vector_query_failed,
                    summarise_rows=_summarise_rows,
                    extract_score_from_row=lambda row: self._extract_score_from_row(
                        row, kind="lexical"
                    ),
                    logger=logger,
                )
                lexical_rows = lexical_outcome.rows
                applied_trgm_limit_value = lexical_outcome.applied_trgm_limit
                fallback_limit_used_value = lexical_outcome.fallback_limit_used
                lexical_query_variant = lexical_outcome.lexical_query_variant
                lexical_fallback_limit_value = lexical_outcome.lexical_fallback_limit

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
                            vector_count_sql = build_vector_sql(
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
                                lexical_count_sql = build_lexical_primary_sql(
                                    where_sql_without_deleted,
                                    "c.id,\n                                    "
                                    + lexical_score_sql,
                                    lexical_match_clause,
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
                                lexical_count_sql = build_lexical_fallback_sql(
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

        filtered_lexical_rows: List[object] = []
        for row in lexical_rows:
            if isinstance(row, Mapping):
                filtered_lexical_rows.append(row)
                continue
            if isinstance(row, Sequence) and not isinstance(
                row, (str, bytes, bytearray)
            ):
                try:
                    row_length = len(row)
                except Exception:
                    row_length = None
                if row_length is None or row_length < _LEXICAL_RESULT_MIN_COLUMNS:
                    try:
                        logger.warning(
                            "rag.hybrid.lexical_row_malformed",
                            extra={
                                "tenant_id": tenant,
                                "case_id": case_value,
                                "row_len": row_length,
                                "row_preview": repr(row)[:200],
                            },
                        )
                    except Exception:
                        pass
                    continue
            elif not isinstance(row, (str, bytes, bytearray)):
                try:
                    row_length = len(row)  # type: ignore[arg-type]
                except Exception:
                    row_length = None
                if row_length is None or row_length < _LEXICAL_RESULT_MIN_COLUMNS:
                    try:
                        logger.warning(
                            "rag.hybrid.lexical_row_malformed",
                            extra={
                                "tenant_id": tenant,
                                "case_id": case_value,
                                "row_len": row_length,
                                "row_preview": repr(row)[:200],
                            },
                        )
                    except Exception:
                        pass
                    continue
            filtered_lexical_rows.append(row)
        lexical_rows = list(filtered_lexical_rows)

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

        rrf_k_raw = _get_setting("RAG_RRF_K", 60)
        try:
            rrf_k = int(rrf_k_raw)
        except (TypeError, ValueError):
            rrf_k = 60
        rrf_k = max(1, rrf_k)

        fusion_result = fuse_candidates(
            vector_rows=vector_rows,
            lexical_rows=lexical_rows,
            query_vec=query_vec,
            query_embedding_empty=query_embedding_empty,
            alpha=alpha_value,
            min_sim=min_sim_value,
            top_k=top_k,
            tenant=tenant,
            case_id=case_value,
            normalized_filters=normalized_filters,
            fallback_limit_used=fallback_limit_used_value,
            distance_score_mode=distance_score_mode,
            rrf_k=rrf_k,
            extract_score_from_row=self._extract_score_from_row,
            normalise_result_row=self._normalise_result_row,
            ensure_chunk_metadata_contract=self._ensure_chunk_metadata_contract,
            logger=logger,
        )
        limited_results = fusion_result.chunks
        fused_candidates = fusion_result.fused_candidates
        below_cutoff_count = fusion_result.below_cutoff
        returned_after_cutoff = fusion_result.returned_after_cutoff
        per_result_scores = fusion_result.scores

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
            below_cutoff=below_cutoff_count,
            returned_after_cutoff=returned_after_cutoff,
            query_embedding_empty=query_embedding_empty,
            applied_trgm_limit=applied_trgm_limit_value,
            fallback_limit_used=fallback_limit_used_value,
            visibility=visibility_mode.value,
            deleted_matches_blocked=deleted_matches_blocked_value,
            cached_total_candidates=total_without_filter,
            scores=per_result_scores if per_result_scores else None,
        )

    def fetch_parent_context(
        self,
        tenant_id: str,
        requests: Mapping[str, Iterable[str]],
    ) -> Dict[str, Dict[str, object]]:
        """Return parent node payloads for the requested document/parent IDs."""

        if not requests:
            return {}

        requested: Dict[uuid.UUID, set[str]] = {}
        for doc_id, parent_ids in requests.items():
            try:
                doc_uuid = uuid.UUID(str(doc_id))
            except (TypeError, ValueError):
                continue
            parent_set: set[str] = set()
            for parent_id in parent_ids:
                try:
                    text = str(parent_id).strip()
                except Exception:  # pragma: no cover - defensive
                    continue
                if text:
                    parent_set.add(text)
            if parent_set:
                requested[doc_uuid] = parent_set

        if not requested:
            return {}

        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        doc_ids = list(requested.keys())
        results: Dict[str, Dict[str, object]] = {}
        documents_table = self._table("documents")

        def _operation() -> Dict[str, Dict[str, object]]:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            SELECT id, metadata->'parent_nodes'
                            FROM {}
                            WHERE tenant_id = %s AND id = ANY(%s)
                            """
                        ).format(documents_table),
                        (str(tenant_uuid), doc_ids),
                    )
                    rows = cur.fetchall()
            payload: Dict[str, Dict[str, object]] = {}
            for row_id, parent_payload in rows:
                requested_ids = requested.get(row_id)
                if not requested_ids:
                    continue
                if not isinstance(parent_payload, Mapping):
                    continue
                subset: Dict[str, object] = {}
                for parent_id in requested_ids:
                    node = parent_payload.get(parent_id)
                    if isinstance(node, Mapping):
                        subset[parent_id] = dict(node)
                    elif node is not None:
                        subset[parent_id] = node
                if subset:
                    payload[str(row_id)] = limit_parent_payload(subset)
            return payload

        try:
            results = self._run_with_retries(_operation, op_name="fetch_parent_context")
        except Exception:  # pragma: no cover - defensive logging
            logger.exception(
                "rag.parents.lookup_failed",
                extra={
                    "tenant_id": tenant_id,
                    "doc_count": len(requested),
                },
            )
            return {}

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
            # Accept legacy key "tenant" as an alias for "tenant_id" to retain
            # backwards compatibility with older call sites/tests.
            tenant_value = chunk.meta.get("tenant_id") or chunk.meta.get("tenant")
            doc_hash = str(chunk.meta.get("hash"))
            raw_source = chunk.meta.get("source")
            try:
                source = str(raw_source).strip()
            except Exception:
                source = ""
            if not source:
                source = "unknown"
            external_id = chunk.meta.get("external_id")
            raw_collection_id = chunk.meta.get("collection_id")
            raw_doc_class = chunk.meta.get("doc_class")
            raw_document_id = chunk.meta.get("document_id")
            provided_document_uuid: uuid.UUID | None = None
            if raw_document_id not in {None, "", "None"}:
                try:
                    provided_document_uuid = (
                        raw_document_id
                        if isinstance(raw_document_id, uuid.UUID)
                        else uuid.UUID(str(raw_document_id).strip())
                    )
                except (TypeError, ValueError, AttributeError):
                    provided_document_uuid = None
            if raw_collection_id in {None, "", "None"} and raw_doc_class not in {
                None,
                "",
                "None",
            }:
                raw_collection_id = raw_doc_class
            collection_id: str | None = None
            if raw_collection_id not in {None, "", "None"}:
                try:
                    collection_id = str(uuid.UUID(str(raw_collection_id).strip()))
                except (TypeError, ValueError, AttributeError):
                    try:
                        candidate = str(raw_collection_id).strip()
                    except Exception:
                        candidate = ""
                    collection_id = candidate or None
            raw_workflow_id = chunk.meta.get("workflow_id")
            workflow_id: str | None = None
            if raw_workflow_id not in {None, "", "None"}:
                try:
                    candidate_workflow = str(raw_workflow_id).strip()
                except Exception:
                    candidate_workflow = ""
                workflow_id = candidate_workflow or None
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
            key = (tenant, workflow_id, external_id_str, collection_id)
            if key not in grouped:
                document_uuid = provided_document_uuid or uuid.uuid4()
                grouped[key] = {
                    "id": document_uuid,
                    "tenant_id": tenant,
                    "workflow_id": workflow_id,
                    "external_id": external_id_str,
                    "hash": doc_hash,
                    "content_hash": doc_hash,
                    "source": source,
                    "collection_id": collection_id,
                    "metadata": {
                        k: v
                        for k, v in chunk.meta.items()
                        if k
                        not in {"tenant_id", "tenant", "hash", "source", "doc_class"}
                    },
                    "chunks": [],
                    "parents": {},
                }
                grouped_metadata = grouped[key]["metadata"]
                grouped_metadata["document_id"] = str(grouped[key]["id"])
                grouped_metadata["document_id"] = str(grouped[key]["id"])
            else:
                doc_collection = grouped[key].get("collection_id")
                if doc_collection is None and collection_id is not None:
                    grouped[key]["collection_id"] = collection_id
                if grouped[key].get("workflow_id") is None and workflow_id is not None:
                    grouped[key]["workflow_id"] = workflow_id
                if (
                    provided_document_uuid is not None
                    and grouped[key].get("id") != provided_document_uuid
                ):
                    logger.warning(
                        "ingestion.doc.document_id_mismatch",
                        extra={
                            "tenant_id": tenant,
                            "external_id": external_id_str,
                            "existing_id": str(grouped[key].get("id")),
                            "provided_id": str(provided_document_uuid),
                        },
                    )
                metadata_block = grouped[key].get("metadata")
                if isinstance(metadata_block, dict):
                    metadata_block["document_id"] = str(grouped[key]["id"])
                    metadata_block["document_id"] = str(grouped[key]["id"])
            chunk_meta = dict(chunk.meta)
            chunk_meta["tenant_id"] = tenant
            chunk_meta["external_id"] = external_id_str
            document_identifier = str(grouped[key]["id"])
            chunk_meta["document_id"] = document_identifier
            chunk_meta["document_id"] = document_identifier
            if collection_id is not None:
                chunk_meta["collection_id"] = collection_id
            elif "collection_id" in chunk_meta:
                chunk_meta.pop("collection_id", None)
            chunk_meta.pop("doc_class", None)
            if workflow_id is not None:
                chunk_meta["workflow_id"] = workflow_id
            elif "workflow_id" in chunk_meta:
                chunk_meta.pop("workflow_id", None)
            parents_map = grouped[key].get("parents")
            chunk_parents = chunk.parents
            if isinstance(parents_map, dict) and isinstance(chunk_parents, Mapping):
                limited_chunk_parents = limit_parent_payload(chunk_parents)
                for parent_id, parent_meta in limited_chunk_parents.items():
                    if isinstance(parent_meta, Mapping):
                        parent_payload = dict(parent_meta)
                        parent_payload.setdefault("document_id", document_identifier)
                    else:
                        parent_payload = parent_meta
                    parents_map[parent_id] = parent_payload
            grouped[key]["chunks"].append(
                Chunk(
                    content=chunk.content,
                    meta=chunk_meta,
                    embedding=chunk.embedding,
                    parents=chunk_parents,
                )
            )
        return grouped

    def _compute_document_embedding(
        self, doc: Mapping[str, object]
    ) -> tuple[List[float], bool] | None:
        return compute_document_embedding(doc)

    def _find_near_duplicate(
        self,
        cur,
        tenant_uuid: uuid.UUID,
        vector: Sequence[float],
        *,
        external_id: str,
        embedding_is_unit_normalised: bool,
        collection_uuid: uuid.UUID | None = None,
    ) -> Dict[str, object] | None:
        return find_near_duplicate(
            client=self,
            cur=cur,
            tenant_uuid=tenant_uuid,
            vector=vector,
            external_id=external_id,
            embedding_is_unit_normalised=embedding_is_unit_normalised,
            collection_uuid=collection_uuid,
            get_setting=_get_setting,
            log=logger,
        )

    @classmethod
    def _log_near_duplicate_similarity_fallback(
        cls, *, tenant_uuid: uuid.UUID, external_id: str
    ) -> None:
        if cls._NEAR_DUPLICATE_PARSE_FALLBACK_LOGGED:
            return
        cls._NEAR_DUPLICATE_PARSE_FALLBACK_LOGGED = True
        logger.warning(
            "ingestion.doc.near_duplicate_similarity_fallback",
            extra={
                "tenant_id": str(tenant_uuid),
                "external_id": external_id,
            },
        )

    def _log_ensure_documents_debug(
        self,
        *,
        event: str,
        tenant_id: object,
        external_id: str,
        document_id: uuid.UUID | None = None,
        json_fields: Mapping[str, object] | None = None,
        raw_fields: Mapping[str, object] | None = None,
    ) -> None:
        try:
            extra: dict[str, object] = {
                "event": event,
                "tenant_id": str(tenant_id),
                "external_id": external_id,
            }
            if document_id is not None:
                extra["document_id"] = str(document_id)
            if json_fields:
                for key, value in json_fields.items():
                    extra[key] = json.dumps(value, default=str, ensure_ascii=False)
            if raw_fields:
                extra.update(raw_fields)
            logger.warning("vector_client.ensure_documents", extra=extra)
        except Exception:
            pass

    def _process_existing_document_for_upsert(
        self,
        cur,
        *,
        key: DocumentKey,
        doc: MutableMapping[str, object],
        tenant_uuid: uuid.UUID,
        external_id: str,
        existing_row: tuple,
        metadata_dict: Mapping[str, object],
        storage_hash: str,
        lifecycle_state: str,
        collection_text: str | None,
        workflow_text: str | None,
        documents_table: sql.Identifier,
        collection_uuid: uuid.UUID | None,
        relation_actor: str | None,
        actions: Dict[DocumentKey, str],
        document_ids: Dict[DocumentKey, uuid.UUID],
        event_suffix: str,
        log_skip_info: bool,
    ) -> None:
        (
            existing_id,
            existing_hash,
            existing_metadata,
            _existing_source,
            existing_lifecycle,
            _existing_collection_value,
        ) = existing_row
        metadata_dict = MetadataHandler.normalise_document_identity(
            doc, existing_id, metadata_dict
        )
        self._log_ensure_documents_debug(
            event=f"DEBUG.2.after_normalise{event_suffix}",
            tenant_id=tenant_uuid,
            external_id=external_id,
            document_id=existing_id,
            json_fields={"metadata_after_normalise": metadata_dict},
        )
        existing_metadata_map = MetadataHandler.coerce_map(existing_metadata)
        self._log_ensure_documents_debug(
            event=f"DEBUG.3.pre_diff{event_suffix}",
            tenant_id=tenant_uuid,
            external_id=external_id,
            document_id=existing_id,
            json_fields={
                "existing_metadata": existing_metadata_map,
                "desired_metadata": metadata_dict,
            },
        )
        metadata_mismatch = MetadataHandler.parent_nodes_differ(
            existing_metadata_map, metadata_dict
        )
        if metadata_mismatch and MetadataHandler.content_hash_matches(
            existing_metadata_map,
            metadata_dict,
            existing_fallback=existing_hash,
            desired_fallback=storage_hash,
        ):
            metadata_mismatch = False
        needs_update = (
            str(existing_hash) != storage_hash
            or (
                existing_lifecycle is not None
                and str(existing_lifecycle).lower() != LIFECYCLE_ACTIVE
            )
            or metadata_mismatch
        )
        self._log_ensure_documents_debug(
            event=f"DEBUG.4.post_diff{event_suffix}",
            tenant_id=tenant_uuid,
            external_id=external_id,
            document_id=existing_id,
            raw_fields={
                "metadata_mismatch": bool(metadata_mismatch),
                "needs_update": bool(needs_update),
            },
        )
        if needs_update:
            self._log_ensure_documents_debug(
                event=f"DEBUG.5.pre_update{event_suffix}",
                tenant_id=tenant_uuid,
                external_id=external_id,
                document_id=existing_id,
                json_fields={"metadata_to_update": metadata_dict},
            )
            metadata_payload = Json(metadata_dict)
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}
                    SET source = %s,
                        hash = %s,
                        metadata = %s,
                        collection_id = %s,
                        workflow_id = %s,
                        lifecycle = %s,
                        deleted_at = NULL
                    WHERE id = %s
                    """
                ).format(documents_table),
                (
                    doc["source"],
                    storage_hash,
                    metadata_payload,
                    collection_text,
                    workflow_text,
                    lifecycle_state,
                    existing_id,
                ),
            )
            actions[key] = "replaced"
        else:
            actions[key] = "skipped"
            if log_skip_info:
                logger.info(
                    "Skipping unchanged document during upsert",
                    extra={
                        "tenant_id": doc["tenant_id"],
                        "external_id": external_id,
                    },
                )
            self._log_ensure_documents_debug(
                event=f"DEBUG.6.update_skipped{event_suffix}",
                tenant_id=tenant_uuid,
                external_id=external_id,
                document_id=existing_id,
                json_fields={"metadata_skipped": metadata_dict},
            )
            MetadataHandler.repair_persisted_metadata(
                cur,
                documents_table=documents_table,
                tenant_uuid=tenant_uuid,
                document_id=existing_id,
                existing_metadata=existing_metadata,
                desired_metadata=metadata_dict,
                log=logger,
            )
        relation_inserted = False
        if collection_uuid is not None:
            relation_inserted = self._ensure_document_collection_relation(
                cur,
                existing_id,
                collection_uuid,
                added_by=relation_actor,
            )
        if relation_inserted:
            current_action = actions.get(key)
            if current_action in {"skipped", "near_duplicate_skipped", None}:
                actions[key] = "linked"
        document_ids[key] = existing_id
        doc["collection_id"] = collection_text
        doc["workflow_id"] = workflow_text

    def _ensure_documents(
        self,
        cur,
        grouped: GroupedDocuments,
    ) -> Tuple[Dict[DocumentKey, uuid.UUID], Dict[DocumentKey, str]]:  # type: ignore[no-untyped-def]
        document_ids: Dict[DocumentKey, uuid.UUID] = {}
        actions: Dict[DocumentKey, str] = {}
        documents_table = self._table("documents")
        for key, doc in grouped.items():
            tenant_uuid = self._coerce_tenant_uuid(doc["tenant_id"])
            external_id = str(doc["external_id"])
            try:
                source_text = str(doc.get("source", "")).strip()
            except Exception:
                source_text = ""
            source = source_text or "unknown"
            doc["source"] = source
            content_hash = str(doc.get("content_hash", doc.get("hash", "")))
            collection_value = doc.get("collection_id")
            collection_uuid = self._parse_collection_uuid(collection_value, strict=True)
            workflow_raw = doc.get("workflow_id")
            workflow_text: str | None = None
            if workflow_raw not in {None, "", "None"}:
                try:
                    candidate_workflow = str(workflow_raw).strip()
                except Exception:
                    candidate_workflow = ""
                workflow_text = candidate_workflow or None
            doc["workflow_id"] = workflow_text
            storage_hash = self._compute_storage_hash(
                cur,
                tenant_uuid,
                content_hash,
                external_id,
                source,
                workflow_id=workflow_text,
            )
            doc["hash"] = storage_hash
            doc["content_hash"] = content_hash
            metadata_dict = self._strip_collection_scope(doc.get("metadata"))
            metadata_dict.setdefault("external_id", external_id)
            if collection_uuid is not None:
                metadata_dict["collection_id"] = str(collection_uuid)
            else:
                metadata_dict.pop("collection_id", None)
            relation_actor: str | None = None
            for actor_key in ("uploader_id", "added_by", "ingested_by"):
                actor_value = metadata_dict.get(actor_key)
                if actor_value in {None, ""}:
                    continue
                try:
                    candidate_actor = str(actor_value).strip()
                except Exception:
                    candidate_actor = ""
                if candidate_actor:
                    relation_actor = candidate_actor
                    break
            raw_lifecycle = doc.get("lifecycle_state")
            if raw_lifecycle is None:
                raw_lifecycle = metadata_dict.get("lifecycle_state")
            lifecycle_state = _normalize_lifecycle_state(raw_lifecycle)
            metadata_dict["lifecycle_state"] = lifecycle_state
            doc["lifecycle_state"] = lifecycle_state
            metadata_dict.setdefault("hash", content_hash)
            if workflow_text is not None:
                metadata_dict["workflow_id"] = workflow_text
            elif "workflow_id" in metadata_dict:
                metadata_dict.pop("workflow_id", None)
            document_uuid_value = doc.get("id")
            document_id_text: str | None = None
            if isinstance(document_uuid_value, uuid.UUID):
                document_id_text = str(document_uuid_value)
            elif document_uuid_value not in {None, "", "None"}:
                try:
                    document_id_text = str(uuid.UUID(str(document_uuid_value)))
                except (TypeError, ValueError, AttributeError):
                    try:
                        document_id_text = str(document_uuid_value).strip()
                    except Exception:
                        document_id_text = None
            if document_id_text:
                metadata_dict["document_id"] = document_id_text
                metadata_dict["document_id"] = document_id_text

            canonical_document_uuid: uuid.UUID | None = None
            if isinstance(document_uuid_value, uuid.UUID):
                canonical_document_uuid = document_uuid_value
            elif document_id_text:
                try:
                    canonical_document_uuid = uuid.UUID(document_id_text)
                except (TypeError, ValueError, AttributeError):
                    canonical_document_uuid = None
            provisional_document_id: uuid.UUID
            if canonical_document_uuid is not None:
                provisional_document_id = canonical_document_uuid
                doc["id"] = canonical_document_uuid
            else:
                if isinstance(document_uuid_value, uuid.UUID):
                    provisional_document_id = document_uuid_value
                else:
                    try:
                        provisional_document_id = uuid.UUID(str(document_uuid_value))
                    except (TypeError, ValueError, AttributeError):
                        provisional_document_id = uuid.uuid4()
                        doc["id"] = provisional_document_id

            metadata_dict = MetadataHandler.normalise_document_identity(
                doc, provisional_document_id, metadata_dict
            )

            if self._near_duplicate_enabled:
                index_kind = str(_get_setting("RAG_INDEX_KIND", "HNSW")).upper()
                try:
                    operator = self._get_distance_operator(cur.connection, index_kind)
                    if not self._is_near_duplicate_operator_supported(
                        index_kind, operator
                    ):
                        self._near_duplicate_enabled = False
                except Exception:
                    self._near_duplicate_enabled = False

            near_duplicate_details = None
            if self._near_duplicate_enabled:
                doc_embedding = self._compute_document_embedding(doc)
                if doc_embedding is not None:
                    embedding_vector, is_unit_normalised = doc_embedding
                    near_duplicate_details = self._find_near_duplicate(
                        cur,
                        tenant_uuid,
                        embedding_vector,
                        external_id=external_id,
                        embedding_is_unit_normalised=is_unit_normalised,
                        collection_uuid=collection_uuid,
                    )
            if near_duplicate_details is not None:
                doc["near_duplicate_info"] = near_duplicate_details
                similarity = float(near_duplicate_details.get("similarity", 0.0))
                log_extra = {
                    "tenant_id": doc["tenant_id"],
                    "external_id": external_id,
                    "similarity": similarity,
                    "matched_external_id": near_duplicate_details.get("external_id"),
                }
                matched_id_value = near_duplicate_details.get("id")
                existing_document_id: uuid.UUID | None = None
                if matched_id_value is not None:
                    try:
                        existing_document_id = (
                            matched_id_value
                            if isinstance(matched_id_value, uuid.UUID)
                            else uuid.UUID(str(matched_id_value))
                        )
                    except (TypeError, ValueError, AttributeError):
                        existing_document_id = None
                if self._near_duplicate_strategy == "skip":
                    if existing_document_id is not None:
                        metadata_dict = MetadataHandler.normalise_document_identity(
                            doc, existing_document_id, metadata_dict
                        )
                    actions[key] = "near_duplicate_skipped"
                    logger.info("ingestion.doc.near_duplicate_skipped", extra=log_extra)
                    continue
                if (
                    self._near_duplicate_strategy == "replace"
                    and existing_document_id is not None
                ):
                    metadata_dict["near_duplicate_of"] = str(
                        near_duplicate_details.get("external_id")
                    )
                    metadata_dict["near_duplicate_similarity"] = similarity
                    metadata_dict = MetadataHandler.normalise_document_identity(
                        doc, existing_document_id, metadata_dict
                    )
                    metadata = Json(metadata_dict)
                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {}
                            SET external_id = %s,
                                source = %s,
                                hash = %s,
                                metadata = %s,
                                workflow_id = %s,
                                deleted_at = NULL
                            WHERE id = %s
                            """
                        ).format(documents_table),
                        (
                            external_id,
                            doc["source"],
                            storage_hash,
                            metadata,
                            workflow_text,
                            existing_document_id,
                        ),
                    )
                    document_ids[key] = existing_document_id
                    actions[key] = "near_duplicate_replaced"
                    doc["collection_id"] = (
                        str(collection_uuid) if collection_uuid is not None else None
                    )
                    doc["workflow_id"] = workflow_text
                    logger.info(
                        "ingestion.doc.near_duplicate_replaced",
                        extra={
                            **log_extra,
                            "document_id": str(existing_document_id),
                        },
                    )
                    continue
                if self._near_duplicate_strategy == "replace":
                    logger.warning(
                        "ingestion.doc.near_duplicate_missing_id",
                        extra=log_extra,
                    )
                    actions[key] = "near_duplicate_skipped"
                    continue

            raw_document_id = doc.get("id")
            if isinstance(raw_document_id, uuid.UUID):
                document_id = raw_document_id
            else:
                try:
                    document_id = uuid.UUID(str(raw_document_id))
                except (TypeError, ValueError, AttributeError):
                    document_id = uuid.uuid4()
                doc["id"] = document_id
            if collection_uuid is not None:
                self._ensure_collection_scope(cur, tenant_uuid, collection_uuid)
            tenant_value = str(tenant_uuid)
            collection_text = (
                str(collection_uuid) if collection_uuid is not None else None
            )
            workflow_clause, workflow_params = self._workflow_predicate_clause(
                workflow_text
            )
            existing: tuple | None
            matched_by_hash = False
            select_external_sql = sql.SQL(
                """
                SELECT id, hash, metadata, source, lifecycle, collection_id
                FROM {}
                WHERE tenant_id = %s
                  AND {}
                  AND external_id = %s
                LIMIT 1
                """
            ).format(documents_table, workflow_clause)
            params: list[object] = [tenant_value]
            params.extend(workflow_params)
            params.append(external_id)
            cur.execute(select_external_sql, tuple(params))
            existing = cur.fetchone()
            if not existing:
                select_hash_sql = sql.SQL(
                    """
                    SELECT id, hash, metadata, source, lifecycle, collection_id
                    FROM {}
                    WHERE tenant_id = %s
                      AND {}
                      AND source = %s
                      AND hash = %s
                    LIMIT 1
                    """
                ).format(documents_table, workflow_clause)
                hash_params: list[object] = [tenant_value]
                hash_params.extend(workflow_params)
                hash_params.extend([source, storage_hash])
                cur.execute(select_hash_sql, tuple(hash_params))
                existing = cur.fetchone()
                matched_by_hash = existing is not None
            if existing and matched_by_hash:
                existing_metadata_map = MetadataHandler.coerce_map(existing[2])
                existing_external = existing_metadata_map.get("external_id")
                desired_external = metadata_dict.get("external_id", external_id)
                try:
                    existing_external_text = (
                        str(existing_external).strip() if existing_external else ""
                    )
                except Exception:
                    existing_external_text = ""
                try:
                    desired_external_text = (
                        str(desired_external).strip() if desired_external else ""
                    )
                except Exception:
                    desired_external_text = ""
                if (
                    existing_external_text
                    and desired_external_text
                    and existing_external_text != desired_external_text
                ):
                    existing = None
            try:
                if existing:
                    self._process_existing_document_for_upsert(
                        cur,
                        key=key,
                        doc=doc,
                        tenant_uuid=tenant_uuid,
                        external_id=external_id,
                        existing_row=existing,
                        metadata_dict=metadata_dict,
                        storage_hash=storage_hash,
                        lifecycle_state=lifecycle_state,
                        collection_text=collection_text,
                        workflow_text=workflow_text,
                        documents_table=documents_table,
                        collection_uuid=collection_uuid,
                        relation_actor=relation_actor,
                        actions=actions,
                        document_ids=document_ids,
                        event_suffix="",
                        log_skip_info=False,
                    )
                    continue

                metadata_dict = MetadataHandler.normalise_document_identity(
                    doc, document_id, metadata_dict
                )
                metadata = Json(metadata_dict)
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (
                            id,
                            tenant_id,
                            collection_id,
                            workflow_id,
                            external_id,
                            source,
                            hash,
                            metadata,
                            lifecycle,
                            deleted_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                    ).format(documents_table),
                    (
                        document_id,
                        tenant_value,
                        collection_text,
                        workflow_text,
                        external_id,
                        doc["source"],
                        storage_hash,
                        metadata,
                        lifecycle_state,
                        None,
                    ),
                )
                document_ids[key] = document_id
                actions[key] = "inserted"
                if collection_uuid is not None:
                    self._ensure_document_collection_relation(
                        cur,
                        document_id,
                        collection_uuid,
                        added_by=relation_actor,
                    )
                doc["collection_id"] = collection_text
            except UniqueViolation:
                conn = cur.connection
                try:
                    conn.rollback()
                except Exception:
                    pass
                with conn.cursor() as retry_cur:
                    try:
                        self._restore_session_after_rollback(retry_cur)
                    except Exception:
                        pass
                    if collection_text is None:
                        retry_sql = sql.SQL(
                            """
                            SELECT id, hash, metadata, source, lifecycle, collection_id
                            FROM {}
                            WHERE tenant_id = %s
                              AND {}
                              AND external_id = %s
                            LIMIT 1
                            """
                        ).format(documents_table, workflow_clause)
                        retry_params: list[object] = [tenant_value]
                    else:
                        retry_sql = sql.SQL(
                            """
                            SELECT id, hash, metadata, source, lifecycle, collection_id
                            FROM {}
                            WHERE tenant_id = %s
                              AND {}
                              AND external_id = %s
                            LIMIT 1
                            """
                        ).format(documents_table, workflow_clause)
                        retry_params = [tenant_value]
                    retry_params.extend(workflow_params)
                    retry_params.append(external_id)
                    retry_cur.execute(retry_sql, tuple(retry_params))
                    duplicate = retry_cur.fetchone()
                    if not duplicate:
                        raise

                    self._process_existing_document_for_upsert(
                        retry_cur,
                        key=key,
                        doc=doc,
                        tenant_uuid=tenant_uuid,
                        external_id=external_id,
                        existing_row=duplicate,
                        metadata_dict=metadata_dict,
                        storage_hash=storage_hash,
                        lifecycle_state=lifecycle_state,
                        collection_text=collection_text,
                        workflow_text=workflow_text,
                        documents_table=documents_table,
                        collection_uuid=collection_uuid,
                        relation_actor=relation_actor,
                        actions=actions,
                        document_ids=document_ids,
                        event_suffix=".retry",
                        log_skip_info=True,
                    )
        return document_ids, actions

    def _ensure_collection_scope(
        self, cur, tenant_uuid: uuid.UUID, collection_uuid: uuid.UUID
    ) -> None:
        try:
            collections_table = self._table("collections")
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (tenant_id, id)
                    VALUES (%s, %s)
                    ON CONFLICT (tenant_id, id) DO NOTHING
                    """
                ).format(collections_table),
                (str(tenant_uuid), str(collection_uuid)),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "ingestion.collections.ensure_failed",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "collection_id": str(collection_uuid),
                    "error": str(exc),
                },
            )

    def _ensure_document_collection_relation(
        self,
        cur,
        document_uuid: uuid.UUID,
        collection_uuid: uuid.UUID,
        *,
        added_by: str | None = None,
    ) -> bool:
        relation_table = self._table("document_collections")
        actor = (added_by or "system").strip() or "system"
        timestamp = datetime.now(timezone.utc)
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {} (document_id, collection_id, added_at, added_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id, collection_id) DO NOTHING
                """
            ).format(relation_table),
            (str(document_uuid), str(collection_uuid), timestamp, actor),
        )
        return bool(cur.rowcount)

    def ensure_collection(
        self,
        *,
        tenant_id: str,
        collection_id: object,
        embedding_profile: str | None = None,
        scope: str | None = None,
    ) -> None:
        """Ensure a vector collection record exists for ``tenant_id``."""

        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        collection_uuid = self._parse_collection_uuid(collection_id, strict=True)
        if collection_uuid is None:
            raise ValueError("invalid_collection_id")

        with self._connection() as conn:  # type: ignore[attr-defined]
            try:
                with conn.cursor() as cur:
                    self._ensure_collection_scope(cur, tenant_uuid, collection_uuid)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        logger.info(
            "rag.collections.ensure",
            extra={
                "tenant_id": str(tenant_uuid),
                "collection_id": str(collection_uuid),
                "embedding_profile": embedding_profile,
                "scope": scope,
            },
        )

    def delete_collection(
        self,
        *,
        tenant_id: str,
        collection_id: object,
    ) -> None:
        """Remove vector collection metadata and orphaned relations."""

        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        collection_uuid = self._parse_collection_uuid(collection_id, strict=True)
        if collection_uuid is None:
            raise ValueError("invalid_collection_id")

        documents_table = self._table("documents")
        chunks_table = self._table("chunks")
        embeddings_table = self._table("embeddings")
        relation_table = self._table("document_collections")
        collections_table = self._table("collections")

        clear_documents_sql = sql.SQL(
            """
            UPDATE {documents}
               SET collection_id = NULL,
                   metadata = jsonb_strip_nulls(
                       coalesce(metadata, '{{}}'::jsonb) - 'collection_id'
                   )
             WHERE tenant_id = %s
               AND collection_id = %s
            """
        ).format(documents=documents_table)

        delete_chunks_sql = sql.SQL(
            """
            DELETE FROM {chunks}
             WHERE tenant_id = %s
               AND collection_id = %s
            """
        ).format(chunks=chunks_table)

        delete_embeddings_sql = sql.SQL(
            """
            DELETE FROM {embeddings}
             WHERE tenant_id = %s
               AND collection_id = %s
            """
        ).format(embeddings=embeddings_table)

        delete_relations_sql = sql.SQL(
            """
            DELETE FROM {relations}
             WHERE collection_id = %s
            """
        ).format(relations=relation_table)

        delete_collection_sql = sql.SQL(
            """
            DELETE FROM {collections}
             WHERE tenant_id = %s
               AND id = %s
            """
        ).format(collections=collections_table)

        params = (str(tenant_uuid), str(collection_uuid))

        with self._connection() as conn:  # type: ignore[attr-defined]
            try:
                with conn.cursor() as cur:
                    cur.execute(delete_embeddings_sql, params)
                    cur.execute(delete_chunks_sql, params)
                    cur.execute(clear_documents_sql, params)
                    cur.execute(delete_relations_sql, (str(collection_uuid),))
                    cur.execute(delete_collection_sql, params)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

        logger.info(
            "rag.collections.deleted",
            extra={
                "tenant_id": str(tenant_uuid),
                "collection_id": str(collection_uuid),
            },
        )

    def update_lifecycle_state(
        self,
        *,
        tenant_id: str,
        document_ids: Iterable[object],
        state: str,
        reason: str | None = None,
        changed_at: datetime | None = None,
        cursor: Any | None = None,
    ) -> int:
        lifecycle_state = _normalize_lifecycle_state(state)
        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        resolved_ids: list[uuid.UUID] = []
        for raw in document_ids:
            if raw in {None, "", "None"}:
                continue
            try:
                resolved = raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw))
            except (TypeError, ValueError, AttributeError):
                continue
            resolved_ids.append(resolved)
        if not resolved_ids:
            return 0

        timestamp = (changed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        changed_iso = timestamp.isoformat()
        reason_text = None
        if isinstance(reason, str):
            normalized_reason = reason.strip()
            if normalized_reason:
                reason_text = normalized_reason
        deleted_timestamp = None
        deleted_iso = None
        if lifecycle_state != LIFECYCLE_ACTIVE:
            deleted_timestamp = timestamp
            deleted_iso = changed_iso

        documents_table = self._table("documents")
        chunks_table = self._table("chunks")

        documents_sql = sql.SQL(
            """
            UPDATE {}
            SET lifecycle = %s,
                deleted_at = %s,
                metadata = jsonb_strip_nulls(
                    coalesce(metadata, '{{}}'::jsonb)
                    || jsonb_build_object(
                        'lifecycle_state', %s,
                        'lifecycle_changed_at', %s,
                        'lifecycle_reason', %s,
                        'deleted_at', %s
                    )
                )
            WHERE tenant_id = %s AND id = ANY(%s)
            """
        ).format(documents_table)

        chunks_sql = sql.SQL(
            """
            UPDATE {}
            SET metadata = jsonb_strip_nulls(
                coalesce(metadata, '{{}}'::jsonb)
                || jsonb_build_object(
                    'lifecycle_state', %s,
                    'lifecycle_changed_at', %s,
                    'lifecycle_reason', %s,
                    'deleted_at', %s
                )
            )
            WHERE document_id = ANY(%s)
            """
        ).format(chunks_table)

        params = (
            lifecycle_state,
            deleted_timestamp,
            lifecycle_state,
            changed_iso,
            reason_text,
            deleted_iso,
            str(tenant_uuid),
            resolved_ids,
        )

        chunk_params = (
            lifecycle_state,
            changed_iso,
            reason_text,
            deleted_iso,
            resolved_ids,
        )

        def _execute(cur) -> int:
            cur.execute(documents_sql, params)
            updated = cur.rowcount or 0
            cur.execute(chunks_sql, chunk_params)
            return updated

        if cursor is not None:
            return _execute(cursor)

        with self._connection() as conn:  # type: ignore[attr-defined]
            try:
                with conn.cursor() as cur:
                    updated_rows = _execute(cur)
                conn.commit()
                return updated_rows
            except Exception:
                conn.rollback()
                raise

    def hard_delete_documents(
        self,
        *,
        tenant_id: str,
        document_ids: Sequence[object],
    ) -> Mapping[str, int]:
        """Permanently remove documents, chunks and embeddings for a tenant."""

        tenant_uuid = self._coerce_tenant_uuid(tenant_id)
        resolved_ids: list[uuid.UUID] = []
        for raw in document_ids:
            if raw in {None, "", "None"}:
                continue
            try:
                resolved = raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw))
            except (TypeError, ValueError, AttributeError):
                continue
            resolved_ids.append(resolved)

        if not resolved_ids:
            return {"documents": 0, "chunks": 0, "embeddings": 0}

        documents_table = self._table("documents")
        chunks_table = self._table("chunks")
        embeddings_table = self._table("embeddings")

        delete_embeddings_sql = sql.SQL(
            """
            DELETE FROM {embeddings}
             WHERE chunk_id IN (
                SELECT id
                  FROM {chunks}
                 WHERE document_id = ANY(%s)
                   AND tenant_id = %s
            )
            """
        ).format(embeddings=embeddings_table, chunks=chunks_table)

        delete_chunks_sql = sql.SQL(
            """
            DELETE FROM {chunks}
             WHERE document_id = ANY(%s)
               AND tenant_id = %s
            """
        ).format(chunks=chunks_table)

        delete_documents_sql = sql.SQL(
            """
            DELETE FROM {documents}
             WHERE tenant_id = %s
               AND id = ANY(%s)
            """
        ).format(documents=documents_table)

        params = (resolved_ids, str(tenant_uuid))

        with self._connection() as conn:  # type: ignore[attr-defined]
            try:
                with conn.cursor() as cur:
                    cur.execute(delete_embeddings_sql, params)
                    embeddings_deleted = cur.rowcount or 0

                    cur.execute(delete_chunks_sql, params)
                    chunks_deleted = cur.rowcount or 0

                    cur.execute(delete_documents_sql, (str(tenant_uuid), resolved_ids))
                    documents_deleted = cur.rowcount or 0

                conn.commit()
            except Exception:
                conn.rollback()
                raise

        return {
            "documents": int(documents_deleted),
            "chunks": int(chunks_deleted),
            "embeddings": int(embeddings_deleted),
        }

    def _compute_storage_hash(
        self,
        cur,
        tenant_uuid: uuid.UUID,
        content_hash: str,
        external_id: str,
        source: str,
        workflow_id: str | None = None,
    ) -> str:
        documents_table = self._table("documents")
        workflow_clause, workflow_params = self._workflow_predicate_clause(workflow_id)
        return compute_storage_hash(
            cur=cur,
            documents_table=documents_table,
            workflow_clause=workflow_clause,
            workflow_params=workflow_params,
            tenant_uuid=tenant_uuid,
            content_hash=content_hash,
            external_id=external_id,
            source=source,
            log=logger,
        )

    def _coerce_tenant_uuid(self, tenant_id: object) -> uuid.UUID:
        """Return a UUID for ``tenant_id`` while keeping legacy IDs stable."""

        if isinstance(tenant_id, uuid.UUID):
            return tenant_id

        text = str(tenant_id or "").strip()
        if not text or text.lower() == "none":
            raise ValueError("Chunk metadata must include a tenant identifier")

        try:
            return uuid.UUID(text)
        except (TypeError, ValueError):
            normalised = text.lower()
            derived = uuid.uuid5(uuid.NAMESPACE_URL, f"tenant:{normalised}")
            logger.warning(
                "Mapped legacy tenant identifier to deterministic UUID",
                extra={
                    "tenant_id": text,
                    "normalised_tenant_id": normalised,
                    "derived_tenant_uuid": str(derived),
                },
            )
            return derived

    def _parse_collection_uuid(
        self, value: object, *, strict: bool
    ) -> uuid.UUID | None:
        if value in {None, ""}:
            return None
        if not strict and value == "None":
            return None
        try:
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        except (TypeError, ValueError, AttributeError) as exc:
            if strict:
                raise ValueError("invalid_collection_id") from exc
            return None

    def _replace_chunks(
        self,
        cur,
        grouped: GroupedDocuments,
        document_ids: Dict[DocumentKey, uuid.UUID],
        doc_actions: Dict[DocumentKey, str],
    ) -> Tuple[int, Dict[DocumentKey, Dict[str, float]]]:  # type: ignore[no-untyped-def]
        chunks_table = self._table("chunks")
        embeddings_table = self._table("embeddings")
        chunk_insert_sql = sql.SQL(
            """
            INSERT INTO {} (
                id,
                document_id,
                ord,
                text,
                tokens,
                metadata,
                tenant_id,
                collection_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        ).format(chunks_table)
        embedding_insert_sql = sql.SQL(
            """
            INSERT INTO {} (id, chunk_id, embedding, tenant_id, collection_id)
            VALUES (%s, %s, %s::vector, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                tenant_id = EXCLUDED.tenant_id,
                collection_id = EXCLUDED.collection_id
        """
        ).format(embeddings_table)
        chunk_insert_sql_str = chunk_insert_sql.as_string(cur.connection)
        embedding_insert_sql_str = embedding_insert_sql.as_string(cur.connection)

        inserted = 0
        per_doc_stats: Dict[DocumentKey, Dict[str, float]] = {}
        index_kind = str(_get_setting("RAG_INDEX_KIND", "HNSW")).upper()
        try:
            storage_operator = self._get_distance_operator(cur.connection, index_kind)
        except Exception:
            storage_operator = None
        store_normalised_embeddings = storage_operator != "<->"
        for key, doc in grouped.items():
            action = doc_actions.get(key, "inserted")
            if action in {"skipped", "near_duplicate_skipped"}:
                per_doc_stats[key] = {"chunk_count": 0, "duration_ms": 0.0}
                continue
            document_id = document_ids[key]
            tenant_value = str(doc.get("tenant_id", ""))
            collection_value = doc.get("collection_id")
            collection_uuid = self._parse_collection_uuid(
                collection_value, strict=False
            )
            collection_text = (
                str(collection_uuid) if collection_uuid is not None else None
            )
            metadata_block = doc.get("metadata", {})
            raw_version_id = (
                metadata_block.get("document_version_id")
                if isinstance(metadata_block, Mapping)
                else None
            )
            document_version_id = None
            if raw_version_id not in {None, "", "None"}:
                try:
                    document_version_id = str(raw_version_id).strip()
                except Exception:
                    document_version_id = None
            started = time.perf_counter()
            if document_version_id:
                cur.execute(
                    sql.SQL(
                        "UPDATE {} "
                        "SET metadata = jsonb_set(metadata, %s::text[], 'false'::jsonb, true) "
                        "WHERE document_id = %s "
                        "AND collection_id IS NOT DISTINCT FROM %s "
                        "AND (metadata ->> 'document_version_id') IS DISTINCT FROM %s"
                    ).format(chunks_table),
                    ("{is_latest}", document_id, collection_uuid, document_version_id),
                )
                cur.execute(
                    sql.SQL(
                        "DELETE FROM {} WHERE chunk_id IN ("
                        "SELECT id FROM {} WHERE document_id = %s"
                        " AND collection_id IS NOT DISTINCT FROM %s"
                        " AND metadata ->> 'document_version_id' = %s)"
                    ).format(embeddings_table, chunks_table),
                    (document_id, collection_uuid, document_version_id),
                )
                cur.execute(
                    sql.SQL(
                        "DELETE FROM {} WHERE document_id = %s"
                        " AND collection_id IS NOT DISTINCT FROM %s"
                        " AND metadata ->> 'document_version_id' = %s"
                    ).format(chunks_table),
                    (document_id, collection_uuid, document_version_id),
                )
            else:
                cur.execute(
                    sql.SQL(
                        "DELETE FROM {} WHERE chunk_id IN ("
                        "SELECT id FROM {} WHERE document_id = %s"
                        " AND collection_id IS NOT DISTINCT FROM %s)"
                    ).format(embeddings_table, chunks_table),
                    (document_id, collection_uuid),
                )
                cur.execute(
                    sql.SQL(
                        "DELETE FROM {} WHERE document_id = %s"
                        " AND collection_id IS NOT DISTINCT FROM %s"
                    ).format(chunks_table),
                    (document_id, collection_uuid),
                )

            chunk_rows = []
            embedding_rows = []
            chunk_count = 0
            for index, chunk in enumerate(doc["chunks"]):
                raw_chunk_hash = (
                    chunk.meta.get("hash") if hasattr(chunk, "meta") else None
                )
                try:
                    chunk_hash = str(raw_chunk_hash or "").strip()
                except Exception:
                    chunk_hash = ""
                if not chunk_hash:
                    chunk_hash = hashlib.sha256(
                        f"{document_id}:{index}".encode("utf-8")
                    ).hexdigest()
                chunk_namespace = (
                    f"{tenant_value}:{document_id}:{collection_text or 'global'}:"
                    f"{index}:{chunk_hash}"
                )
                chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, chunk_namespace)
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
                            "document_id": str(document_id),
                            "chunk_id": str(chunk_id),
                            "source": doc.get("source"),
                        },
                    )
                tokens = self._estimate_tokens(chunk.content)
                chunk_metadata = self._strip_collection_scope(
                    chunk.meta, remove_collection=True
                )
                chunk_rows.append(
                    (
                        chunk_id,
                        document_id,
                        index,
                        chunk.content,
                        tokens,
                        Json(chunk_metadata),
                        tenant_value,
                        collection_text,
                    )
                )
                chunk_count += 1
                if normalised_embedding is not None:
                    embedding_to_store: Sequence[float] | None = None
                    if store_normalised_embeddings:
                        embedding_to_store = normalised_embedding
                    else:
                        try:
                            embedding_to_store = [
                                float(value) for value in (embedding_values or [])
                            ]
                        except (TypeError, ValueError):
                            embedding_to_store = normalised_embedding
                    if embedding_to_store:
                        vector_value = self._format_vector(embedding_to_store)
                        embedding_rows.append(
                            (
                                uuid.uuid4(),
                                chunk_id,
                                vector_value,
                                tenant_value,
                                collection_text,
                            )
                        )

            if chunk_rows:
                cur.executemany(chunk_insert_sql_str, chunk_rows)
            if embedding_rows:
                cur.executemany(embedding_insert_sql_str, embedding_rows)
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
        return format_vector(values, expected_dim=expected_dim)

    def _format_vector_lenient(self, values: Sequence[float]) -> str:
        """Format a vector without enforcing provider dimension.

        This helper is used for query-time formatting only, to avoid turning a
        temporary dimension mismatch (e.g. during tests or provider changes)
        into a hard failure that would bypass the vector search entirely.
        """
        return format_vector_lenient(values)

    def _generate_hyde_document(
        self, query: str, metadata: Mapping[str, Any]
    ) -> str | None:
        label = str(_get_setting("RAG_HYDE_MODEL_LABEL", "simple-query")).strip()
        if not label:
            label = "simple-query"
        max_chars_setting = _get_setting("RAG_HYDE_MAX_CHARS", 2000)
        try:
            max_chars = int(max_chars_setting)
        except (TypeError, ValueError):
            max_chars = 2000
        max_chars = max(0, max_chars)
        prompt = _HYDE_PROMPT_TEMPLATE.format(query=query)
        tenant_id = metadata.get("tenant_id")
        case_id = metadata.get("case_id")
        try:
            from ai_core.llm.client import LlmClientError, call as llm_call
        except Exception as exc:
            logger.warning(
                "rag.hyde.unavailable",
                extra={
                    "tenant_id": tenant_id,
                    "case_id": case_id,
                    "error": str(exc),
                },
            )
            return None
        try:
            response = llm_call(label, prompt, dict(metadata))
        except LlmClientError as exc:
            logger.warning(
                "rag.hyde.failed",
                extra={
                    "tenant_id": tenant_id,
                    "case_id": case_id,
                    "error": str(exc),
                },
            )
            return None
        text = response.get("text") if isinstance(response, Mapping) else None
        if not isinstance(text, str):
            return None
        candidate = text.strip()
        if not candidate:
            return None
        if max_chars and len(candidate) > max_chars:
            candidate = candidate[:max_chars]
        logger.info(
            "rag.hyde.generated",
            extra={
                "tenant_id": tenant_id,
                "case_id": case_id,
                "model_label": label,
                "chars": len(candidate),
            },
        )
        return candidate

    def _embed_query(self, query: str) -> List[float]:
        return embed_query(query, log=logger, get_client=get_embedding_client)

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
_SCHEMA_CLIENTS: Dict[str, PgVectorClient] = {}


def _resolve_vector_schema() -> str:
    """Return the schema configured for the default vector store.

    The management commands rely on :func:`get_default_client` to run SQL
    statements such as index rebuilds. In multi-store setups we determine the
    schema according to the active default scope, honouring the ``default``
    flag used by :class:`~ai_core.rag.vector_store.VectorStoreRouter`.
    """

    for env_var in ("RAG_VECTOR_SCHEMA", "DEV_TENANT_SCHEMA"):
        schema_env = os.getenv(env_var)
        if schema_env:
            schema_env = schema_env.strip()
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


def get_client_for_schema(schema: str | None) -> PgVectorClient:
    schema_value = str(schema or "").strip() or _resolve_vector_schema()
    client = _SCHEMA_CLIENTS.get(schema_value)
    if client is None:
        client = PgVectorClient.from_env(schema=schema_value)
        _SCHEMA_CLIENTS[schema_value] = client
    return client


def reset_schema_clients() -> None:
    for client in _SCHEMA_CLIENTS.values():
        client.close()
    _SCHEMA_CLIENTS.clear()


def reset_default_client() -> None:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is not None:
        _DEFAULT_CLIENT.close()
    _DEFAULT_CLIENT = None
    reset_schema_clients()


atexit.register(reset_default_client)

try:  # pragma: no cover - celery optional
    from celery.signals import worker_shutdown
except Exception:  # pragma: no cover
    worker_shutdown = None
else:  # pragma: no cover - requires celery runtime

    @worker_shutdown.connect  # type: ignore[attr-defined]
    def _close_pgvector_pool(**_kwargs):
        reset_default_client()
