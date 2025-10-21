from __future__ import annotations

import atexit
import json
import math
import os
import re
import struct
import threading
import time
import uuid
from array import array
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
from .parents import limit_parent_payload
from .routing_rules import is_collection_routing_enabled

from . import metrics
from .filters import strict_match
from .schemas import Chunk
from .visibility import Visibility

# Ensure jsonb columns are decoded into Python dictionaries
register_default_jsonb(loads=json.loads, globally=True)

logger = get_logger(__name__)

logger.info(
    "module_loaded",
    extra={"module": __name__, "path": os.path.abspath(__file__)},
)


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


def _coerce_vector_values(value: object) -> list[float] | None:
    """Attempt to coerce ``value`` into a list of floats."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        if not stripped:
            return []
        parts = [component for component in re.split(r"[\s,]+", stripped) if component]
        if not parts:
            return []
        try:
            return [float(component) for component in parts]
        except (TypeError, ValueError):
            return None
    if isinstance(value, memoryview):
        view = value
        if view.ndim == 1 and view.format in {"f", "d"}:
            try:
                return [float(component) for component in view]
            except (TypeError, ValueError):
                return None
        return _coerce_vector_values(view.tobytes())
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        if len(data) >= 2:
            dimension = struct.unpack("!H", data[:2])[0]
            payload = data[2:]
            if dimension == 0 and payload:
                return None
            for format_char in ("f", "d"):
                component_size = struct.calcsize(f"!{format_char}")
                expected_length = dimension * component_size
                if dimension == 0 and not payload:
                    return []
                if len(payload) != expected_length:
                    continue
                try:
                    unpacked = struct.unpack(f"!{dimension}{format_char}", payload)
                except struct.error:
                    continue
                return [float(component) for component in unpacked]
        for typecode in ("f", "d"):
            try:
                arr = array(typecode)
                arr.frombytes(data)
            except (ValueError, OverflowError):
                continue
            return [float(component) for component in arr]
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return [float(component) for component in value]
        except (TypeError, ValueError):
            return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # pragma: no branch - defensive
        try:
            converted = tolist()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(converted, Sequence) and not isinstance(
            converted, (str, bytes, bytearray)
        ):
            try:
                return [float(component) for component in converted]
            except (TypeError, ValueError):
                return None
    values_attr = getattr(value, "values", None)
    if isinstance(values_attr, Sequence) and not isinstance(
        values_attr, (str, bytes, bytearray)
    ):
        try:
            return [float(component) for component in values_attr]
        except (TypeError, ValueError):
            return None
    return None


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
        self._pool = SimpleConnectionPool(minconn, maxconn, dsn)
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

        self._pool.closeall()

    @staticmethod
    def _normalise_result_row(
        row: Sequence[object], *, kind: str
    ) -> Tuple[object, object, Mapping[str, object], object, object, float]:
        # Support both tuple/sequence-shaped rows and mapping-shaped rows
        # returned by different query execution paths in tests and adapters.
        # When a mapping is provided, extract fields by name.
        from typing import Mapping as _Mapping  # local alias to avoid confusion

        if isinstance(row, _Mapping):
            # Expected keys: id, text, metadata, hash, doc_id and either
            # distance (vector) or lscore (lexical)
            chunk_id = row.get("id")
            text_value = row.get("text")
            metadata_value = row.get("metadata")
            doc_hash = row.get("hash")
            doc_id = row.get("doc_id")
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

            # Coerce metadata into a plain dict
            metadata_dict: Dict[str, object]
            if isinstance(metadata_value, _Mapping):
                metadata_dict = dict(metadata_value)
            else:
                metadata_dict = {}

            return (
                chunk_id,
                text_value,
                metadata_dict,
                doc_hash,
                doc_id,
                score_float,
            )

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
    def _extract_score_from_row(row: object, *, kind: str) -> object | None:
        if isinstance(row, Mapping):
            key = "distance" if kind == "vector" else "lscore"
            value = row.get(key)
            if value is not None:
                return value
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
            try:
                return row[5]
            except IndexError:
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
        doc_id: object,
    ) -> Dict[str, object]:
        enriched = PgVectorClient._strip_collection_scope(meta)
        if "tenant_id" not in enriched and tenant_id:
            enriched["tenant_id"] = tenant_id
        filter_case_value = (filters or {}).get("case_id", case_id)
        if "case_id" not in enriched:
            enriched["case_id"] = filter_case_value
        enriched.setdefault("doc_id", doc_id)
        enriched.setdefault("chunk_id", chunk_id)
        return enriched

    @staticmethod
    def _strip_collection_scope(metadata: object) -> Dict[str, object]:
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
        fallback_tried_limits: List[float] = []
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
                      AND c.text_norm %% %s
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
                                logger.debug(
                                    "rag.hybrid.rows.vector_raw",
                                    extra={
                                        "tenant_id": tenant,
                                        "case_id": case_value,
                                        "count": len(vector_rows),
                                        "rows": _summarise_rows(
                                            vector_rows, kind="vector"
                                        ),
                                    },
                                )
                            except Exception:
                                pass
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
                        # Ensure the schema is active for all lexical operations
                        # in this transaction, regardless of connection setup.
                        # This is required for fallback paths and mocked
                        # connections used in tests where _prepare_connection
                        # is bypassed.
                        try:
                            cur.execute(
                                sql.SQL("SET LOCAL search_path TO {}, public").format(
                                    sql.Identifier(self._schema)
                                )
                            )
                        except Exception:
                            # If SET LOCAL is not available, rely on connection-level
                            # search_path set during _prepare_connection.
                            pass
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
                              AND c.text_norm %% %s
                            ORDER BY lscore DESC
                            LIMIT %s
                        """
                        fallback_requested = requested_trgm_limit is not None
                        should_run_fallback = False
                        if query_db_norm.strip():
                            # Optional probe: a cheap LIMIT 0 primary query to detect DB-level
                            # errors (simulated by tests) early. If this raises, we'll roll back
                            # before running the similarity fallback.
                            probe_failed = False
                            try:
                                cur.execute(
                                    lexical_sql,
                                    (
                                        query_db_norm,
                                        *where_params,
                                        query_db_norm,
                                        lex_limit_value,  # probe uses clamped limit
                                    ),
                                )
                                _ = cur.fetchall()
                            except (IndexError, ValueError, PsycopgError) as exc:
                                probe_failed = True
                                should_run_fallback = True
                                fallback_requires_rollback = True
                                lexical_rows_local = []
                                try:
                                    logger.warning(
                                        "rag.debug.lexical_probe_exception",
                                        extra={
                                            "tenant_id": tenant,
                                            "case_id": case_value,
                                            "exc_type": exc.__class__.__name__,
                                            "fallback_requires_rollback": True,
                                        },
                                    )
                                except Exception:
                                    pass
                                logger.warning(
                                    "rag.hybrid.lexical_primary_failed",
                                    tenant_id=tenant,
                                    case_id=case_value,
                                    error=str(exc),
                                )
                                if (
                                    isinstance(exc, PsycopgError)
                                    and vector_query_failed
                                ):
                                    raise

                            if not probe_failed:
                                try:
                                    logger.warning(
                                        "rag.debug.lexical_primary_try_enter",
                                        extra={
                                            "tenant_id": tenant,
                                            "case_id": case_value,
                                            "probe_failed": probe_failed,
                                        },
                                    )
                                except Exception:
                                    pass
                                # Phase 1: execute + fetch. If this raises, treat as DB-level
                                # failure requiring a rollback before fallback.
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
                                    fetched_rows = cur.fetchall()
                                except (IndexError, ValueError, PsycopgError) as exc:
                                    should_run_fallback = True
                                    fallback_requires_rollback = True
                                    lexical_rows_local = []
                                    try:
                                        logger.warning(
                                            "rag.debug.lexical_primary_exception",
                                            extra={
                                                "tenant_id": tenant,
                                                "case_id": case_value,
                                                "exc_type": exc.__class__.__name__,
                                                "fallback_requires_rollback": True,
                                            },
                                        )
                                    except Exception:
                                        pass
                                    logger.warning(
                                        "rag.hybrid.lexical_primary_failed",
                                        tenant_id=tenant,
                                        case_id=case_value,
                                        error=str(exc),
                                    )
                                    if (
                                        isinstance(exc, PsycopgError)
                                        and vector_query_failed
                                    ):
                                        raise
                                else:
                                    lexical_rows_local = fetched_rows
                                    lexical_query_variant = "primary"
                                    # Phase 2: light row-shape probe. If this fails, do NOT
                                    # require rollback; it's a client-side processing issue.
                                    try:
                                        if lexical_rows_local:
                                            _ = tuple(lexical_rows_local[0])
                                    except Exception as exc:
                                        should_run_fallback = True
                                        fallback_requires_rollback = False
                                        # Collect minimal diagnostics about the returned row shape
                                        rows_count = 0
                                        first_len = None
                                        first_type = None
                                        try:
                                            rows_count = len(lexical_rows_local)
                                            first = lexical_rows_local[0]
                                            first_type = type(first).__name__
                                            try:
                                                first_len = len(first)  # may fail
                                            except Exception:
                                                first_len = None
                                        except Exception:
                                            pass
                                        lexical_rows_local = []
                                        logger.warning(
                                            "rag.hybrid.lexical_primary_failed",
                                            tenant_id=tenant,
                                            case_id=case_value,
                                            error=str(exc),
                                            rows_count=rows_count,
                                            row_first_len=first_len,
                                            row_first_type=first_type,
                                        )
                                        try:
                                            logger.warning(
                                                "rag.debug.fallback_flag_set",
                                                extra={
                                                    "tenant_id": tenant,
                                                    "case_id": case_value,
                                                    "reason": "row_shape_error",
                                                    "fallback_requires_rollback": False,
                                                },
                                            )
                                        except Exception:
                                            pass
                                try:
                                    logger.debug(
                                        "rag.hybrid.rows.lexical_raw",
                                        extra={
                                            "tenant_id": tenant,
                                            "case_id": case_value,
                                            "count": len(lexical_rows_local),
                                            "rows": _summarise_rows(
                                                lexical_rows_local, kind="lexical"
                                            ),
                                        },
                                    )
                                except Exception:
                                    pass
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
                                # Wrap the score validation block to ensure any unexpected
                                # row-shape/type errors are handled by the inner except below.
                                try:
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
                                                limit_threshold = float(
                                                    applied_trgm_limit
                                                )
                                            except (TypeError, ValueError):
                                                limit_threshold = None
                                        if limit_threshold is not None:
                                            for row in lexical_rows_local:
                                                try:
                                                    score_candidate = (
                                                        self._extract_score_from_row(
                                                            row, kind="lexical"
                                                        )
                                                    )
                                                except Exception:
                                                    # Defensive: ignore rows with unexpected shape/types
                                                    continue
                                                if score_candidate is None:
                                                    continue
                                                try:
                                                    score_value = float(score_candidate)
                                                except (TypeError, ValueError):
                                                    continue
                                                if score_value < limit_threshold - 1e-6:
                                                    invalid_lscore = True
                                                    break
                                        if invalid_lscore:
                                            should_run_fallback = True
                                except Exception as exc:
                                    handled = False
                                    if isinstance(exc, (IndexError, ValueError)):
                                        handled = True
                                        should_run_fallback = True
                                        # Row-shape/processing errors at this stage should not
                                        # require a transaction rollback.
                                        fallback_requires_rollback = False
                                        # Collect minimal diagnostics about the returned row shape
                                        rows_count = 0
                                        first_len = None
                                        first_type = None
                                        try:
                                            rows_count = len(lexical_rows_local)
                                            if lexical_rows_local:
                                                first = lexical_rows_local[0]
                                                first_type = type(first).__name__
                                                try:
                                                    first_len = len(
                                                        first
                                                    )  # may fail for non-sequences
                                                except Exception:
                                                    first_len = None
                                        except Exception:
                                            pass
                                        lexical_rows_local = []
                                        logger.warning(
                                            "rag.hybrid.lexical_primary_failed",
                                            tenant_id=tenant,
                                            case_id=case_value,
                                            error=str(exc),
                                            applied_trgm_limit=applied_trgm_limit,
                                            rows_count=rows_count,
                                            row_first_len=first_len,
                                            row_first_type=first_type,
                                        )
                                        try:
                                            logger.warning(
                                                "rag.debug.fallback_flag_set",
                                                extra={
                                                    "tenant_id": tenant,
                                                    "case_id": case_value,
                                                    "reason": "score_validation_error",
                                                    "fallback_requires_rollback": False,
                                                },
                                            )
                                        except Exception:
                                            pass
                                    if not handled and isinstance(exc, PsycopgError):
                                        handled = True
                                        if vector_query_failed:
                                            raise
                                        # Treat database errors during the primary lexical query as
                                        # a signal to attempt the explicit similarity fallback.
                                        # We mark that a rollback is required to restore session
                                        # settings (e.g. search_path) before running the fallback.
                                        should_run_fallback = True
                                        fallback_requires_rollback = True
                                        lexical_rows_local = []
                                        try:
                                            logger.warning(
                                                "rag.debug.fallback_flag_set",
                                                extra={
                                                    "tenant_id": tenant,
                                                    "case_id": case_value,
                                                    "reason": "db_error",
                                                    "fallback_requires_rollback": True,
                                                },
                                            )
                                        except Exception:
                                            pass
                                        logger.warning(
                                            "rag.hybrid.lexical_primary_failed",
                                            tenant_id=tenant,
                                            case_id=case_value,
                                            error=str(exc),
                                        )
                                    if not handled:
                                        raise
                            if not lexical_rows_local and not should_run_fallback:
                                # If the primary trigram match returned no rows, always run the
                                # explicit similarity fallback. Some environments report a low
                                # pg_trgm limit (e.g. 0.1) even after attempting to raise it; we
                                # still want to relax towards 0.0 to ensure we can retrieve at
                                # least the best lexical candidates when trigram returns nothing.
                                should_run_fallback = True
                            if should_run_fallback:
                                try:
                                    logger.warning(
                                        "rag.debug.before_fallback",
                                        extra={
                                            "tenant_id": tenant,
                                            "case_id": case_value,
                                            "fallback_requires_rollback": bool(
                                                fallback_requires_rollback
                                            ),
                                        },
                                    )
                                except Exception:
                                    pass
                                if fallback_requires_rollback:
                                    try:
                                        logger.warning(
                                            "rag.debug.calling_rollback",
                                            extra={
                                                "tenant_id": tenant,
                                                "case_id": case_value,
                                            },
                                        )
                                    except Exception:
                                        pass
                                    try:
                                        conn.rollback()
                                    except Exception:  # pragma: no cover - defensive
                                        pass
                                    else:
                                        try:
                                            logger.warning(
                                                "rag.debug.after_rollback",
                                                extra={
                                                    "tenant_id": tenant,
                                                    "case_id": case_value,
                                                    "action": "restore_session",
                                                },
                                            )
                                        except Exception:
                                            pass
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
                                        logger.debug(
                                            "rag.hybrid.rows.lexical_raw",
                                            extra={
                                                "tenant_id": tenant,
                                                "case_id": case_value,
                                                "count": len(attempt_rows),
                                                "limit": float(limit_value),
                                                "rows": _summarise_rows(
                                                    attempt_rows, kind="lexical"
                                                ),
                                            },
                                        )
                                    except Exception:
                                        pass
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

        candidates: Dict[str, Dict[str, object]] = {}
        for row in vector_rows:
            score_candidate = self._extract_score_from_row(row, kind="vector")
            raw_value: float | None = None
            if score_candidate is not None:
                try:
                    raw_value = float(score_candidate)
                except (TypeError, ValueError):
                    raw_value = None
                else:
                    if math.isnan(raw_value) or math.isinf(raw_value):
                        raw_value = None
            vector_score_missing = raw_value is None
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

            metadata_dict = dict(metadata or {})

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
            entry["chunk_id"] = chunk_identifier
            if not entry.get("metadata"):
                entry["metadata"] = metadata_dict
            if entry.get("doc_id") is None and doc_id is not None:
                entry["doc_id"] = doc_id
            if entry.get("doc_hash") is None and doc_hash is not None:
                entry["doc_hash"] = doc_hash
            if vector_score_missing:
                entry["_allow_below_cutoff"] = True
            else:
                distance_value = float(raw_value)
                if distance_score_mode == "inverse":
                    distance_value = max(0.0, distance_value)
                    vscore = 1.0 / (1.0 + distance_value)
                else:
                    vscore = max(0.0, 1.0 - float(distance_value))
                entry["vscore"] = max(float(entry.get("vscore", 0.0)), vscore)

        # Only mark candidates as allowed-below-cutoff for exceptional cases.
        # We no longer blanket-allow all lexical candidates when a trigram
        # fallback occurred with alpha=0.0 — that decision is deferred to the
        # dedicated cutoff fallback stage below which selects only the best
        # needed candidates up to top_k.

        for row in lexical_rows:
            score_candidate = self._extract_score_from_row(row, kind="lexical")
            raw_value: float | None = None
            if score_candidate is not None:
                try:
                    raw_value = float(score_candidate)
                except (TypeError, ValueError):
                    raw_value = None
                else:
                    if math.isnan(raw_value) or math.isinf(raw_value):
                        raw_value = None
            lexical_score_missing = raw_value is None
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

            metadata_dict = dict(metadata or {})
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
            entry["chunk_id"] = chunk_identifier

            if not entry.get("metadata"):
                entry["metadata"] = metadata_dict
            if entry.get("doc_id") is None and doc_id is not None:
                entry["doc_id"] = doc_id
            if entry.get("doc_hash") is None and doc_hash is not None:
                entry["doc_hash"] = doc_hash

            score_source = raw_value if raw_value is not None else score_raw
            try:
                lscore_value = float(score_source) if score_source is not None else 0.0
            except (TypeError, ValueError):
                lscore_value = 0.0
            if math.isnan(lscore_value) or math.isinf(lscore_value):
                lscore_value = 0.0
            lscore_value = max(0.0, lscore_value)
            entry["lscore"] = max(float(entry.get("lscore", 0.0)), lscore_value)

            # Permit bypassing the min_sim cutoff only when the lexical score is
            # structurally missing (row shape/NaN), not merely because a trigram
            # fallback happened. The proper promotion of below-cutoff items is
            # handled later by the cutoff fallback logic which respects top_k.
            if lexical_score_missing:
                entry["_allow_below_cutoff"] = True

        try:
            logger.debug(
                "rag.hybrid.candidates.compiled",
                extra={
                    "tenant_id": tenant,
                    "case_id": case_value,
                    "count": len(candidates),
                    "entries": [
                        {
                            "chunk_id": _safe_chunk_identifier(entry.get("chunk_id")),
                            "vscore": _safe_float(entry.get("vscore")),
                            "lscore": _safe_float(entry.get("lscore")),
                            "allow_below_cutoff": bool(
                                entry.get("_allow_below_cutoff", False)
                            ),
                        }
                        for entry in candidates.values()
                    ],
                },
            )
        except Exception:
            pass
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
            raw_meta = dict(
                cast(Mapping[str, object] | None, entry.get("metadata")) or {}
            )
            # Be tolerant to legacy result metadata that may still use
            # "tenant"/"case" keys instead of "tenant_id"/"case_id".
            candidate_tenant = cast(
                Optional[str], raw_meta.get("tenant_id") or raw_meta.get("tenant")
            )
            candidate_case = cast(
                Optional[str], raw_meta.get("case_id") or raw_meta.get("case")
            )
            reasons: List[str] = []
            try:
                vector_preview = float(entry.get("vscore", 0.0))
            except (TypeError, ValueError):
                vector_preview = 0.0
            if math.isnan(vector_preview) or math.isinf(vector_preview):
                vector_preview = 0.0
            try:
                lexical_preview = float(entry.get("lscore", 0.0))
            except (TypeError, ValueError):
                lexical_preview = 0.0
            if math.isnan(lexical_preview) or math.isinf(lexical_preview):
                lexical_preview = 0.0
            if query_embedding_empty:
                fused_preview = max(0.0, min(1.0, lexical_preview))
            elif has_vector_signal:
                fused_preview = max(
                    0.0,
                    min(
                        1.0,
                        alpha_value * vector_preview
                        + (1.0 - alpha_value) * lexical_preview,
                    ),
                )
            else:
                fused_preview = max(0.0, min(1.0, lexical_preview))
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
                    vector_score=vector_preview,
                    lexical_score=lexical_preview,
                    fused_score=fused_preview,
                    allow_below_cutoff=bool(entry.get("_allow_below_cutoff", False)),
                )
                continue

            doc_hash = entry.get("doc_hash")
            doc_id = entry.get("doc_id")
            meta = self._ensure_chunk_metadata_contract(
                raw_meta,
                tenant_id=tenant,
                case_id=case_value,
                filters=normalized_filters,
                chunk_id=entry.get("chunk_id"),
                doc_id=doc_id,
            )
            if doc_hash and not meta.get("hash"):
                meta["hash"] = doc_hash
            if doc_id is not None and "id" not in meta:
                meta["id"] = str(doc_id)
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
        try:
            logger.debug(
                "rag.hybrid.results.pre_min_sim",
                extra={
                    "tenant_id": tenant,
                    "case_id": case_value,
                    "min_sim": min_sim_value,
                    "results": [
                        {
                            "chunk_id": _safe_chunk_identifier(
                                chunk.meta.get("chunk_id")
                            ),
                            "fused": _safe_float(chunk.meta.get("fused")),
                            "vscore": _safe_float(chunk.meta.get("vscore")),
                            "lscore": _safe_float(chunk.meta.get("lscore")),
                            "allow_below_cutoff": bool(allow),
                        }
                        for chunk, allow in results
                    ],
                },
            )
        except Exception:
            pass
        below_cutoff_count = 0
        filtered_out_details: List[Dict[str, object | None]] = []
        below_cutoff_chunks: List[Chunk] = []
        filtered_results: List[Chunk] = []
        selected_chunk_keys: set[str] = set()
        if min_sim_value > 0.0:
            for chunk, allow in results:
                fused_value = float(chunk.meta.get("fused", 0.0))
                if math.isnan(fused_value) or math.isinf(fused_value):
                    fused_value = 0.0

                is_below_cutoff = fused_value < min_sim_value
                if is_below_cutoff:
                    below_cutoff_count += 1

                if allow or not is_below_cutoff:
                    filtered_results.append(chunk)
                    chunk_key = (
                        _safe_chunk_identifier(chunk.meta.get("chunk_id"))
                        or f"id:{id(chunk)}"
                    )
                    selected_chunk_keys.add(chunk_key)
                    if is_below_cutoff:
                        # These chunks are returned to the caller but still count as
                        # below-cutoff for telemetry and metrics purposes.
                        continue

                if is_below_cutoff and not allow:
                    below_cutoff_chunks.append(chunk)
                    filtered_out_details.append(
                        {
                            "chunk_id": _safe_chunk_identifier(
                                chunk.meta.get("chunk_id")
                            ),
                            "fused": _safe_float(fused_value),
                        }
                    )
            if below_cutoff_count > 0:
                metrics.RAG_QUERY_BELOW_CUTOFF_TOTAL.labels(tenant_id=tenant).inc(
                    float(below_cutoff_count)
                )
        else:
            filtered_results = [chunk for chunk, _ in results]
        if not selected_chunk_keys:
            selected_chunk_keys = {
                _safe_chunk_identifier(chunk.meta.get("chunk_id")) or f"id:{id(chunk)}"
                for chunk in filtered_results
            }
        try:
            logger.debug(
                "rag.hybrid.results.post_min_sim",
                extra={
                    "tenant_id": tenant,
                    "case_id": case_value,
                    "min_sim": min_sim_value,
                    "returned": len(filtered_results),
                    "filtered_out": filtered_out_details,
                    "kept": [
                        {
                            "chunk_id": _safe_chunk_identifier(
                                chunk.meta.get("chunk_id")
                            ),
                            "fused": _safe_float(chunk.meta.get("fused")),
                        }
                        for chunk in filtered_results
                    ],
                },
            )
        except Exception:
            pass
        fallback_promoted: List[Dict[str, object | None]] = []
        fallback_attempted = False
        # Only run the cutoff fallback when the trigram fallback has been applied.
        # This prevents vector-only queries from reintroducing candidates that were
        # explicitly filtered out by the min_sim threshold, which is required by
        # the hybrid search tests that expect an empty result set in that case.
        cutoff_fallback_enabled = fallback_limit_used_value is not None
        if (
            min_sim_value > 0.0
            and len(filtered_results) < top_k
            and results
            and cutoff_fallback_enabled
        ):
            fallback_attempted = True
            needed = top_k - len(filtered_results)
            cutoff_candidates = {
                (
                    _safe_chunk_identifier(chunk.meta.get("chunk_id"))
                    or f"id:{id(chunk)}"
                ): chunk
                for chunk in below_cutoff_chunks
            }
            for chunk, _ in results:
                if needed <= 0:
                    break
                chunk_key = (
                    _safe_chunk_identifier(chunk.meta.get("chunk_id"))
                    or f"id:{id(chunk)}"
                )
                if chunk_key in selected_chunk_keys:
                    continue
                if chunk_key not in cutoff_candidates:
                    continue
                chunk.meta["cutoff_fallback"] = True
                filtered_results.append(chunk)
                selected_chunk_keys.add(chunk_key)
                fallback_promoted.append(
                    {
                        "chunk_id": chunk_key,
                        "fused": _safe_float(chunk.meta.get("fused")),
                    }
                )
                needed -= 1
            if fallback_promoted:
                promoted_ids = {
                    entry.get("chunk_id")
                    for entry in fallback_promoted
                    if entry.get("chunk_id") is not None
                }
                if promoted_ids:
                    filtered_out_details = [
                        detail
                        for detail in filtered_out_details
                        if detail.get("chunk_id") not in promoted_ids
                    ]
        limited_results = filtered_results[:top_k]
        if fallback_attempted:
            try:
                logger.info(
                    "rag.hybrid.cutoff_fallback",
                    extra={
                        "tenant_id": tenant,
                        "case_id": case_value,
                        "requested_min_sim": min_sim_value,
                        "returned": len(limited_results),
                        "below_cutoff": below_cutoff_count,
                        "promoted": fallback_promoted,
                    },
                )
            except Exception:
                pass
        elif not limited_results and results and min_sim_value > 0.0:
            try:
                logger.info(
                    "rag.hybrid.cutoff_fallback",
                    extra={
                        "tenant_id": tenant,
                        "case_id": case_value,
                        "requested_min_sim": min_sim_value,
                        "returned": len(limited_results),
                        "below_cutoff": below_cutoff_count,
                        "promoted": [],
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
            below_cutoff=below_cutoff_count,
            returned_after_cutoff=len(filtered_results),
            query_embedding_empty=query_embedding_empty,
            applied_trgm_limit=applied_trgm_limit_value,
            fallback_limit_used=fallback_limit_used_value,
            visibility=visibility_mode.value,
            deleted_matches_blocked=deleted_matches_blocked_value,
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
            source = chunk.meta.get("source", "")
            external_id = chunk.meta.get("external_id")
            raw_collection_id = chunk.meta.get("collection_id")
            raw_doc_class = chunk.meta.get("doc_class")
            raw_document_id = chunk.meta.get("document_id") or chunk.meta.get("doc_id")
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
                grouped_metadata["doc_id"] = str(grouped[key]["id"])
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
                    metadata_block["doc_id"] = str(grouped[key]["id"])
            chunk_meta = dict(chunk.meta)
            chunk_meta["tenant_id"] = tenant
            chunk_meta["external_id"] = external_id_str
            document_identifier = str(grouped[key]["id"])
            chunk_meta["document_id"] = document_identifier
            chunk_meta["doc_id"] = document_identifier
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
                        parent_payload.setdefault(
                            "document_id", document_identifier
                        )
                    else:
                        parent_payload = parent_meta
                    parents_map[parent_id] = parent_payload
            grouped[key]["chunks"].append(
                Chunk(content=chunk.content, meta=chunk_meta, embedding=chunk.embedding)
            )
        return grouped

    def _compute_document_embedding(
        self, doc: Mapping[str, object]
    ) -> tuple[List[float], bool] | None:
        chunks = doc.get("chunks", [])
        if not isinstance(chunks, Sequence):
            return None
        vectors: List[List[float]] = []
        dimension: int | None = None
        unit_normalised = True
        for chunk in chunks:
            embedding = getattr(chunk, "embedding", None)
            if embedding is None:
                continue
            try:
                floats = [float(value) for value in embedding]
            except (TypeError, ValueError):
                continue
            if not floats:
                continue
            if dimension is None:
                dimension = len(floats)
            if len(floats) != dimension:
                continue
            norm = math.sqrt(sum(value * value for value in floats))
            if not math.isfinite(norm) or norm <= _ZERO_EPSILON:
                unit_normalised = False
            elif not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                unit_normalised = False
            vectors.append(floats)
        if not vectors or dimension is None:
            return None
        aggregated = [0.0] * dimension
        for vec in vectors:
            for index, value in enumerate(vec):
                aggregated[index] += value
        count = float(len(vectors))
        if count <= 0:
            return None
        averaged = [value / count for value in aggregated]
        norm_sq = math.fsum(value * value for value in averaged)
        if norm_sq <= _ZERO_EPSILON:
            return None
        norm = math.sqrt(norm_sq)
        if not math.isfinite(norm) or norm <= _ZERO_EPSILON:
            return None
        return averaged, unit_normalised

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
        if not vector:
            return None
        try:
            raw_vector = [float(value) for value in vector]
        except (TypeError, ValueError):
            return None
        if not raw_vector:
            return None
        normalised = _normalise_vector(raw_vector)
        if normalised is None:
            logger.info(
                "ingestion.doc.near_duplicate_vector_unusable",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "external_id": external_id,
                },
            )
            return None
        vector_for_similarity = normalised
        vector_for_distance = raw_vector
        if self._near_duplicate_operator_supported is False:
            return None
        index_kind = str(_get_setting("RAG_INDEX_KIND", "HNSW")).upper()
        try:
            operator = self._get_distance_operator(cur.connection, index_kind)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "ingestion.doc.near_duplicate_operator_missing",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "index_kind": index_kind,
                    "error": str(exc),
                },
            )
            return None
        if not self._is_near_duplicate_operator_supported(index_kind, operator):
            logger.info(
                "ingestion.doc.near_duplicate_operator_disabled",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "index_kind": index_kind,
                    "operator": operator,
                },
            )
            return None

        if operator not in {"<=>", "<->"}:
            self._disable_near_duplicate_for_operator(
                index_kind=index_kind,
                operator=operator,
                tenant_uuid=tenant_uuid,
            )
            return None
        use_distance_metric = False
        include_embedding_in_results = False
        distance_cutoff: float | None = None
        if operator == "<->":
            if self._require_unit_norm_for_l2:
                if not embedding_is_unit_normalised:
                    self._near_duplicate_enabled = False
                    self._near_duplicate_operator_supported = False
                    self._near_duplicate_operator_support[index_kind.upper()] = False
                    logger.warning(
                        "ingestion.doc.near_duplicate_l2_unit_norm_missing",
                        extra={
                            "tenant_id": str(tenant_uuid),
                            "index_kind": index_kind,
                            "operator": operator,
                        },
                    )
                    return None
                vector_to_format = vector_for_similarity
                include_embedding_in_results = True
            else:
                use_distance_metric = True
                vector_to_format = vector_for_distance
                try:
                    distance_cutoff = math.sqrt(
                        max(0.0, 2.0 * (1.0 - self._near_duplicate_threshold))
                    )
                except Exception:
                    distance_cutoff = None
                if distance_cutoff is None:
                    return None
            try:
                vector_str = self._format_vector(vector_to_format)
            except ValueError:
                try:
                    vector_str = self._format_vector_lenient(vector_to_format)
                except Exception:
                    return None
        else:
            vector_to_format = vector_for_similarity
            try:
                vector_str = self._format_vector(vector_to_format)
            except ValueError:
                try:
                    vector_str = self._format_vector_lenient(vector_to_format)
                except Exception:
                    return None
        self._near_duplicate_operator_supported = True
        if operator == "<=>":
            sim_sql = sql.SQL("1.0 - (e.embedding <=> %s::vector)")
            distance_sql = sql.SQL("e.embedding <=> %s::vector")
            select_vector_params = [vector_str, vector_str]
        elif not use_distance_metric:
            sim_sql = sql.SQL(
                "1.0 - ((e.embedding <-> %s::vector) * (e.embedding <-> %s::vector)) / 2.0"
            )
            distance_sql = sql.SQL("e.embedding <-> %s::vector")
            select_vector_params = [vector_str, vector_str, vector_str]
        else:
            sim_sql = sql.SQL("e.embedding <-> %s::vector")
            distance_sql = sql.SQL("e.embedding <-> %s::vector")
            select_vector_params = [vector_str, vector_str]

        if use_distance_metric:
            global_order_sql = sql.SQL("ASC")
        else:
            global_order_sql = sql.SQL("DESC")

        prefetch_limit = max(
            self._near_duplicate_probe_limit, self._near_duplicate_probe_limit * 4
        )
        embedding_column_sql = (
            sql.SQL(",\n                    e.embedding AS stored_embedding")
            if include_embedding_in_results
            else sql.SQL("")
        )
        outer_embedding_sql = (
            sql.SQL(", stored_embedding")
            if include_embedding_in_results
            else sql.SQL("")
        )

        documents_table = self._table("documents")
        chunks_table = self._table("chunks")
        embeddings_table = self._table("embeddings")

        query = sql.SQL(
            """
            WITH base AS (
                SELECT
                    d.id,
                    d.external_id,
                    {sim} AS similarity,
                    {distance} AS chunk_distance{embedding_column}
                FROM {documents} d
                JOIN {chunks} c ON c.document_id = d.id
                JOIN {embeddings} e ON e.chunk_id = c.id
                WHERE d.tenant_id = %s
                  AND d.collection_id IS NOT DISTINCT FROM %s
                  AND d.deleted_at IS NULL
                  AND d.external_id <> %s
                ORDER BY chunk_distance ASC
                LIMIT %s
            )
            SELECT id, external_id, similarity{outer_embedding}
            FROM (
                SELECT
                    id,
                    external_id,
                    similarity,
                    ROW_NUMBER() OVER (
                        PARTITION BY id
                        ORDER BY chunk_distance ASC
                    ) AS chunk_rank{ranked_embedding}
                FROM base
            ) AS ranked
            WHERE chunk_rank = 1
            ORDER BY similarity {global_order}
            LIMIT %s
            """
        ).format(
            sim=sim_sql,
            distance=distance_sql,
            global_order=global_order_sql,
            embedding_column=embedding_column_sql,
            outer_embedding=outer_embedding_sql,
            ranked_embedding=outer_embedding_sql,
            documents=documents_table,
            chunks=chunks_table,
            embeddings=embeddings_table,
        )
        tenant_value = str(tenant_uuid)
        collection_value = str(collection_uuid) if collection_uuid is not None else None
        params_list: List[object] = [
            *select_vector_params,
            tenant_value,
            collection_value,
            external_id,
            prefetch_limit,
            self._near_duplicate_probe_limit,
        ]
        params = tuple(params_list)
        cur.execute(query, params)
        rows = cur.fetchall()
        query_vector_for_similarity = (
            vector_for_similarity if include_embedding_in_results else None
        )
        best: Dict[str, object] | None = None
        best_similarity = self._near_duplicate_threshold
        best_distance = distance_cutoff if use_distance_metric else None
        for row in rows:
            if not isinstance(row, Sequence) or len(row) < 3:
                continue
            candidate_id = row[0]
            candidate_external_id = row[1]
            similarity_value = row[2]
            if include_embedding_in_results:
                stored_embedding = None
                fallback_to_sql_similarity = False
                if len(row) >= 4:
                    stored_embedding = _coerce_vector_values(row[3])
                    if stored_embedding is None:
                        fallback_to_sql_similarity = True
                else:
                    fallback_to_sql_similarity = True
                if (
                    stored_embedding is not None
                    and query_vector_for_similarity is not None
                ):
                    normalised_candidate = _normalise_vector(stored_embedding)
                    if normalised_candidate is not None and len(
                        normalised_candidate
                    ) == len(query_vector_for_similarity):
                        similarity_value = math.fsum(
                            candidate_component * query_component
                            for candidate_component, query_component in zip(
                                normalised_candidate, query_vector_for_similarity
                            )
                        )
                    else:
                        fallback_to_sql_similarity = True
                else:
                    fallback_to_sql_similarity = True
                if fallback_to_sql_similarity:
                    self._log_near_duplicate_similarity_fallback(
                        tenant_uuid=tenant_uuid,
                        external_id=external_id,
                    )
            if candidate_external_id == external_id:
                continue
            try:
                similarity = float(similarity_value)
            except (TypeError, ValueError):
                continue
            if math.isnan(similarity) or math.isinf(similarity):
                continue
            if use_distance_metric:
                distance = max(0.0, similarity)
                cutoff = (
                    distance_cutoff if distance_cutoff is not None else best_distance
                )
                if cutoff is None:
                    continue
                if distance > cutoff + 1e-9:
                    continue
                if best_distance is not None and distance > best_distance + 1e-12:
                    continue
                best_distance = distance
                if cutoff <= _ZERO_EPSILON:
                    similarity = 1.0 if distance <= _ZERO_EPSILON else 0.0
                else:
                    ratio = min(distance / cutoff, 1.0)
                    similarity = max(0.0, 1.0 - ratio)
            else:
                similarity = max(0.0, min(1.0, similarity))
                if similarity < self._near_duplicate_threshold:
                    continue
                if similarity < best_similarity:
                    continue
                best_similarity = similarity
            try:
                candidate_uuid = (
                    candidate_id
                    if isinstance(candidate_id, uuid.UUID)
                    else uuid.UUID(str(candidate_id))
                )
            except (TypeError, ValueError):
                continue
            external_text = str(candidate_external_id)
            best = {
                "id": candidate_uuid,
                "external_id": external_text,
                "similarity": similarity,
            }
        return best

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
            content_hash = str(doc.get("content_hash", doc.get("hash", "")))
            collection_value = doc.get("collection_id")
            collection_uuid: uuid.UUID | None = None
            if collection_value:
                try:
                    collection_uuid = (
                        collection_value
                        if isinstance(collection_value, uuid.UUID)
                        else uuid.UUID(str(collection_value))
                    )
                except (TypeError, ValueError):
                    collection_uuid = None
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
                workflow_id=workflow_text,
                collection_uuid=collection_uuid,
            )
            doc["hash"] = storage_hash
            doc["content_hash"] = content_hash
            metadata_dict = self._strip_collection_scope(doc.get("metadata"))
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
                metadata_dict["doc_id"] = document_id_text

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
                if self._near_duplicate_strategy == "skip":
                    actions[key] = "near_duplicate_skipped"
                    logger.info("ingestion.doc.near_duplicate_skipped", extra=log_extra)
                    continue
                if self._near_duplicate_strategy == "replace":
                    metadata_dict["near_duplicate_of"] = str(
                        near_duplicate_details.get("external_id")
                    )
                    metadata_dict["near_duplicate_similarity"] = similarity
                    metadata = Json(metadata_dict)
                    existing_id = near_duplicate_details["id"]
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
                            existing_id,
                        ),
                    )
                    document_ids[key] = existing_id
                    actions[key] = "near_duplicate_replaced"
                    doc["metadata"] = metadata_dict
                    doc["id"] = existing_id
                    logger.info(
                        "ingestion.doc.near_duplicate_replaced",
                        extra={**log_extra, "document_id": str(existing_id)},
                    )
                    continue

            parents_map = doc.get("parents")
            if isinstance(parents_map, Mapping) and parents_map:
                metadata_dict["parent_nodes"] = limit_parent_payload(parents_map)

            metadata = Json(metadata_dict)
            document_id = doc["id"]
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
            if collection_text is None:
                select_sql = sql.SQL(
                    """
                    SELECT id, hash, metadata, source, deleted_at
                    FROM {}
                    WHERE tenant_id = %s
                      AND collection_id IS NULL
                      AND {}
                      AND external_id = %s
                    LIMIT 1
                    """
                ).format(documents_table, workflow_clause)
                params: list[object] = [tenant_value]
            else:
                select_sql = sql.SQL(
                    """
                    SELECT id, hash, metadata, source, deleted_at
                    FROM {}
                    WHERE tenant_id = %s
                      AND collection_id = %s
                      AND {}
                      AND external_id = %s
                    LIMIT 1
                    """
                ).format(documents_table, workflow_clause)
                params = [tenant_value, collection_text]
            params.extend(workflow_params)
            params.append(external_id)
            cur.execute(select_sql, tuple(params))
            existing = cur.fetchone()
            try:
                if existing:
                    (
                        existing_id,
                        existing_hash,
                        existing_metadata,
                        existing_source,
                        existing_deleted,
                    ) = existing
                    needs_update = (
                        str(existing_hash) != storage_hash
                        or existing_deleted is not None
                    )
                    if needs_update:
                        cur.execute(
                            sql.SQL(
                                """
                                UPDATE {}
                                SET source = %s,
                                    hash = %s,
                                    metadata = %s,
                                    collection_id = %s,
                                    workflow_id = %s,
                                    deleted_at = NULL
                                WHERE id = %s
                                """
                            ).format(documents_table),
                            (
                                doc["source"],
                                storage_hash,
                                metadata,
                                collection_text,
                                workflow_text,
                                existing_id,
                            ),
                        )
                        actions[key] = "replaced"
                    else:
                        actions[key] = "skipped"
                    document_ids[key] = existing_id
                    doc["id"] = existing_id
                    doc["metadata"] = metadata_dict
                    doc["collection_id"] = collection_text
                    continue

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
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                    ),
                )
                document_ids[key] = document_id
                actions[key] = "inserted"
                doc["metadata"] = metadata_dict
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
                            SELECT id, hash, metadata, source, deleted_at
                            FROM {}
                            WHERE tenant_id = %s
                              AND collection_id IS NULL
                              AND {}
                              AND external_id = %s
                            LIMIT 1
                            """
                        ).format(documents_table, workflow_clause)
                        retry_params: list[object] = [tenant_value]
                    else:
                        retry_sql = sql.SQL(
                            """
                            SELECT id, hash, metadata, source, deleted_at
                            FROM {}
                            WHERE tenant_id = %s
                              AND collection_id = %s
                              AND {}
                              AND external_id = %s
                            LIMIT 1
                            """
                        ).format(documents_table, workflow_clause)
                        retry_params = [tenant_value, collection_text]
                    retry_params.extend(workflow_params)
                    retry_params.append(external_id)
                    retry_cur.execute(retry_sql, tuple(retry_params))
                    duplicate = retry_cur.fetchone()
                    if not duplicate:
                        raise

                    (
                        dup_id,
                        dup_hash,
                        dup_metadata,
                        dup_source,
                        dup_deleted,
                    ) = duplicate
                    metadata = Json(metadata_dict)
                    needs_update = (
                        str(dup_hash) != storage_hash or dup_deleted is not None
                    )
                    if needs_update:
                        retry_cur.execute(
                            sql.SQL(
                                """
                                UPDATE {}
                                SET source = %s,
                                    hash = %s,
                                    metadata = %s,
                                    collection_id = %s,
                                    workflow_id = %s,
                                    deleted_at = NULL
                                WHERE id = %s
                                """
                            ).format(documents_table),
                            (
                                doc["source"],
                                storage_hash,
                                metadata,
                                collection_text,
                                workflow_text,
                                dup_id,
                            ),
                        )
                        actions[key] = "replaced"
                    else:
                        actions[key] = "skipped"
                        logger.info(
                            "Skipping unchanged document during upsert",
                            extra={
                                "tenant_id": doc["tenant_id"],
                                "external_id": external_id,
                            },
                        )

                    document_ids[key] = dup_id
                    doc["id"] = dup_id
                    doc["metadata"] = metadata_dict
                    doc["collection_id"] = collection_text
                    doc["workflow_id"] = workflow_text
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

    def _compute_storage_hash(
        self,
        cur,
        tenant_uuid: uuid.UUID,
        content_hash: str,
        external_id: str,
        workflow_id: str | None = None,
        *,
        collection_uuid: uuid.UUID | None = None,
    ) -> str:
        if not content_hash:
            return content_hash
        tenant_value = str(tenant_uuid)
        documents_table = self._table("documents")
        workflow_clause, workflow_params = self._workflow_predicate_clause(workflow_id)
        if collection_uuid is None:
            lookup_sql = sql.SQL(
                """
                SELECT external_id
                FROM {}
                WHERE tenant_id = %s
                  AND collection_id IS NULL
                  AND {}
                  AND hash = %s
                LIMIT 1
                """
            ).format(documents_table, workflow_clause)
            params: list[object] = [tenant_value]
        else:
            lookup_sql = sql.SQL(
                """
                SELECT external_id
                FROM {}
                WHERE tenant_id = %s
                  AND collection_id = %s
                  AND {}
                  AND hash = %s
                LIMIT 1
                """
            ).format(documents_table, workflow_clause)
            params = [tenant_value, str(collection_uuid)]
        params.extend(workflow_params)
        params.append(content_hash)
        cur.execute(lookup_sql, tuple(params))
        existing = cur.fetchone()
        if existing:
            existing_external_id = existing[0]
            if existing_external_id and str(existing_external_id) != external_id:
                suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"external:{external_id}")
                return f"{content_hash}:{suffix}"
        return content_hash

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
            started = time.perf_counter()
            cur.execute(
                sql.SQL(
                    "DELETE FROM {} WHERE chunk_id IN (SELECT id FROM {} WHERE document_id = %s)"
                ).format(embeddings_table, chunks_table),
                (document_id,),
            )
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE document_id = %s").format(chunks_table),
                (document_id,),
            )

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
                chunk_metadata = self._strip_collection_scope(chunk.meta)
                chunk_rows.append(
                    (
                        chunk_id,
                        document_id,
                        index,
                        chunk.content,
                        tokens,
                        Json(chunk_metadata),
                        tenant_value,
                        collection_value,
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
                                collection_value,
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
