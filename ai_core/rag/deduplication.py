"""Deduplication helpers for document ingestion."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from psycopg2 import sql

from common.logging import get_logger

from .vector_utils import _coerce_vector_values, _normalise_vector

__all__ = [
    "DedupSignatures",
    "NearDuplicateMatch",
    "NearDuplicateSignature",
    "compute_document_embedding",
    "compute_near_duplicate_signature",
    "find_near_duplicate",
    "match_near_duplicate",
]

_NEAR_DUPLICATE_TOKEN_RE = re.compile(r"\w+")
_ZERO_EPSILON = 1e-12

logger = get_logger(__name__)


@dataclass(frozen=True)
class NearDuplicateSignature:
    """Set-based token signature used for near-duplicate checks."""

    fingerprint: str
    tokens: Tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_tokens = tuple(_normalize_dedup_token_sequence(self.tokens))
        object.__setattr__(self, "tokens", normalized_tokens)
        if not self.fingerprint:
            raise ValueError("near_duplicate_fingerprint_required")


@dataclass(frozen=True)
class DedupSignatures:
    """Stable signatures describing a document for deduplication."""

    content_hash: str
    near_duplicate: Optional[NearDuplicateSignature] = None

    def __post_init__(self) -> None:
        if not self.content_hash:
            raise ValueError("content_hash_required")


@dataclass(frozen=True)
class NearDuplicateMatch:
    """Result of comparing a signature with known near-duplicates."""

    document_id: str
    similarity: float


def _normalize_dedup_token_sequence(tokens: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(
        sorted({token.strip().lower() for token in tokens if token and token.strip()})
    )
    if not normalized:
        raise ValueError("near_duplicate_tokens_required")
    return normalized


def _tokenize_near_duplicate_text(text: str) -> Tuple[str, ...]:
    return tuple(_NEAR_DUPLICATE_TOKEN_RE.findall(text.lower()))


def compute_near_duplicate_signature(
    primary_text: Optional[str],
) -> Optional[NearDuplicateSignature]:
    """Return a stable token signature for *primary_text* if available."""

    text = (primary_text or "").strip()
    if not text:
        return None
    tokens = _tokenize_near_duplicate_text(text)
    if not tokens:
        return None
    normalized_tokens = tuple(sorted(set(tokens)))
    fingerprint_source = "\u001f".join(normalized_tokens)
    fingerprint = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()
    return NearDuplicateSignature(fingerprint=fingerprint, tokens=normalized_tokens)


def compute_document_embedding(
    doc: Mapping[str, object],
) -> tuple[list[float], bool] | None:
    chunks = doc.get("chunks", [])
    if not isinstance(chunks, Sequence):
        return None
    vectors: list[list[float]] = []
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


def find_near_duplicate(
    *,
    client,
    cur,
    tenant_uuid: uuid.UUID,
    vector: Sequence[float],
    external_id: str,
    embedding_is_unit_normalised: bool,
    collection_uuid: uuid.UUID | None = None,
    get_setting: Callable[[str, str], str],
    log=None,
) -> Dict[str, object] | None:
    active_logger = log or logger
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
        try:
            active_logger.info(
                "ingestion.doc.near_duplicate_vector_unusable",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "external_id": external_id,
                },
            )
        except Exception:
            pass
        return None
    vector_for_similarity = normalised
    vector_for_distance = raw_vector
    if client._near_duplicate_operator_supported is False:  # type: ignore[attr-defined]
        return None
    index_kind = str(get_setting("RAG_INDEX_KIND", "HNSW")).upper()
    try:
        operator = client._get_distance_operator(cur.connection, index_kind)
    except Exception as exc:  # pragma: no cover - defensive
        try:
            active_logger.warning(
                "ingestion.doc.near_duplicate_operator_missing",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "index_kind": index_kind,
                    "error": str(exc),
                },
            )
        except Exception:
            pass
        return None
    if not client._is_near_duplicate_operator_supported(index_kind, operator):
        try:
            active_logger.info(
                "ingestion.doc.near_duplicate_operator_disabled",
                extra={
                    "tenant_id": str(tenant_uuid),
                    "index_kind": index_kind,
                    "operator": operator,
                },
            )
        except Exception:
            pass
        return None

    if operator not in {"<=>", "<->"}:
        client._disable_near_duplicate_for_operator(  # type: ignore[attr-defined]
            index_kind=index_kind,
            operator=operator,
            tenant_uuid=tenant_uuid,
        )
        return None
    use_distance_metric = False
    include_embedding_in_results = False
    distance_cutoff: float | None = None
    if operator == "<->":
        if client._require_unit_norm_for_l2:  # type: ignore[attr-defined]
            if not embedding_is_unit_normalised:
                client._near_duplicate_enabled = False  # type: ignore[attr-defined]
                client._near_duplicate_operator_supported = False  # type: ignore[attr-defined]
                client._near_duplicate_operator_support[  # type: ignore[attr-defined]
                    index_kind.upper()
                ] = False
                try:
                    active_logger.warning(
                        "ingestion.doc.near_duplicate_l2_unit_norm_missing",
                        extra={
                            "tenant_id": str(tenant_uuid),
                            "index_kind": index_kind,
                            "operator": operator,
                        },
                    )
                except Exception:
                    pass
                return None
            vector_to_format = vector_for_similarity
            include_embedding_in_results = True
        else:
            use_distance_metric = True
            vector_to_format = vector_for_distance
            try:
                distance_cutoff = math.sqrt(
                    max(0.0, 2.0 * (1.0 - client._near_duplicate_threshold))  # type: ignore[attr-defined]
                )
            except Exception:
                distance_cutoff = None
            if distance_cutoff is None:
                return None
        try:
            vector_str = client._format_vector(vector_to_format)
        except ValueError:
            try:
                vector_str = client._format_vector_lenient(vector_to_format)
            except Exception:
                return None
    else:
        vector_to_format = vector_for_similarity
        try:
            vector_str = client._format_vector(vector_to_format)
        except ValueError:
            try:
                vector_str = client._format_vector_lenient(vector_to_format)
            except Exception:
                return None
    client._near_duplicate_operator_supported = True  # type: ignore[attr-defined]
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
        client._near_duplicate_probe_limit,  # type: ignore[attr-defined]
        client._near_duplicate_probe_limit * 4,  # type: ignore[attr-defined]
    )
    embedding_column_sql = (
        sql.SQL(",\n                    e.embedding AS stored_embedding")
        if include_embedding_in_results
        else sql.SQL("")
    )
    outer_embedding_sql = (
        sql.SQL(", stored_embedding") if include_embedding_in_results else sql.SQL("")
    )

    documents_table = client._table("documents")
    chunks_table = client._table("chunks")
    embeddings_table = client._table("embeddings")

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
              AND COALESCE(d.lifecycle, 'active') = 'active'
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
    params_list: list[object] = [
        *select_vector_params,
        tenant_value,
        collection_value,
        external_id,
        prefetch_limit,
        client._near_duplicate_probe_limit,  # type: ignore[attr-defined]
    ]
    params = tuple(params_list)
    cur.execute(query, params)
    rows = cur.fetchall()
    query_vector_for_similarity = (
        vector_for_similarity if include_embedding_in_results else None
    )
    best: Dict[str, object] | None = None
    best_similarity = client._near_duplicate_threshold  # type: ignore[attr-defined]
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
            if stored_embedding is not None and query_vector_for_similarity is not None:
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
                client._log_near_duplicate_similarity_fallback(  # type: ignore[attr-defined]
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
            cutoff = distance_cutoff if distance_cutoff is not None else best_distance
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
            if similarity < client._near_duplicate_threshold:  # type: ignore[attr-defined]
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


def match_near_duplicate(
    signature: Optional[NearDuplicateSignature],
    known: Optional[Mapping[str, NearDuplicateSignature]],
    *,
    threshold: float,
    exclude: Optional[str] = None,
) -> Optional[NearDuplicateMatch]:
    """Compare *signature* with known near-duplicates and return the best match."""

    if signature is None or not known:
        return None
    best_id: Optional[str] = None
    best_similarity = 0.0
    signature_tokens = set(signature.tokens)
    for doc_id, candidate in known.items():
        if exclude is not None and doc_id == exclude:
            continue
        similarity = _jaccard_similarity(signature_tokens, set(candidate.tokens))
        if similarity > best_similarity:
            best_id = doc_id
            best_similarity = similarity
    if best_id is None or best_similarity < threshold:
        return None
    return NearDuplicateMatch(best_id, best_similarity)


def _jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    set_left = set(left)
    set_right = set(right)
    if not set_left and not set_right:
        return 1.0
    union = set_left | set_right
    if not union:
        return 0.0
    return len(set_left & set_right) / len(union)
