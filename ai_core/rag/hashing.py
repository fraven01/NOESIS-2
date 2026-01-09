"""Hashing utilities for RAG ingestion and deduplication."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Mapping, Optional, Sequence

from psycopg2 import sql

from common.logging import get_logger

from .deduplication import DedupSignatures, compute_near_duplicate_signature

__all__ = [
    "build_dedup_signatures",
    "compute_content_hash",
    "compute_storage_hash",
    "extract_primary_text_hash",
]

logger = get_logger(__name__)

_HASH_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIMARY_TEXT_HASH_KEYS = {"sha256": "crawler.primary_text_hash_sha256"}


def _hash_payload(payload: bytes, algorithm: str) -> str:
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError("unsupported_hash_algorithm") from exc
    hasher.update(payload)
    return hasher.hexdigest()


def extract_primary_text_hash(
    parse_stats: Mapping[str, object] | None, algorithm: str
) -> Optional[str]:
    """Return the stored primary text hash encoded in *parse_stats*."""

    key = _PRIMARY_TEXT_HASH_KEYS.get(algorithm.lower())
    if key is None:
        return None
    if not isinstance(parse_stats, Mapping):
        return None
    raw = parse_stats.get(key)
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    if not candidate:
        return None
    if not _HASH_HEX_RE.fullmatch(candidate):
        return None
    return candidate


def compute_content_hash(
    *,
    normalized_primary_text: str,
    stored_primary_text_hash: Optional[str],
    payload_bytes: bytes,
    algorithm: str = "sha256",
) -> str:
    """Return a deterministic content hash for crawler documents."""

    if normalized_primary_text:
        payload = normalized_primary_text.encode("utf-8")
        return _hash_payload(payload, algorithm)
    if stored_primary_text_hash:
        return stored_primary_text_hash
    return _hash_payload(payload_bytes, algorithm)


def build_dedup_signatures(
    *,
    primary_text: Optional[str],
    normalized_primary_text: str,
    stored_primary_text_hash: Optional[str],
    payload_bytes: bytes,
    algorithm: str = "sha256",
) -> DedupSignatures:
    content_hash = compute_content_hash(
        normalized_primary_text=normalized_primary_text,
        stored_primary_text_hash=stored_primary_text_hash,
        payload_bytes=payload_bytes,
        algorithm=algorithm,
    )
    near_signature = compute_near_duplicate_signature(primary_text)
    return DedupSignatures(content_hash=content_hash, near_duplicate=near_signature)


def compute_storage_hash(
    *,
    cur,
    documents_table: sql.Identifier,
    workflow_clause: sql.Composable,
    workflow_params: Sequence[object],
    tenant_uuid: uuid.UUID,
    content_hash: str,
    external_id: str,
    source: str,
    log: object | None = None,
) -> str:
    """Return a stable storage hash with collision avoidance."""

    if not content_hash:
        return content_hash
    tenant_value = str(tenant_uuid)
    lookup_sql = sql.SQL(
        """
        SELECT external_id
        FROM {}
        WHERE tenant_id = %s
          AND {}
          AND source = %s
          AND hash = %s
        LIMIT 1
        """
    ).format(documents_table, workflow_clause)
    params: list[object] = [tenant_value]
    params.extend(workflow_params)
    params.extend([source, content_hash])
    cur.execute(lookup_sql, tuple(params))
    existing = cur.fetchone()
    if existing:
        existing_external_id = existing[0]
        if existing_external_id and str(existing_external_id) != external_id:
            active_logger = log if log is not None else logger
            try:
                active_logger.debug(  # type: ignore[call-arg]
                    "ingestion.doc.hash_reused",
                    extra={
                        "tenant_id": tenant_value,
                        "source": source,
                        "hash": content_hash,
                        "existing_external_id": str(existing_external_id),
                        "incoming_external_id": external_id,
                    },
                )
            except Exception:
                pass
            composite = f"{content_hash}|{external_id}"
            return hashlib.sha256(composite.encode("utf-8")).hexdigest()
    return content_hash
