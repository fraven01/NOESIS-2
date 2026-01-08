"""PostgreSQL-backed embedding cache helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

from psycopg2 import sql

from common.logging import get_logger

from .normalization import normalise_text
from .vector_client import PgVectorClient
from .vector_utils import _coerce_vector_values

logger = get_logger(__name__)


DEFAULT_EMBEDDING_CACHE_TTL_DAYS = 90


def compute_text_hash(text: str) -> str:
    normalised = normalise_text(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def fetch_cached_embeddings(
    client: PgVectorClient,
    text_hashes: Sequence[str],
    *,
    model_version: str,
    now: datetime | None = None,
) -> dict[str, tuple[list[float], datetime]]:
    cleaned = [value for value in text_hashes if value]
    if not cleaned:
        return {}
    timestamp = now or datetime.now(timezone.utc)
    table = client._table("embedding_cache")
    query = sql.SQL(
        """
        SELECT text_hash, embedding, created_at
          FROM {}
         WHERE text_hash = ANY(%s)
           AND model_version = %s
           AND expires_at > %s
        """
    ).format(table)
    results: dict[str, tuple[list[float], datetime]] = {}
    with client.connection() as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cur:
            cur.execute(query, (cleaned, model_version, timestamp))
            for text_hash, embedding, created_at in cur.fetchall():
                vector = _coerce_vector_values(embedding)
                if vector is None:
                    continue
                if not isinstance(created_at, datetime):
                    created_at = timestamp
                results[str(text_hash)] = (vector, created_at)
    return results


def store_cached_embeddings(
    client: PgVectorClient,
    *,
    embeddings: Mapping[str, Sequence[float]],
    model_version: str,
    created_at: datetime | None = None,
    ttl_days: int = DEFAULT_EMBEDDING_CACHE_TTL_DAYS,
) -> int:
    if not embeddings:
        return 0
    created_at_value = created_at or datetime.now(timezone.utc)
    expires_at = created_at_value + timedelta(days=int(ttl_days))
    table = client._table("embedding_cache")
    insert_sql = sql.SQL(
        """
        INSERT INTO {} (text_hash, model_version, embedding, created_at, expires_at)
        VALUES (%s, %s, %s::vector, %s, %s)
        ON CONFLICT (text_hash, model_version) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                created_at = EXCLUDED.created_at,
                expires_at = EXCLUDED.expires_at
        """
    ).format(table)
    rows: list[tuple[str, str, str, datetime, datetime]] = []
    for text_hash, vector in embeddings.items():
        vector_value = client._format_vector(vector)
        rows.append(
            (
                str(text_hash),
                model_version,
                vector_value,
                created_at_value,
                expires_at,
            )
        )
    with client.connection() as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cur:
            cur.executemany(insert_sql.as_string(conn), rows)
        conn.commit()
    return len(rows)
