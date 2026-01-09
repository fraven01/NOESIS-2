from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from .query_builder import build_vector_sql


@dataclass(frozen=True)
class VectorSearchOutcome:
    rows: List[tuple]
    distance_operator: str | None
    vector_query_failed: bool


def run_vector_search(
    *,
    client: object,
    conn,
    query_vec: str | None,
    vector_format_error: Exception | None,
    index_kind: str,
    ef_search: int,
    probes: int,
    where_sql: str,
    where_params: Sequence[object],
    vec_limit: int,
    tenant: str,
    case_id: str | None,
    statement_timeout_ms: int,
    summarise_rows: Callable[[Iterable[object], str], list[dict[str, object | None]]],
    logger,
) -> VectorSearchOutcome:
    vector_rows: List[tuple] = []
    vector_query_failed = vector_format_error is not None
    distance_operator_value: str | None = None

    if vector_format_error is not None:
        try:
            conn.rollback()
        except Exception:  # pragma: no cover - defensive
            pass
        logger.warning(
            "rag.hybrid.vector_query_failed",
            tenant_id=tenant,
            tenant=tenant,
            case_id=case_id,
            error=str(vector_format_error),
        )
    elif query_vec is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SET LOCAL statement_timeout = %s",
                    (str(statement_timeout_ms),),
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
                distance_operator = client._get_distance_operator(conn, index_kind)
                distance_operator_value = distance_operator
                vector_sql = build_vector_sql(
                    where_sql,
                    "c.id,\n"
                    "                                    c.text,\n"
                    "                                    c.metadata,\n"
                    "                                    d.hash,\n"
                    "                                    d.id,\n"
                    "                                    c.collection_id,\n"
                    "                                    e.embedding "
                    + f"{distance_operator} %s::vector AS distance",
                    "distance",
                )
                # Bind parameters in textual order: SELECT (vector), WHERE params, LIMIT
                cur.execute(vector_sql, (query_vec, *where_params, vec_limit))
                vector_rows = cur.fetchall()
                try:
                    logger.debug(
                        "rag.hybrid.rows.vector_raw",
                        extra={
                            "tenant_id": tenant,
                            "case_id": case_id,
                            "count": len(vector_rows),
                            "rows": summarise_rows(vector_rows, kind="vector"),
                        },
                    )
                except Exception:
                    pass
                try:
                    logger.warning(
                        "rag.debug.rows.vector",
                        extra={
                            "count": len(vector_rows),
                            "first_len": (len(vector_rows[0]) if vector_rows else 0),
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
                case_id=case_id,
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
                    (str(statement_timeout_ms),),
                )
                cur.execute(
                    "SELECT 1 FROM embeddings e LIMIT %s",
                    (vec_limit,),
                )
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

    return VectorSearchOutcome(
        rows=vector_rows,
        distance_operator=distance_operator_value,
        vector_query_failed=vector_query_failed,
    )
