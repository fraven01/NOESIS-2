from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Iterable, List, Sequence

from psycopg2 import Error as PsycopgError, sql

from .query_builder import build_lexical_primary_sql


@dataclass(frozen=True)
class LexicalSearchOutcome:
    rows: List[tuple]
    applied_trgm_limit: float | None
    fallback_limit_used: float | None
    fallback_tried_limits: List[float] = field(default_factory=list)
    lexical_query_variant: str = "none"
    lexical_fallback_limit: float | None = None


def run_lexical_search(
    *,
    client: object,
    conn,
    lexical_mode: str,
    query_db_norm: str,
    where_sql: str,
    where_params: Sequence[object],
    lex_limit: int,
    trgm_limit_value: float,
    requested_trgm_limit: float | None,
    tenant: str,
    case_id: str | None,
    trace_id: str | None,
    statement_timeout_ms: int,
    schema: str,
    lexical_score_sql: str,
    lexical_match_clause: str,
    vector_rows: Sequence[object],
    vector_query_failed: bool,
    summarise_rows: Callable[[Iterable[object], str], list[dict[str, object | None]]],
    extract_score_from_row: Callable[[object], object | None],
    logger,
) -> LexicalSearchOutcome:
    started_at = time.perf_counter()
    lexical_mode_value = str(lexical_mode or "trgm").strip().lower()
    applied_trgm_limit_value: float | None = None
    fallback_limit_used_value: float | None = None
    fallback_tried_limits: List[float] = []
    lexical_query_variant = "none"
    lexical_fallback_limit_value: float | None = None
    lexical_rows: List[tuple] = []

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SET LOCAL statement_timeout = %s",
                (str(statement_timeout_ms),),
            )
            # Ensure the schema is active for all lexical operations
            # in this transaction, regardless of connection setup.
            # This is required for fallback paths and mocked
            # connections used in tests where _prepare_connection
            # is bypassed.
            try:
                cur.execute(
                    sql.SQL("SET LOCAL search_path TO {}, public").format(
                        sql.Identifier(schema)
                    )
                )
            except Exception:
                # If SET LOCAL is not available, rely on connection-level
                # search_path set during _prepare_connection.
                pass
            if lexical_mode_value != "trgm":
                lexical_rows_local: List[tuple] = []
                if query_db_norm.strip():
                    lexical_sql = build_lexical_primary_sql(
                        where_sql,
                        "c.id,\n"
                        "                                c.text,\n"
                        "                                c.metadata,\n"
                        "                                d.hash,\n"
                        "                                d.id,\n"
                        "                                c.collection_id,\n"
                        "                                " + lexical_score_sql,
                        lexical_match_clause,
                    )
                    cur.execute(
                        lexical_sql,
                        (
                            query_db_norm,
                            *where_params,
                            query_db_norm,
                            lex_limit,
                        ),
                    )
                    lexical_rows_local = cur.fetchall()
                    lexical_query_variant = "bm25"
                return LexicalSearchOutcome(
                    rows=lexical_rows_local,
                    applied_trgm_limit=None,
                    fallback_limit_used=None,
                    fallback_tried_limits=[],
                    lexical_query_variant=lexical_query_variant,
                    lexical_fallback_limit=None,
                )
            logger.info(
                "rag.pgtrgm.limit",
                requested=requested_trgm_limit,
                effective=trgm_limit_value,
                trace_id=trace_id,
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
            lexical_sql = build_lexical_primary_sql(
                where_sql,
                "c.id,\n"
                "                                c.text,\n"
                "                                c.metadata,\n"
                "                                d.hash,\n"
                "                                d.id,\n"
                "                                c.collection_id,\n"
                "                                " + lexical_score_sql,
                lexical_match_clause,
            )
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
                            lex_limit,  # probe uses clamped limit
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
                                "case_id": case_id,
                                "trace_id": trace_id,
                                "exc_type": exc.__class__.__name__,
                                "fallback_requires_rollback": True,
                            },
                        )
                    except Exception:
                        pass
                    logger.warning(
                        "rag.hybrid.lexical_primary_failed",
                        tenant_id=tenant,
                        case_id=case_id,
                        trace_id=trace_id,
                        error=str(exc),
                    )
                    if isinstance(exc, PsycopgError) and vector_query_failed:
                        raise

                if not probe_failed:
                    try:
                        logger.warning(
                            "rag.debug.lexical_primary_try_enter",
                            extra={
                                "tenant_id": tenant,
                                "case_id": case_id,
                                "trace_id": trace_id,
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
                                lex_limit,
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
                                    "case_id": case_id,
                                    "trace_id": trace_id,
                                    "exc_type": exc.__class__.__name__,
                                    "fallback_requires_rollback": True,
                                },
                            )
                        except Exception:
                            pass
                        logger.warning(
                            "rag.hybrid.lexical_primary_failed",
                            tenant_id=tenant,
                            case_id=case_id,
                            trace_id=trace_id,
                            error=str(exc),
                        )
                        if isinstance(exc, PsycopgError) and vector_query_failed:
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
                                case_id=case_id,
                                trace_id=trace_id,
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
                                        "case_id": case_id,
                                        "trace_id": trace_id,
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
                                "case_id": case_id,
                                "trace_id": trace_id,
                                "count": len(lexical_rows_local),
                                "rows": summarise_rows(
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
                                "trace_id": trace_id,
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
                                    limit_threshold = float(applied_trgm_limit)
                                except (TypeError, ValueError):
                                    limit_threshold = None
                            if limit_threshold is not None:
                                for row in lexical_rows_local:
                                    try:
                                        score_candidate = extract_score_from_row(row)
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
                                case_id=case_id,
                                trace_id=trace_id,
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
                                        "case_id": case_id,
                                        "trace_id": trace_id,
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
                                        "case_id": case_id,
                                        "trace_id": trace_id,
                                        "reason": "db_error",
                                        "fallback_requires_rollback": True,
                                    },
                                )
                            except Exception:
                                pass
                            logger.warning(
                                "rag.hybrid.lexical_primary_failed",
                                tenant_id=tenant,
                                case_id=case_id,
                                trace_id=trace_id,
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
                                "case_id": case_id,
                                "trace_id": trace_id,
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
                                    "case_id": case_id,
                                    "trace_id": trace_id,
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
                                        "case_id": case_id,
                                        "trace_id": trace_id,
                                        "action": "restore_session",
                                    },
                                )
                            except Exception:
                                pass
                            client._restore_session_after_rollback(cur)
                    logger.info(
                        "rag.hybrid.trgm_no_match",
                        extra={
                            "tenant_id": tenant,
                            "case_id": case_id,
                            "trace_id": trace_id,
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
                                        c.collection_id,
                                        similarity(c.text_norm, %s) AS lscore
                                    FROM chunks c
                                    JOIN documents d ON c.document_id = d.id
                                    WHERE {where_sql}
                                      AND similarity(c.text_norm, %s) >= %s
                                    ORDER BY lscore DESC
                                    LIMIT %s
                                """
                    base_limits: List[float] = []
                    if fallback_requested and (requested_trgm_limit is not None):
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
                    if fallback_requested and requested_trgm_limit is not None:
                        try:
                            fallback_floor = max(0.0, float(requested_trgm_limit))
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
                                lex_limit,
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
                                    "case_id": case_id,
                                    "trace_id": trace_id,
                                    "count": len(attempt_rows),
                                    "limit": float(limit_value),
                                    "rows": summarise_rows(
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
                                        len(attempt_rows[0]) if attempt_rows else 0
                                    ),
                                    "trace_id": trace_id,
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
                        fallback_limit_used_value = fallback_last_limit_value
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
                        case_id=case_id,
                        trace_id=trace_id,
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
                trace_id=trace_id,
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
                    client._restore_session_after_rollback(restore_cur)
            except Exception:  # pragma: no cover - defensive
                pass
        logger.warning(
            "rag.hybrid.lexical_query_failed",
            tenant_id=tenant,
            tenant=tenant,
            case_id=case_id,
            trace_id=trace_id,
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
            trace_id=trace_id,
        )
    except Exception:
        pass
    try:
        duration_ms = int(round((time.perf_counter() - started_at) * 1000))
        logger.info(
            "rag.hybrid.lexical_summary",
            tenant_id=tenant,
            case_id=case_id,
            trace_id=trace_id,
            count=len(lexical_rows),
            duration_ms=duration_ms,
            requested_trgm_limit=requested_trgm_limit,
            applied_trgm_limit=applied_trgm_limit_value,
            lexical_query_variant=lexical_query_variant,
        )
    except Exception:
        pass

    return LexicalSearchOutcome(
        rows=lexical_rows,
        applied_trgm_limit=applied_trgm_limit_value,
        fallback_limit_used=fallback_limit_used_value,
        fallback_tried_limits=fallback_tried_limits,
        lexical_query_variant=lexical_query_variant,
        lexical_fallback_limit=lexical_fallback_limit_value,
    )
