"""SQL template builders for hybrid search queries."""

from __future__ import annotations


def build_vector_sql(
    where_sql_value: str, select_columns: str, order_by_clause: str
) -> str:
    # where_sql_value contains union (OR) scope predicates between case and
    # collection filters to intentionally broaden context when both are provided.
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


def build_lexical_primary_sql(
    where_sql_value: str, select_columns: str, match_clause: str
) -> str:
    return f"""
        SELECT
            {select_columns}
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE {where_sql_value}
          AND {match_clause}
        ORDER BY lscore DESC
        LIMIT %s
    """


def build_lexical_fallback_sql(where_sql_value: str, select_columns: str) -> str:
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
