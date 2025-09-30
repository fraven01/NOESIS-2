from __future__ import annotations

import io
import re

import pytest
from django.core.management import call_command
from django.db import connection


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
@pytest.mark.parametrize(
    "index_kind,settings_overrides,expected_index,expected_params",
    [
        (
            "HNSW",
            {"RAG_HNSW_M": 18, "RAG_HNSW_EF_CONSTRUCTION": 88},
            "embeddings_embedding_hnsw",
            {"m": 18, "ef_construction": 88},
        ),
        (
            "IVFFLAT",
            {"RAG_IVF_LISTS": 512},
            "embeddings_embedding_ivfflat",
            {"lists": 512},
        ),
    ],
)
def test_rebuild_rag_index_creates_expected_index(
    settings,
    index_kind: str,
    settings_overrides: dict[str, int],
    expected_index: str,
    expected_params: dict[str, int],
) -> None:
    settings.RAG_INDEX_KIND = index_kind
    for key, value in settings_overrides.items():
        setattr(settings, key, value)

    stdout = io.StringIO()
    call_command("rebuild_rag_index", stdout=stdout)
    message = stdout.getvalue()
    assert f"{index_kind}" in message

    with connection.cursor() as cur:
        cur.execute("SET search_path TO rag, public")
        cur.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'embeddings'
              AND indexname = %s
            """,
            (expected_index,),
        )
        row = cur.fetchone()

    assert row is not None, f"expected index {expected_index} to be present"
    indexdef = row[0]

    for param, value in expected_params.items():
        pattern = rf"{param}\s*=\s*'?{value}'?(?:::\w+)?"
        assert re.search(pattern, indexdef), f"missing {param}={value} in {indexdef!r}"
