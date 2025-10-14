"""Tests for the rag_schema_smoke management command."""

from __future__ import annotations

import io

import pytest
from django.core.management import call_command


@pytest.fixture(autouse=True)
def _reset_settings(settings) -> None:
    settings.RAG_VECTOR_STORES = {
        "global": {
            "backend": "pgvector",
            "schema": "rag",
            "dimension": 1536,
        }
    }
    settings.RAG_EMBEDDING_PROFILES = {
        "standard": {
            "model": "oai-embed-large",
            "dimension": 1536,
            "vector_space": "global",
            "chunk_hard_limit": 512,
        }
    }


def test_rag_schema_smoke_reports_success(settings) -> None:
    buffer = io.StringIO()
    call_command("rag_schema_smoke", space="global", stdout=buffer)

    output = buffer.getvalue()
    assert "Rendered schema for space global" in output
    assert "schema=rag" in output
    assert "dimension=1536" in output


def test_rag_schema_smoke_can_print_sql(settings) -> None:
    buffer = io.StringIO()
    call_command("rag_schema_smoke", space="global", show_sql=True, stdout=buffer)

    output = buffer.getvalue()
    assert "CREATE SCHEMA" in output
    assert "vector(1536)" in output
