import re

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction
from psycopg2 import errors, sql

from ai_core.rag import vector_client


class Command(BaseCommand):
    help = "Rebuild the embeddings vector index according to RAG_INDEX_KIND."

    def handle(self, *args, **_options):
        index_kind = str(getattr(settings, "RAG_INDEX_KIND", "HNSW")).upper()
        if index_kind not in {"HNSW", "IVFFLAT"}:
            raise CommandError(f"Unsupported RAG_INDEX_KIND '{index_kind}'")

        hnsw_m = int(getattr(settings, "RAG_HNSW_M", 32))
        hnsw_ef = int(getattr(settings, "RAG_HNSW_EF_CONSTRUCTION", 200))
        ivf_lists = int(getattr(settings, "RAG_IVF_LISTS", 2048))

        schema_name = vector_client.get_default_schema()
        scope = f"{schema_name}.embeddings"
        hnsw_index_name = "embeddings_embedding_hnsw"
        ivfflat_index_name = "embeddings_embedding_ivfflat"
        expected_index = hnsw_index_name if index_kind == "HNSW" else ivfflat_index_name
        try:
            if connection.in_atomic_block:
                row = self._rebuild_index(
                    schema_name,
                    index_kind,
                    hnsw_m,
                    hnsw_ef,
                    ivf_lists,
                    expected_index,
                    hnsw_index_name,
                    ivfflat_index_name,
                )
            else:
                with transaction.atomic():
                    row = self._rebuild_index(
                        schema_name,
                        index_kind,
                        hnsw_m,
                        hnsw_ef,
                        ivf_lists,
                        expected_index,
                        hnsw_index_name,
                        ivfflat_index_name,
                    )
        except errors.UndefinedFile as exc:
            raise CommandError(
                f"Required database extension is not available: {exc}"
            ) from exc
        except errors.UndefinedTable as exc:
            raise CommandError(
                "Vector table 'embeddings' not found; ensure the RAG schema is initialised"
            ) from exc

        if not row:
            raise CommandError(
                f"Reindex failed: expected index '{expected_index}' was not created"
            )
        indexdef = row[0] or ""
        if index_kind == "HNSW":
            m_match = re.search(r"m\s*=\s*(\d+)", indexdef)
            ef_match = re.search(r"ef_construction\s*=\s*(\d+)", indexdef)
            details = f"m={m_match.group(1) if m_match else hnsw_m} ef_construction={ef_match.group(1) if ef_match else hnsw_ef}"
        else:
            lists_match = re.search(r"lists\s*=\s*(\d+)", indexdef)
            details = f"lists={lists_match.group(1) if lists_match else ivf_lists}"

        self.stdout.write(
            self.style.SUCCESS(
                f"Rebuilt embeddings index using {index_kind} (scope: {scope}, {details})"
            )
        )

    def _rebuild_index(
        self,
        schema_name: str,
        index_kind: str,
        hnsw_m: int,
        hnsw_ef: int,
        ivf_lists: int,
        expected_index: str,
        hnsw_index_name: str,
        ivfflat_index_name: str,
    ) -> tuple[str] | None:
        with connection.cursor() as cur:
            self._ensure_schema(cur, schema_name)
            self._ensure_tables(cur, schema_name)

            for index_name in (hnsw_index_name, ivfflat_index_name):
                cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {}.{}").format(
                        sql.Identifier(schema_name),
                        sql.Identifier(index_name),
                    )
                )

            operator_class = self._resolve_operator_class(cur, index_kind)
            table_identifier = sql.Identifier(schema_name, "embeddings")

            if index_kind == "HNSW":
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX {index_name} ON {table}
                        USING hnsw (embedding {operator_class})
                        WITH (m = %s, ef_construction = %s)
                        """
                    ).format(
                        index_name=sql.Identifier(hnsw_index_name),
                        table=table_identifier,
                        operator_class=sql.Identifier(operator_class),
                    ),
                    (hnsw_m, hnsw_ef),
                )
            else:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX {index_name} ON {table}
                        USING ivfflat (embedding {operator_class})
                        WITH (lists = %s)
                        """
                    ).format(
                        index_name=sql.Identifier(ivfflat_index_name),
                        table=table_identifier,
                        operator_class=sql.Identifier(operator_class),
                    ),
                    (ivf_lists,),
                )

            cur.execute(
                sql.SQL("ANALYZE {}").format(
                    sql.Identifier(schema_name, "embeddings")
                )
            )
            cur.execute(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = %s
                  AND tablename = 'embeddings'
                  AND indexname = %s
                """,
                (schema_name, expected_index),
            )
            return cur.fetchone()

    def _resolve_operator_class(self, cur, index_kind: str) -> str:
        access_method = "hnsw" if index_kind == "HNSW" else "ivfflat"
        preferred = "vector_cosine_ops"
        if self._operator_class_exists(cur, preferred, access_method):
            return preferred

        for fallback in ("vector_l2_ops", "vector_ip_ops"):
            if self._operator_class_exists(cur, fallback, access_method):
                return fallback

        raise CommandError(
            "No compatible operator class found for pgvector index creation. "
            "Ensure the pgvector extension is installed with cosine or L2 support."
        )

    def _operator_class_exists(
        self, cur, operator_class: str, access_method: str
    ) -> bool:
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

    def _ensure_schema(self, cur, schema_name: str) -> None:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(schema_name)
            )
        )
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    def _ensure_tables(self, cur, schema_name: str) -> None:
        documents = sql.Identifier(schema_name, "documents")
        chunks = sql.Identifier(schema_name, "chunks")
        embeddings = sql.Identifier(schema_name, "embeddings")

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {documents} (
                    id UUID PRIMARY KEY,
                    tenant_id UUID NOT NULL,
                    source TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    hash TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    deleted_at TIMESTAMP WITH TIME ZONE,
                    external_id TEXT
                )
                """
            ).format(documents=documents)
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {chunks} (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES {documents}(id) ON DELETE CASCADE,
                    ord INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    text_norm TEXT GENERATED ALWAYS AS (lower(regexp_replace(text, '\\s+', ' ', 'g'))) STORED
                )
                """
            ).format(chunks=chunks, documents=documents)
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {embeddings} (
                    id UUID PRIMARY KEY,
                    chunk_id UUID NOT NULL REFERENCES {chunks}(id) ON DELETE CASCADE,
                    embedding vector(1536) NOT NULL
                )
                """
            ).format(embeddings=embeddings, chunks=chunks)
        )

        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS embeddings_chunk_idx
                    ON {embeddings} (chunk_id)
                """
            ).format(embeddings=embeddings)
        )
