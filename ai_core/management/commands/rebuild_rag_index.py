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

        client = vector_client.get_default_client()
        schema_name = getattr(client, "_schema", "rag")
        scope = f"{schema_name}.embeddings"
        hnsw_index_name = "embeddings_embedding_hnsw"
        ivfflat_index_name = "embeddings_embedding_ivfflat"
        expected_index = hnsw_index_name if index_kind == "HNSW" else ivfflat_index_name
        row = None
        conn = connection
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(schema_name)
                    )
                )
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except errors.UndefinedFile as exc:
            conn.rollback()
            raise CommandError(
                "pgvector extension is not available; ensure it is installed in the database"
            ) from exc
        except Exception:
            conn.rollback()
            raise
        else:
            if not conn.get_autocommit() and not conn.in_atomic_block:
                conn.commit()

        try:
            with transaction.atomic():
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SET LOCAL search_path TO {}, public").format(
                            sql.Identifier(schema_name)
                        )
                    )
                    for index_name in (hnsw_index_name, ivfflat_index_name):
                        cur.execute(
                            sql.SQL("DROP INDEX IF EXISTS {}").format(
                                sql.Identifier(index_name)
                            )
                        )
                    if index_kind == "HNSW":
                        cur.execute(
                            sql.SQL(
                                """
                                CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops)
                                WITH (m = %s, ef_construction = %s)
                                """
                            ).format(
                                sql.Identifier(hnsw_index_name),
                                sql.Identifier(schema_name, "embeddings"),
                            ),
                            (hnsw_m, hnsw_ef),
                        )
                    else:
                        cur.execute(
                            sql.SQL(
                                """
                                CREATE INDEX {} ON {} USING ivfflat (embedding vector_cosine_ops)
                                WITH (lists = %s)
                                """
                            ).format(
                                sql.Identifier(ivfflat_index_name),
                                sql.Identifier(schema_name, "embeddings"),
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
                    row = cur.fetchone()
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
