import re
from collections.abc import Mapping

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from psycopg2 import sql


class Command(BaseCommand):
    help = "Rebuild the embeddings vector index according to RAG_INDEX_KIND."

    def handle(self, *args, **_options):
        index_kind = str(getattr(settings, "RAG_INDEX_KIND", "HNSW")).upper()
        if index_kind not in {"HNSW", "IVFFLAT"}:
            raise CommandError(f"Unsupported RAG_INDEX_KIND '{index_kind}'")

        hnsw_m = int(getattr(settings, "RAG_HNSW_M", 32))
        hnsw_ef = int(getattr(settings, "RAG_HNSW_EF_CONSTRUCTION", 200))
        ivf_lists = int(getattr(settings, "RAG_IVF_LISTS", 2048))

        schema = self._resolve_vector_schema()
        embeddings_table = sql.Identifier(schema, "embeddings")
        hnsw_index = sql.Identifier(schema, "embeddings_embedding_hnsw")
        ivfflat_index = sql.Identifier(schema, "embeddings_embedding_ivfflat")

        with connection.cursor() as cur:
            cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(hnsw_index))
            cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(ivfflat_index))
            if index_kind == "HNSW":
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX {index}
                        ON {table} USING hnsw (embedding vector_cosine_ops)
                        WITH (m = %s, ef_construction = %s)
                        """
                    ).format(index=hnsw_index, table=embeddings_table),
                    (hnsw_m, hnsw_ef),
                )
            else:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX {index}
                        ON {table} USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = %s)
                        """
                    ).format(index=ivfflat_index, table=embeddings_table),
                    (ivf_lists,),
                )
            cur.execute(sql.SQL("ANALYZE {}").format(embeddings_table))
            expected_index = (
                "embeddings_embedding_hnsw"
                if index_kind == "HNSW"
                else "embeddings_embedding_ivfflat"
            )
            cur.execute(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = %s
                  AND tablename = %s
                  AND indexname = %s
                """,
                (schema, "embeddings", expected_index),
            )
            row = cur.fetchone()
            if not row:
                raise CommandError(
                    f"Reindex failed: expected index '{expected_index}' was not created"
                )
            indexdef = row[0] or ""
            if index_kind == "HNSW":
                m_match = re.search(r"m\s*=\s*(\d+)", indexdef)
                ef_match = re.search(r"ef_construction\s*=\s*(\d+)", indexdef)
                m_value = m_match.group(1) if m_match else hnsw_m
                ef_value = ef_match.group(1) if ef_match else hnsw_ef
                details = f"m={m_value} ef_construction={ef_value}"
            else:
                lists_match = re.search(r"lists\s*=\s*(\d+)", indexdef)
                lists_value = lists_match.group(1) if lists_match else ivf_lists
                details = f"lists={lists_value}"

        self.stdout.write(
            self.style.SUCCESS(
                f"Rebuilt embeddings index using {index_kind} (scope: {schema}.embeddings, {details})"
            )
        )

    def _resolve_vector_schema(self) -> str:
        stores = getattr(settings, "RAG_VECTOR_STORES", {})
        default_scope = getattr(settings, "RAG_VECTOR_DEFAULT_SCOPE", None)

        if isinstance(stores, Mapping) and stores:
            scope = (
                default_scope
                if isinstance(default_scope, str) and default_scope in stores
                else ("global" if "global" in stores else next(iter(stores), ""))
            )
            config = stores.get(scope, {}) if scope else {}
            if isinstance(config, Mapping):
                schema = config.get("schema")
                if isinstance(schema, str) and schema:
                    return schema

        return "rag"
