"""Management command to inspect chunks for a specific document."""

from __future__ import annotations

import json
import uuid
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from psycopg2 import sql

from ai_core.rag.vector_client import get_client_for_schema


class Command(BaseCommand):
    """Inspect chunks stored in the RAG vector store for a document."""

    help = "Inspect chunks for a document in the vector store"

    def add_arguments(self, parser):
        parser.add_argument(
            "document_id",
            type=str,
            help="Document UUID to inspect",
        )
        parser.add_argument(
            "--tenant-id",
            type=str,
            required=True,
            help="Tenant UUID",
        )
        parser.add_argument(
            "--schema",
            type=str,
            default="rag",
            help="RAG schema name (default: rag)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=50,
            help="Max chunks to return (default: 50)",
        )
        parser.add_argument(
            "--show-text",
            action="store_true",
            help="Show full chunk text",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            dest="output_json",
            help="Output as JSON",
        )

    def handle(self, *args, **options):
        document_id = options["document_id"]
        tenant_id = options["tenant_id"]
        schema = options["schema"]
        limit = options["limit"]
        show_text = options["show_text"]
        output_json = options["output_json"]

        try:
            doc_uuid = uuid.UUID(document_id)
        except ValueError as e:
            raise CommandError(f"Invalid document_id: {e}")

        try:
            tenant_uuid = uuid.UUID(tenant_id)
        except ValueError as e:
            raise CommandError(f"Invalid tenant_id: {e}")

        client = get_client_for_schema(schema)
        chunks_table = client._table("chunks")

        query = sql.SQL(
            """
            SELECT
                c.id,
                c.ord,
                c.text,
                c.metadata,
                length(c.text) as text_length
            FROM {} c
            WHERE c.document_id = %s
              AND c.tenant_id = %s
            ORDER BY c.ord
            LIMIT %s
            """
        ).format(chunks_table)

        rows: list[tuple[Any, ...]] = []
        total_count = 0

        with client._connection() as conn:
            with conn.cursor() as cur:
                # Get total count
                count_query = sql.SQL(
                    "SELECT COUNT(*) FROM {} WHERE document_id = %s AND tenant_id = %s"
                ).format(chunks_table)
                cur.execute(count_query, [doc_uuid, tenant_uuid])
                total_count = cur.fetchone()[0]

                # Get chunks
                cur.execute(query, [doc_uuid, tenant_uuid, limit])
                rows = list(cur.fetchall())

        if output_json:
            result = {
                "document_id": str(doc_uuid),
                "tenant_id": str(tenant_uuid),
                "total_chunks": total_count,
                "shown_chunks": len(rows),
                "chunks": [],
            }
            for chunk_id, ord_val, text, metadata, text_len in rows:
                chunk_data = {
                    "chunk_id": str(chunk_id),
                    "ord": ord_val,
                    "text_length": text_len,
                    "metadata": metadata,
                }
                if show_text:
                    chunk_data["text"] = text
                else:
                    chunk_data["text_preview"] = (
                        text[:200] + "..." if len(text) > 200 else text
                    )
                result["chunks"].append(chunk_data)

            self.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            self.stdout.write(
                self.style.SUCCESS(f"\n=== Chunk Analysis for Document {doc_uuid} ===")
            )
            self.stdout.write(f"Tenant: {tenant_uuid}")
            self.stdout.write(f"Schema: {schema}")
            self.stdout.write(f"Total chunks: {total_count}")
            self.stdout.write(f"Showing: {len(rows)}\n")

            for chunk_id, ord_val, text, metadata, text_len in rows:
                self.stdout.write(
                    self.style.WARNING(f"\n--- Chunk {ord_val} (ID: {chunk_id}) ---")
                )
                self.stdout.write(f"Length: {text_len} chars")

                # Show relevant metadata
                if metadata:
                    meta_keys = [
                        "document_version_id",
                        "source",
                        "case_id",
                        "is_latest",
                        "section",
                        "heading",
                    ]
                    shown_meta = {k: v for k, v in metadata.items() if k in meta_keys}
                    if shown_meta:
                        self.stdout.write(
                            f"Metadata: {json.dumps(shown_meta, ensure_ascii=False)}"
                        )

                    # Show parent info if present
                    if "parent" in metadata:
                        self.stdout.write(f"Parent: {metadata['parent']}")

                if show_text:
                    self.stdout.write(f"\nText:\n{text}")
                else:
                    preview = text[:300] + "..." if len(text) > 300 else text
                    self.stdout.write(f"\nPreview:\n{preview}")

            # Summary statistics
            if rows:
                lengths = [row[4] for row in rows]
                self.stdout.write(self.style.SUCCESS("\n=== Statistics ==="))
                self.stdout.write(f"Min length: {min(lengths)} chars")
                self.stdout.write(f"Max length: {max(lengths)} chars")
                self.stdout.write(
                    f"Avg length: {sum(lengths) / len(lengths):.0f} chars"
                )
