#!/usr/bin/env python
"""Debug script to check vector store for document chunks."""
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings")

import django

django.setup()

from uuid import UUID
from django.conf import settings
from django.db import connection

DOC_ID = "fd168f2b-3289-47a8-8b76-0e8271374a66"


def main():
    doc_uuid = UUID(DOC_ID)
    print(f"=== Looking for chunks for document: {DOC_ID} ===\n")

    # Get vector store configuration
    vector_config = getattr(settings, "RAG_VECTOR_STORES", {})
    print(f"RAG_VECTOR_STORES config: {list(vector_config.keys())}")

    if "global" in vector_config:
        global_config = vector_config["global"]
        print(f"Global vector store: {global_config}")

    # Look for chunks table in database
    print("\n=== Checking for vector tables ===")
    with connection.cursor() as cursor:
        # List all tables with 'chunk' or 'vector' in name
        cursor.execute(
            """
            SELECT schemaname, tablename 
            FROM pg_tables 
            WHERE tablename LIKE '%chunk%' OR tablename LIKE '%vector%' OR tablename LIKE '%embedding%'
            ORDER BY schemaname, tablename
        """
        )
        tables = cursor.fetchall()
        print(f"Found {len(tables)} relevant tables:")
        for schema, table in tables:
            print(f"  - {schema}.{table}")

    # Also check for langchain tables
    print("\n=== Checking for langchain tables ===")
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT schemaname, tablename 
            FROM pg_tables 
            WHERE tablename LIKE 'langchain%'
            ORDER BY schemaname, tablename
        """
        )
        tables = cursor.fetchall()
        print(f"Found {len(tables)} langchain tables:")
        for schema, table in tables:
            print(f"  - {schema}.{table}")

            # Count rows and check document_id
            try:
                cursor.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
                count = cursor.fetchone()[0]
                print(f"    Total rows: {count}")

                # Look for our document
                cursor.execute(
                    f"""
                    SELECT cmetadata 
                    FROM "{schema}"."{table}" 
                    WHERE cmetadata->>'document_id' = %s
                    LIMIT 5
                """,
                    [str(doc_uuid)],
                )
                matching = cursor.fetchall()
                print(f"    Chunks with our document_id: {len(matching)}")
                for row in matching:
                    print(f"      Metadata: {row[0]}")
            except Exception as e:
                print(f"    Error querying: {e}")

    # Check rag schema
    print("\n=== Checking rag schema ===")
    with connection.cursor() as cursor:
        try:
            cursor.execute(
                """
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'rag'
            """
            )
            tables = cursor.fetchall()
            print(f"Found {len(tables)} tables in rag schema:")
            for (table,) in tables:
                print(f"  - rag.{table}")

                # Try to find our doc
                try:
                    cursor.execute(
                        f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_schema = 'rag' AND table_name = %s
                    """,
                        [table],
                    )
                    columns = [c[0] for c in cursor.fetchall()]

                    if "document_id" in columns or "cmetadata" in columns:
                        print(f"    Checking for document...")
                        if "document_id" in columns:
                            cursor.execute(
                                f'SELECT COUNT(*) FROM "rag"."{table}" WHERE document_id = %s',
                                [str(doc_uuid)],
                            )
                        else:
                            cursor.execute(
                                f"""
                                SELECT COUNT(*) FROM "rag"."{table}" 
                                WHERE cmetadata->>'document_id' = %s
                            """,
                                [str(doc_uuid)],
                            )
                        count = cursor.fetchone()[0]
                        print(f"    Matching chunks: {count}")
                except Exception as e:
                    print(f"    Error: {e}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
