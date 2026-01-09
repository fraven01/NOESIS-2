import os

DOC_ID = "02ba5e0c-75ad-480e-ae18-2adc54fbbb3e"
TENANT = "dev"


def _setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
    import django

    django.setup()


def inspect():
    _setup_django()
    from django.db import connection
    from django_tenants.utils import schema_context
    from documents.models import Document, DocumentAsset

    print(f"--- INSPECTION START: {DOC_ID} ---")
    try:
        with schema_context(TENANT):
            # 1. Check Document
            print("\n[1] Checking Document...")
            try:
                doc = Document.objects.get(id=DOC_ID)
                print("    FOUND: Yes")
                print(f"    ID: {doc.id}")
                print(f"    Created: {doc.created_at}")
                print(f"    Title: {doc.metadata.get('title', 'N/A')}")
            except Document.DoesNotExist:
                print("    FOUND: NO")
                return # Stop if doc not found

            # 2. Check DocumentAssets (Chunks)
            print("\n[2] Checking DocumentAssets (Chunks)...")
            chunks = DocumentAsset.objects.filter(
                document_id=DOC_ID, metadata__asset_kind="chunk"
            )
            chunk_count = chunks.count()
            print(f"    COUNT: {chunk_count}")

            if chunk_count > 0:
                first = chunks.first()
                print(f"    SAMPLE CHUNK ID: {first.asset_id}")
                print(f"    SAMPLE CONTENT: {str(first.content)[:50]}...")
            else:
                # Check if ANY assets exist
                all_assets = DocumentAsset.objects.filter(document_id=DOC_ID).count()
                print(f"    TOTAL ASSETS (Any Kind): {all_assets}")

            # 3. Check 'rag' Schema for Vector Data
            print("\n[3] Checking 'rag' schema for Vector Data...")
            with connection.cursor() as cursor:
                # Check if rag schema exists
                cursor.execute(
                    "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'rag'"
                )
                if not cursor.fetchone():
                    print("    Schema 'rag' NOT FOUND.")
                else:
                    print("    Schema 'rag' FOUND.")

                    # Check rag.chunks
                    cursor.execute(
                        "SELECT count(*) FROM rag.chunks WHERE document_id = %s",
                        [DOC_ID],
                    )
                    rag_chunk_count = cursor.fetchone()[0]
                    print(f"    rag.chunks COUNT: {rag_chunk_count}")

                    if rag_chunk_count > 0:
                        # Sample content
                        cursor.execute(
                            "SELECT id, text, metadata FROM rag.chunks WHERE document_id = %s LIMIT 1",
                            [DOC_ID],
                        )
                        row = cursor.fetchone()
                        print(f"    SAMPLE RAG CHUNK ID: {row[0]}")
                        print(f"    SAMPLE RAG CHUNK TEXT: {row[1][:100]}...")
                        print(f"    SAMPLE RAG CHUNK META: {row[2]}")

                        # Check rag.embeddings linked to these chunks
                        cursor.execute(
                            """
                            SELECT count(*)
                            FROM rag.embeddings e
                            JOIN rag.chunks c ON e.chunk_id = c.id
                            WHERE c.document_id = %s
                            """,
                            [DOC_ID],
                        )
                        rag_embedding_count = cursor.fetchone()[0]
                        print(f"    rag.embeddings COUNT: {rag_embedding_count}")
                    else:
                        print("    rag.embeddings COUNT: 0 (No chunks in rag.chunks)")

            # 4. Exemplary Assets (Non-Chunk)
            print("\n[4] Sampling Non-Chunk Assets (DocumentAsset)...")
            assets = DocumentAsset.objects.filter(document_id=DOC_ID).exclude(
                metadata__asset_kind="chunk"
            )[:5]
            for a in assets:
                kind = a.metadata.get("asset_kind", "N/A")
                media = a.media_type
                print(f"    - ID: {a.asset_id}, Kind: {kind}, Media: {media}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
    print("--- INSPECTION END ---")


if __name__ == "__main__":
    inspect()
