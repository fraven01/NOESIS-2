import psycopg2

dsn = "postgresql://noesis2:noesis2@localhost:5432/noesis2"

try:
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Check if schema 'dev' exists
            cur.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'dev';"
            )
            if not cur.fetchone():
                print("Schema 'dev' DOES NOT EXIST.")
            else:
                print("Schema 'dev' FOUND.")

                # Check tables in 'dev'
                cur.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'dev';"
                )
                tables = [row[0] for row in cur.fetchall()]
                print("Tables in 'dev':")
                for t in tables:
                    print(f" - {t}")

                if "documents_documentcollection" not in tables:
                    print(
                        "CRITICAL: documents_documentcollection MISSING in 'dev' schema."
                    )
                else:
                    print("OK: documents_documentcollection exists.")

except Exception as e:
    print(f"Connection failed: {e}")
