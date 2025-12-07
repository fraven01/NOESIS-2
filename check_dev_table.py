import os
import django
from django.db import connection

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
django.setup()


def check_schema():
    with connection.cursor() as cursor:
        # Check if schema exists
        cursor.execute(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'dev';"
        )
        if not cursor.fetchone():
            print("Schema 'dev' DOES NOT EXIST.")
            return

        print("Schema 'dev' exists.")

        # Check for table
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'dev' AND table_name = 'documents_documentcollection';"
        )
        if cursor.fetchone():
            print("Table 'documents_documentcollection' FOUND in 'dev' schema.")
        else:
            print("Table 'documents_documentcollection' MISSING in 'dev' schema.")


if __name__ == "__main__":
    check_schema()
