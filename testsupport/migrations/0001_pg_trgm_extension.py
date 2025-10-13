from django.db import migrations


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.RunSQL(
            sql=(
                # Ensure pg_trgm exists and lives in the public schema
                "CREATE EXTENSION IF NOT EXISTS pg_trgm;\n"
                "ALTER EXTENSION pg_trgm SET SCHEMA public;"
            ),
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
