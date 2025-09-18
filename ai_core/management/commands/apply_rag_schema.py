import os
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction


class Command(BaseCommand):
    help = "Apply docs/rag/schema.sql to the connected database (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            default=str(
                Path(settings.BASE_DIR) / "docs" / "rag" / "schema.sql"
            ),
            help="Path to schema.sql (default: docs/rag/schema.sql)",
        )

    def handle(self, *args, **options):
        sql_path = Path(options["path"]).resolve()
        if not sql_path.exists():
            raise CommandError(f"Schema file not found: {sql_path}")

        sql = sql_path.read_text(encoding="utf-8")
        if not sql.strip():
            self.stdout.write(self.style.WARNING("Schema file is empty; nothing to do."))
            return

        # Execute as a single script; schema.sql is idempotent
        try:
            with transaction.atomic():
                with connection.cursor() as cur:
                    cur.execute(sql)
        except Exception as exc:  # noqa: BLE001 - we want to surface any SQL error
            raise CommandError(f"Failed to apply RAG schema: {exc}") from exc

        self.stdout.write(self.style.SUCCESS(f"Applied RAG schema from {sql_path}"))

