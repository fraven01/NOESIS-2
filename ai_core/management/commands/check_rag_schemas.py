"""Health-check command for configured vector schemas."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from psycopg2 import sql

from ai_core.rag.embedding_config import (
    EmbeddingConfigurationError,
    get_embedding_configuration,
)


class Command(BaseCommand):
    help = "Verify that all pgvector-backed vector spaces expose the required tables."

    def handle(self, *args, **options):
        try:
            configuration = get_embedding_configuration()
        except EmbeddingConfigurationError as exc:  # pragma: no cover - config guard
            raise CommandError(str(exc)) from exc

        spaces = list(configuration.vector_spaces.values())
        if not spaces:
            self.stdout.write(
                self.style.WARNING("No vector spaces configured; nothing to check.")
            )
            return

        checked = 0
        with connection.cursor() as cursor:
            for space in spaces:
                if space.backend.lower() != "pgvector":
                    self.stdout.write(
                        self.style.WARNING(
                            "Skipping vector space '%s' (backend=%s)"
                            % (space.id, space.backend)
                        )
                    )
                    continue

                for table in ("documents", "chunks", "embeddings", "embedding_cache"):
                    identifier = sql.Identifier(space.schema, table)
                    try:
                        cursor.execute(
                            sql.SQL("SELECT 1 FROM {} LIMIT 0").format(identifier)
                        )
                    except Exception as exc:  # noqa: BLE001 - surface SQL errors
                        raise CommandError(
                            "Vector schema check failed for %s.%s: %s"
                            % (space.schema, table, exc)
                        ) from exc

                checked += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        "Vector schema healthy for space '%s' (schema=%s, dimension=%s)"
                        % (space.id, space.schema, space.dimension)
                    )
                )

        if checked == 0:
            self.stdout.write(
                self.style.WARNING(
                    "Configured vector spaces are not backed by pgvector; nothing checked."
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS("Verified %s vector schema(s)." % checked)
            )
