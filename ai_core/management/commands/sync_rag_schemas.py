"""Apply the vector store schema for all configured spaces."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from ai_core.rag.embedding_config import (
    EmbeddingConfigurationError,
    get_embedding_configuration,
)
from ai_core.rag.vector_schema import (
    VectorSchemaError,
    build_vector_schema_plan,
)


class Command(BaseCommand):
    help = "Render and apply docs/rag/schema.sql for every pgvector space."

    def handle(self, *args, **options):
        try:
            configuration = get_embedding_configuration()
        except EmbeddingConfigurationError as exc:  # pragma: no cover - config guard
            raise CommandError(str(exc)) from exc

        try:
            plan = build_vector_schema_plan(configuration)
        except VectorSchemaError as exc:
            raise CommandError(str(exc)) from exc

        if not plan:
            self.stdout.write(
                self.style.WARNING("No vector spaces configured; nothing to do.")
            )
            return

        applied = 0
        with connection.cursor() as cursor:
            for ddl in plan:
                space_config = configuration.vector_spaces.get(ddl.space_id)
                if (
                    space_config is not None
                    and space_config.backend.lower() != "pgvector"
                ):
                    self.stdout.write(
                        self.style.WARNING(
                            "Skipping vector space '%s' (backend=%s)"
                            % (ddl.space_id, space_config.backend)
                        )
                    )
                    continue

                try:
                    cursor.execute(ddl.sql)
                except Exception as exc:  # noqa: BLE001 - surface SQL errors verbatim
                    raise CommandError(
                        "Failed to apply schema for space '%s' (%s): %s"
                        % (ddl.space_id, ddl.schema, exc)
                    ) from exc

                applied += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        "Applied schema for space '%s' (schema=%s, dimension=%s)"
                        % (ddl.space_id, ddl.schema, ddl.dimension)
                    )
                )

        if applied == 0:
            self.stdout.write(
                self.style.WARNING(
                    "Configured vector spaces are not backed by pgvector; nothing applied."
                )
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "Applied %s vector schema(s) successfully." % applied
                )
            )
