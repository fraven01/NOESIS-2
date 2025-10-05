"""Smoke-test helper for vector schema rendering."""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from ai_core.rag.embedding_config import get_embedding_configuration
from ai_core.rag.vector_schema import render_schema_sql


class Command(BaseCommand):
    help = "Render the vector schema SQL for a configured space to validate templates."

    def add_arguments(self, parser):
        parser.add_argument(
            "--space",
            required=True,
            help="Embedding vector space identifier to render",
        )
        parser.add_argument(
            "--show-sql",
            action="store_true",
            help="Print the rendered SQL instead of only reporting success",
        )

    def handle(self, *args, **options):
        space_id = str(options["space"]).strip()
        if not space_id:
            raise CommandError("--space must be provided")

        configuration = get_embedding_configuration()
        try:
            space = configuration.vector_spaces[space_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise CommandError(f"Vector space '{space_id}' is not configured") from exc

        sql = render_schema_sql(space.schema, space.dimension)
        if options.get("show_sql"):
            self.stdout.write(sql)
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "Rendered schema for space %s (schema=%s, dimension=%s)"
                    % (space.id, space.schema, space.dimension)
                )
            )

        return sql
