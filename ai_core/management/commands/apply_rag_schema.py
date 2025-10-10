from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from ai_core.rag.vector_schema import render_schema_sql


class Command(BaseCommand):
    help = "Apply the vector schema template for a single configured space."

    def add_arguments(self, parser):
        parser.add_argument(
            "--schema",
            required=True,
            help="Target database schema (e.g. rag, rag_demo)",
        )
        parser.add_argument(
            "--dimension",
            required=True,
            type=int,
            help="Vector dimension that should be enforced (positive integer)",
        )
        parser.add_argument(
            "--path",
            default=None,
            help=(
                "Optional override for the schema template. Defaults to "
                "docs/rag/schema.sql"
            ),
        )

    def handle(self, *args, **options):
        schema_name = str(options["schema"]).strip()
        dimension = int(options["dimension"])
        if not schema_name:
            raise CommandError("--schema must be provided")
        if dimension <= 0:
            raise CommandError("--dimension must be a positive integer")

        template_override = options.get("path")
        if template_override:
            template_path = Path(template_override).resolve()
            if not template_path.exists():
                raise CommandError(f"Schema template not found: {template_path}")
            template = template_path.read_text(encoding="utf-8")
            if "{{SCHEMA_NAME}}" not in template or "{{VECTOR_DIM}}" not in template:
                raise CommandError(
                    "Schema template must contain {{SCHEMA_NAME}} and {{VECTOR_DIM}} placeholders"
                )
            rendered_sql = template.replace("{{SCHEMA_NAME}}", schema_name).replace(
                "{{VECTOR_DIM}}", str(dimension)
            )
        else:
            try:
                rendered_sql = render_schema_sql(schema_name, dimension)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise CommandError(str(exc)) from exc

        try:
            with connection.cursor() as cursor:
                cursor.execute(rendered_sql)
        except Exception as exc:  # noqa: BLE001 - surface SQL errors verbatim
            raise CommandError(
                f"Failed to apply RAG schema for '{schema_name}': {exc}"
            ) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Applied RAG schema for %s (dimension=%s)" % (schema_name, dimension)
            )
        )
