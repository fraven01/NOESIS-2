"""DDL helpers for vector space schemas and dimension-specific tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from django.core.exceptions import ImproperlyConfigured
from django.db import connection
from psycopg2 import errorcodes, errors, sql

from .embedding_config import (
    EmbeddingConfiguration,
    VectorSpaceConfig,
    get_embedding_configuration,
)


class VectorSchemaError(ImproperlyConfigured):
    """Raised when vector schema DDL cannot be generated."""


class VectorSchemaErrorCode:
    """Machine-readable error codes for schema validation failures."""

    TEMPLATE_NOT_FOUND = "SCHEMA_TEMPLATE_MISSING"
    SCHEMA_DIMENSION_CONFLICT = "SCHEMA_DIM_CONFLICT"
    DIMENSION_RENDER_FAILED = "SCHEMA_DIM_RENDER_FAILED"


_DOC_HINT = "See README.md (Fehlercodes Abschnitt) for remediation guidance."

_SCHEMA_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "rag" / "schema.sql"
)

_SCHEMA_PLACEHOLDER = "{{SCHEMA_NAME}}"
_VECTOR_DIM_PLACEHOLDER = "{{VECTOR_DIM}}"


def _format_error(code: str, message: str) -> str:
    return f"{code}: {message}. {_DOC_HINT}"


@dataclass(frozen=True, slots=True)
class VectorSchemaDDL:
    """Rendered DDL statements for a configured vector space."""

    space_id: str
    schema: str
    dimension: int
    sql: str


def _load_schema_template() -> str:
    try:
        return _SCHEMA_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise VectorSchemaError(
            _format_error(
                VectorSchemaErrorCode.TEMPLATE_NOT_FOUND,
                f"Vector schema template missing at {_SCHEMA_TEMPLATE_PATH}",
            )
        ) from exc


def _render_schema_sql(schema: str, dimension: int, template: str) -> str:
    if not schema:
        raise VectorSchemaError(
            _format_error(
                VectorSchemaErrorCode.DIMENSION_RENDER_FAILED,
                "Vector space schema must be provided",
            )
        )
    if dimension <= 0:
        raise VectorSchemaError(
            _format_error(
                VectorSchemaErrorCode.DIMENSION_RENDER_FAILED,
                "Vector space dimension must be positive",
            )
        )

    if _SCHEMA_PLACEHOLDER not in template:
        raise VectorSchemaError(
            _format_error(
                VectorSchemaErrorCode.DIMENSION_RENDER_FAILED,
                "Schema template does not contain the {{SCHEMA_NAME}} placeholder",
            )
        )

    if _VECTOR_DIM_PLACEHOLDER not in template:
        raise VectorSchemaError(
            _format_error(
                VectorSchemaErrorCode.DIMENSION_RENDER_FAILED,
                "Schema template does not contain the {{VECTOR_DIM}} placeholder",
            )
        )

    schema_sql = template.replace(_SCHEMA_PLACEHOLDER, schema)
    rendered_sql = schema_sql.replace(_VECTOR_DIM_PLACEHOLDER, str(dimension))
    if _VECTOR_DIM_PLACEHOLDER in rendered_sql:
        raise VectorSchemaError(
            _format_error(
                VectorSchemaErrorCode.DIMENSION_RENDER_FAILED,
                "Schema template replacement for {{VECTOR_DIM}} did not complete",
            )
        )
    return rendered_sql


def _ensure_schema_dimension_isolation(spaces: Iterable[VectorSpaceConfig]) -> None:
    by_schema: Dict[str, int] = {}
    for space in spaces:
        schema = space.schema
        dimension = space.dimension
        if not schema:
            continue
        existing = by_schema.get(schema)
        if existing is None:
            by_schema[schema] = dimension
            continue
        if existing != dimension:
            raise VectorSchemaError(
                _format_error(
                    VectorSchemaErrorCode.SCHEMA_DIMENSION_CONFLICT,
                    (
                        "Vector spaces with differing dimensions map to the same "
                        f"schema '{schema}' ({existing} vs {dimension})"
                    ),
                )
            )


def build_vector_schema_plan(
    config: EmbeddingConfiguration | None = None,
) -> List[VectorSchemaDDL]:
    """Return rendered DDL statements for all configured vector spaces."""

    configuration = config or get_embedding_configuration()
    spaces = list(configuration.vector_spaces.values())
    _ensure_schema_dimension_isolation(spaces)
    template = _load_schema_template()

    plan: List[VectorSchemaDDL] = []
    for space in spaces:
        sql = _render_schema_sql(space.schema, space.dimension, template)
        plan.append(
            VectorSchemaDDL(
                space_id=space.id,
                schema=space.schema,
                dimension=space.dimension,
                sql=sql,
            )
        )
    return plan


def _iter_exception_chain(exc: Exception) -> Iterator[Exception]:
    current: Exception | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        cause = getattr(current, "__cause__", None)
        if cause is None or not isinstance(cause, Exception):
            break
        current = cause


def _is_missing_relation_error(exc: Exception) -> bool:
    missing_codes = {
        getattr(errors.UndefinedTable, "sqlstate", None),
        getattr(errors.InvalidSchemaName, "sqlstate", None),
        getattr(errorcodes, "UNDEFINED_TABLE", None),
        getattr(errorcodes, "INVALID_SCHEMA_NAME", None),
    }
    missing_codes = {code for code in missing_codes if code}
    for candidate in _iter_exception_chain(exc):
        pg_code = getattr(candidate, "pgcode", None)
        if pg_code and pg_code in missing_codes:
            return True
        message = str(candidate).lower()
        if "does not exist" in message and (
            "relation" in message or "table" in message or "schema" in message
        ):
            return True
    return False


def ensure_vector_space_schema(space: VectorSpaceConfig) -> bool:
    """Ensure the pgvector schema for ``space`` exists.

    Returns ``True`` if the schema DDL was executed and ``False`` if it was already
    present or the backend is not pgvector. Errors unrelated to missing relations
    are propagated so callers can handle misconfigurations explicitly.
    """

    if space.backend.lower() != "pgvector":
        return False

    with connection.cursor() as cursor:
        identifier = sql.Identifier(space.schema, "embeddings")
        try:
            cursor.execute(sql.SQL("SELECT 1 FROM {} LIMIT 0").format(identifier))
        except Exception as exc:
            if not _is_missing_relation_error(exc):
                raise
            ddl = render_schema_sql(space.schema, space.dimension)
            cursor.execute(ddl)
            return True
    return False


def validate_vector_schemas() -> None:
    """Validate vector-space schema assignments without rendering SQL."""

    configuration = get_embedding_configuration()
    _ensure_schema_dimension_isolation(configuration.vector_spaces.values())


def render_schema_sql(schema: str, dimension: int) -> str:
    """Render the schema template for a single vector space."""

    template = _load_schema_template()
    return _render_schema_sql(schema, dimension, template)


__all__ = [
    "VectorSchemaDDL",
    "VectorSchemaError",
    "VectorSchemaErrorCode",
    "build_vector_schema_plan",
    "ensure_vector_space_schema",
    "render_schema_sql",
    "validate_vector_schemas",
]
