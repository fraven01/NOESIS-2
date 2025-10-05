"""DDL helpers for vector space schemas and dimension-specific tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from django.core.exceptions import ImproperlyConfigured

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
    "render_schema_sql",
    "validate_vector_schemas",
]
