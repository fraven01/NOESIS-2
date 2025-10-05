"""Configuration loader for embedding profiles and vector spaces."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, cast

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


class EmbeddingConfigurationError(ImproperlyConfigured):
    """Raised when the embedding configuration is invalid."""


class EmbeddingConfigErrorCode:
    """Machine-readable error codes for embedding configuration issues."""

    CONFIG_NOT_MAPPING = "EMB_CONFIG_TYPE"
    VECTOR_SPACE_NOT_MAPPING = "EMB_SPACE_TYPE"
    VECTOR_SPACE_BACKEND_REQUIRED = "EMB_SPACE_BACKEND_REQUIRED"
    VECTOR_SPACE_SCHEMA_REQUIRED = "EMB_SPACE_SCHEMA_REQUIRED"
    VECTOR_SPACE_DIMENSION_INVALID = "EMB_SPACE_DIM_INVALID"
    VECTOR_SPACES_EMPTY = "EMB_NO_SPACES"
    PROFILE_NOT_MAPPING = "EMB_PROFILE_TYPE"
    PROFILE_MODEL_REQUIRED = "EMB_PROFILE_MODEL_REQUIRED"
    PROFILE_DIMENSION_INVALID = "EMB_PROFILE_DIM_INVALID"
    PROFILE_SPACE_REQUIRED = "EMB_PROFILE_SPACE_REQUIRED"
    UNKNOWN_VECTOR_SPACE = "EMB_UNKNOWN_SPACE"
    DIMENSION_MISMATCH = "EMB_DIM_MISMATCH"
    PROFILES_EMPTY = "EMB_NO_PROFILES"
    UNKNOWN_PROFILE = "EMB_UNKNOWN_PROFILE"


_EMBEDDING_DOC_HINT = "See README.md (Fehlercodes Abschnitt) for remediation guidance."


def _format_error(code: str, message: str) -> str:
    return f"{code}: {message}. {_EMBEDDING_DOC_HINT}"


@dataclass(frozen=True, slots=True)
class VectorSpaceConfig:
    """Describe a physical vector space for embeddings."""

    id: str
    backend: str
    schema: str
    dimension: int


@dataclass(frozen=True, slots=True)
class EmbeddingProfileConfig:
    """Logical embedding profile bound to a vector space."""

    id: str
    model: str
    dimension: int
    vector_space: str


@dataclass(frozen=True, slots=True)
class EmbeddingConfiguration:
    """Container holding validated vector spaces and embedding profiles."""

    vector_spaces: Dict[str, VectorSpaceConfig]
    embedding_profiles: Dict[str, EmbeddingProfileConfig]


def _ensure_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmbeddingConfigurationError(
            _format_error(
                EmbeddingConfigErrorCode.CONFIG_NOT_MAPPING,
                f"{name} must be a mapping",
            )
        )
    return cast(Mapping[str, Any], value)


def _coerce_dimension(raw: object, *, context: str, error_code: str) -> int:
    try:
        dimension = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EmbeddingConfigurationError(
            _format_error(
                error_code,
                f"{context} dimension must be an integer",
            )
        ) from exc
    if dimension <= 0:
        raise EmbeddingConfigurationError(
            _format_error(
                error_code,
                f"{context} dimension must be positive",
            )
        )
    return dimension


def _parse_vector_spaces(
    raw_spaces: Mapping[str, object],
) -> Dict[str, VectorSpaceConfig]:
    parsed: Dict[str, VectorSpaceConfig] = {}
    for space_id, raw_config in raw_spaces.items():
        if not isinstance(raw_config, Mapping):
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.VECTOR_SPACE_NOT_MAPPING,
                    f"Vector space '{space_id}' must be a mapping",
                )
            )
        backend = str(raw_config.get("backend", "")).strip()
        if not backend:
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.VECTOR_SPACE_BACKEND_REQUIRED,
                    f"Vector space '{space_id}' must define a backend",
                )
            )
        schema = str(raw_config.get("schema", "")).strip()
        if not schema:
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.VECTOR_SPACE_SCHEMA_REQUIRED,
                    f"Vector space '{space_id}' must define a schema or namespace",
                )
            )
        dimension = _coerce_dimension(
            raw_config.get("dimension"),
            context=f"Vector space '{space_id}'",
            error_code=EmbeddingConfigErrorCode.VECTOR_SPACE_DIMENSION_INVALID,
        )
        parsed[space_id] = VectorSpaceConfig(
            id=str(space_id),
            backend=backend,
            schema=schema,
            dimension=dimension,
        )
    if not parsed:
        raise EmbeddingConfigurationError(
            _format_error(
                EmbeddingConfigErrorCode.VECTOR_SPACES_EMPTY,
                "No vector spaces configured",
            )
        )
    return parsed


def _parse_embedding_profiles(
    raw_profiles: Mapping[str, object],
    *,
    vector_spaces: Mapping[str, VectorSpaceConfig],
) -> Dict[str, EmbeddingProfileConfig]:
    parsed: Dict[str, EmbeddingProfileConfig] = {}
    for profile_id, raw_config in raw_profiles.items():
        if not isinstance(raw_config, Mapping):
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.PROFILE_NOT_MAPPING,
                    f"Embedding profile '{profile_id}' must be a mapping",
                )
            )
        model = str(raw_config.get("model", "")).strip()
        if not model:
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.PROFILE_MODEL_REQUIRED,
                    f"Embedding profile '{profile_id}' must define a model alias",
                )
            )
        dimension = _coerce_dimension(
            raw_config.get("dimension"),
            context=f"Embedding profile '{profile_id}'",
            error_code=EmbeddingConfigErrorCode.PROFILE_DIMENSION_INVALID,
        )
        vector_space_id = str(raw_config.get("vector_space", "")).strip()
        if not vector_space_id:
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.PROFILE_SPACE_REQUIRED,
                    f"Embedding profile '{profile_id}' must reference a vector space",
                )
            )
        if vector_space_id not in vector_spaces:
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.UNKNOWN_VECTOR_SPACE,
                    (
                        "Embedding profile "
                        f"'{profile_id}' references unknown vector space "
                        f"'{vector_space_id}'"
                    ),
                )
            )
        space = vector_spaces[vector_space_id]
        if space.dimension != dimension:
            raise EmbeddingConfigurationError(
                _format_error(
                    EmbeddingConfigErrorCode.DIMENSION_MISMATCH,
                    (
                        "Dimension mismatch between embedding profile "
                        f"'{profile_id}' ({dimension}) and vector space "
                        f"'{vector_space_id}' ({space.dimension})"
                    ),
                )
            )
        parsed[profile_id] = EmbeddingProfileConfig(
            id=str(profile_id),
            model=model,
            dimension=dimension,
            vector_space=vector_space_id,
        )
    if not parsed:
        raise EmbeddingConfigurationError(
            _format_error(
                EmbeddingConfigErrorCode.PROFILES_EMPTY,
                "No embedding profiles configured",
            )
        )
    return parsed


@lru_cache(maxsize=1)
def get_embedding_configuration() -> EmbeddingConfiguration:
    """Return the validated embedding configuration from Django settings."""

    raw_spaces = _ensure_mapping(
        getattr(settings, "RAG_VECTOR_STORES", {}), name="RAG_VECTOR_STORES"
    )
    raw_profiles = _ensure_mapping(
        getattr(settings, "RAG_EMBEDDING_PROFILES", {}),
        name="RAG_EMBEDDING_PROFILES",
    )

    vector_spaces = _parse_vector_spaces(raw_spaces)
    embedding_profiles = _parse_embedding_profiles(
        raw_profiles, vector_spaces=vector_spaces
    )
    return EmbeddingConfiguration(
        vector_spaces=dict(vector_spaces),
        embedding_profiles=dict(embedding_profiles),
    )


def validate_embedding_configuration() -> None:
    """Validate the embedding configuration and raise on inconsistencies."""

    get_embedding_configuration()


def reset_embedding_configuration_cache() -> None:
    """Clear cached configuration to honour runtime test overrides."""

    get_embedding_configuration.cache_clear()  # type: ignore[attr-defined]


def get_vector_space(space_id: str) -> VectorSpaceConfig:
    """Return a configured vector space by identifier."""

    config = get_embedding_configuration().vector_spaces
    try:
        return config[space_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise EmbeddingConfigurationError(
            _format_error(
                EmbeddingConfigErrorCode.UNKNOWN_VECTOR_SPACE,
                f"Unknown vector space '{space_id}'",
            )
        ) from exc


def get_embedding_profile(profile_id: str) -> EmbeddingProfileConfig:
    """Return a configured embedding profile by identifier."""

    config = get_embedding_configuration().embedding_profiles
    try:
        return config[profile_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise EmbeddingConfigurationError(
            _format_error(
                EmbeddingConfigErrorCode.UNKNOWN_PROFILE,
                f"Unknown embedding profile '{profile_id}'",
            )
        ) from exc


__all__ = [
    "EmbeddingConfiguration",
    "EmbeddingConfigurationError",
    "EmbeddingProfileConfig",
    "EmbeddingConfigErrorCode",
    "VectorSpaceConfig",
    "get_embedding_configuration",
    "get_embedding_profile",
    "get_vector_space",
    "reset_embedding_configuration_cache",
    "validate_embedding_configuration",
]
