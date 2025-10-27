"""Provider-specific adapters shared between crawler and ingestion pipelines."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from documents.contract_utils import normalize_string
from documents.contracts import DocumentMeta

_PROVIDER_TAG_PREFIX = "provider_tag:"
_PROVIDER_TAG_KEY_LIMIT = 128 - len(_PROVIDER_TAG_PREFIX)
_EXTERNAL_VALUE_LIMIT = 512
_PROVIDER_VALUE_LIMIT = 128


class ProviderReference(BaseModel):
    """Adapter exposing crawler provider metadata from document contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(..., description="Name of the data provider")
    external_id: str = Field(..., description="Provider-native identifier")
    canonical_source: str = Field(..., description="Canonical source URI")
    provider_tags: Mapping[str, str] = Field(
        default_factory=dict,
        description="Normalized provider tags derived from external references.",
    )

    @field_validator("provider", "external_id", "canonical_source", mode="before")
    @classmethod
    def _normalize_identifier(cls, value: Optional[str]) -> str:
        normalized = normalize_string(str(value or ""))
        if not normalized:
            raise ValueError("identifier_required")
        return normalized

    @field_validator("provider_tags", mode="before")
    @classmethod
    def _normalize_tags(
        cls, value: Optional[Mapping[str, str]]
    ) -> Mapping[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("provider_tags_type")
        normalized: Dict[str, str] = {}
        for raw_key, raw_val in value.items():
            key = normalize_string(str(raw_key))
            if not key:
                raise ValueError("provider_tag_key")
            if len(key) > _PROVIDER_TAG_KEY_LIMIT:
                key = key[:_PROVIDER_TAG_KEY_LIMIT]
            val = normalize_string(str(raw_val))
            if not val:
                raise ValueError("provider_tag_value")
            if len(val) > _EXTERNAL_VALUE_LIMIT:
                val = val[:_EXTERNAL_VALUE_LIMIT]
            normalized[key] = val
        return normalized


def build_external_reference(
    *,
    provider: str,
    external_id: str,
    provider_tags: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Compose a normalized external reference payload for document metadata."""

    normalized_provider = _sanitize_external_value(provider, limit=_PROVIDER_VALUE_LIMIT)
    normalized_external_id = _sanitize_external_value(external_id)
    tags = _normalize_external_tags(provider_tags)
    data: Dict[str, str] = {
        "provider": normalized_provider,
        "external_id": normalized_external_id,
    }
    for key, value in tags.items():
        data[f"{_PROVIDER_TAG_PREFIX}{key}"] = value
    return data


def parse_provider_reference(meta: DocumentMeta) -> ProviderReference:
    """Return the provider reference derived from the document metadata."""

    external = meta.external_ref or {}
    provider = external.get("provider", "")
    external_id = external.get("external_id", "")
    canonical_source = meta.origin_uri or ""
    tags: Dict[str, str] = {}
    for key, value in external.items():
        if key.startswith(_PROVIDER_TAG_PREFIX):
            tag_key = key[len(_PROVIDER_TAG_PREFIX) :]
            tags[tag_key] = value
    return ProviderReference(
        provider=provider,
        external_id=external_id,
        canonical_source=canonical_source,
        provider_tags=tags,
    )


def _sanitize_external_value(value: str, *, limit: int = _EXTERNAL_VALUE_LIMIT) -> str:
    normalized = normalize_string(value)
    if not normalized:
        raise ValueError("external_ref_value")
    if len(normalized) > limit:
        return normalized[:limit]
    return normalized


def _normalize_external_tags(
    provider_tags: Optional[Mapping[str, str]]
) -> Dict[str, str]:
    if not provider_tags:
        return {}
    normalized: Dict[str, str] = {}
    for key, value in provider_tags.items():
        norm_key = normalize_string(str(key))
        if not norm_key:
            continue
        if len(norm_key) > _PROVIDER_TAG_KEY_LIMIT:
            norm_key = norm_key[:_PROVIDER_TAG_KEY_LIMIT]
        norm_val = normalize_string(str(value))
        if not norm_val:
            continue
        if len(norm_val) > _EXTERNAL_VALUE_LIMIT:
            norm_val = norm_val[:_EXTERNAL_VALUE_LIMIT]
        normalized[norm_key] = norm_val
    return normalized


__all__ = ["ProviderReference", "build_external_reference", "parse_provider_reference"]
