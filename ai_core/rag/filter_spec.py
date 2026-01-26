"""Typed filter specification for retrieval queries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _coerce_text(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    try:
        return str(value).strip() or None
    except Exception:
        return None


def _coerce_str_list(value: object | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    text = _coerce_text(value)
    return [text] if text else None


class FilterSpec(BaseModel):
    """Typed filters for retrieval requests."""

    tenant_id: str | None = Field(
        default=None, description="Tenant identifier for retrieval scoping."
    )
    case_id: str | None = Field(
        default=None, description="Case identifier for retrieval scoping."
    )
    collection_id: str | None = Field(
        default=None, description="Collection identifier for retrieval scoping."
    )
    collection_ids: list[str] | None = Field(
        default=None, description="Collection identifiers for retrieval scoping."
    )
    document_id: str | None = Field(
        default=None, description="Document identifier for narrow retrieval."
    )
    document_version_id: str | None = Field(
        default=None, description="Document version identifier for retrieval."
    )
    is_latest: bool | None = Field(
        default=None, description="Whether to filter to latest document version."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata filters."
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: object) -> object:
        if isinstance(data, FilterSpec):
            return data
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        known_keys = {
            "tenant_id",
            "case_id",
            "collection_id",
            "collection_ids",
            "document_id",
            "document_version_id",
            "is_latest",
            "metadata",
        }
        extra_payload = {
            key: value for key, value in payload.items() if key not in known_keys
        }
        if extra_payload:
            metadata = dict(payload.get("metadata") or {})
            metadata.update(extra_payload)
            payload["metadata"] = metadata
            for key in extra_payload:
                payload.pop(key, None)

        if "tenant_id" in payload:
            payload["tenant_id"] = _coerce_text(payload.get("tenant_id"))
        if "case_id" in payload:
            payload["case_id"] = _coerce_text(payload.get("case_id"))
        if "collection_id" in payload:
            payload["collection_id"] = _coerce_text(payload.get("collection_id"))
        if "collection_ids" in payload:
            payload["collection_ids"] = _coerce_str_list(payload.get("collection_ids"))
        if "document_id" in payload:
            payload["document_id"] = _coerce_text(payload.get("document_id"))
        if "document_version_id" in payload:
            payload["document_version_id"] = _coerce_text(
                payload.get("document_version_id")
            )
        return payload

    def as_mapping(self) -> dict[str, Any]:
        """Return the filters as a mapping for vector clients."""
        payload: dict[str, Any] = {
            "collection_id": self.collection_id,
            "collection_ids": self.collection_ids,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "is_latest": self.is_latest,
        }
        for key, value in self.metadata.items():
            if key in payload:
                continue
            payload[key] = value
        return {key: value for key, value in payload.items() if value is not None}


def build_filter_spec(
    *,
    tenant_id: str | None,
    case_id: str | None = None,
    collection_id: str | None = None,
    document_id: str | None = None,
    document_version_id: str | None = None,
    raw_filters: Mapping[str, Any] | None = None,
) -> FilterSpec:
    """Compose a FilterSpec from raw filters and explicit scope values."""
    payload = dict(raw_filters) if isinstance(raw_filters, Mapping) else {}
    return FilterSpec(
        tenant_id=_coerce_text(payload.pop("tenant_id", None)) or tenant_id,
        case_id=_coerce_text(payload.pop("case_id", None)) or case_id,
        collection_id=_coerce_text(payload.pop("collection_id", None)) or collection_id,
        collection_ids=_coerce_str_list(payload.pop("collection_ids", None)),
        document_id=_coerce_text(payload.pop("document_id", None)) or document_id,
        document_version_id=_coerce_text(payload.pop("document_version_id", None))
        or document_version_id,
        is_latest=payload.pop("is_latest", None),
        metadata=payload,
    )


__all__ = ["FilterSpec", "build_filter_spec"]
