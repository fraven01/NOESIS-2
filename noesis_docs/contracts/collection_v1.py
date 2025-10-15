"""Collection contract models (v1)."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAX_IDENTIFIER_LENGTH = 128


class CollectionRef(BaseModel):
    """Reference information for a collection."""

    tenant_id: str
    collection_id: UUID
    slug: Optional[str] = Field(default=None, max_length=_MAX_IDENTIFIER_LENGTH)
    version_label: Optional[str] = Field(default=None, max_length=_MAX_IDENTIFIER_LENGTH)

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("tenant_id must not be empty")
        return trimmed

    @field_validator("slug", "version_label", mode="before")
    @classmethod
    def _normalise_optional_identifiers(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("value must be a string")

        trimmed = value.strip()
        if trimmed == "":
            return None
        return trimmed


class CollectionLink(CollectionRef):
    """Link information for a collection."""

    model_config = CollectionRef.model_config
