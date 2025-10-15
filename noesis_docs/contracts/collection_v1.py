"""Collection contract models (v1)."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator
from typing_extensions import Annotated

_MAX_IDENTIFIER_LENGTH = 128
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

SlugStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=_MAX_IDENTIFIER_LENGTH,
        pattern=_IDENTIFIER_PATTERN.pattern,
    ),
]


class CollectionRef(BaseModel):
    """Reference information for a collection."""

    tenant_id: str
    collection_id: UUID
    slug: Optional[SlugStr] = None
    version_label: Optional[SlugStr] = None

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _normalise_tenant_id(cls, value: str) -> str:
        if isinstance(value, str):
            return unicodedata.normalize("NFKC", value)
        return value

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

        normalized = unicodedata.normalize("NFKC", value)
        trimmed = normalized.strip()
        if trimmed == "":
            return None
        return trimmed


class CollectionLink(CollectionRef):
    """Link information for a collection."""

    pass
