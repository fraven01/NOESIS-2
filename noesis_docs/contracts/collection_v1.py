"""Collection contract models (v1).

This module exposes Pydantic v2 models that ensure the core constraints we enforce
for collection references in the Noesis platform. The contracts are intentionally
lean and focus on runtime validation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator
from typing_extensions import Annotated

__all__ = [
    "CollectionRef",
    "CollectionLink",
    "collection_ref_schema",
    "collection_link_schema",
]

_MAX_IDENTIFIER_LENGTH = 128
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_INVISIBLE_UNICODE_CATEGORIES = {"Cf", "Cc", "Cs"}

# NOTE: We keep the slug constraints generic so they can be reused for both the
# optional ``slug`` and ``version_label`` fields. Pydantic will handle empty
# values as ``None`` after we trim them in the pre-validation hook below.
SlugStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=_MAX_IDENTIFIER_LENGTH,
        pattern=_IDENTIFIER_PATTERN.pattern,
    ),
]


def _strip_invisible(text: str) -> str:
    """Remove invisible/control characters from ``text``."""

    return "".join(
        char for char in text if unicodedata.category(char) not in _INVISIBLE_UNICODE_CATEGORIES
    )


class CollectionRef(BaseModel):
    """Reference information for a collection."""

    tenant_id: str
    collection_id: UUID
    slug: Optional[SlugStr] = None
    version_label: Optional[SlugStr] = None

    model_config = ConfigDict(extra="forbid")

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
        cleaned = _strip_invisible(trimmed)
        if not cleaned:
            raise ValueError("tenant_id must not be empty")
        return cleaned

    @field_validator("slug", "version_label", mode="before")
    @classmethod
    def _normalise_optional_identifiers(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("value must be a string")

        normalized = unicodedata.normalize("NFKC", value)
        trimmed = normalized.strip()
        cleaned = _strip_invisible(trimmed)
        if not cleaned:
            return None
        return cleaned


class CollectionLink(CollectionRef):
    """Link information for a collection."""

    pass


def collection_ref_schema() -> dict:
    """Return the JSON schema for :class:`CollectionRef`."""

    return CollectionRef.model_json_schema()


def collection_link_schema() -> dict:
    """Return the JSON schema for :class:`CollectionLink`."""

    return CollectionLink.model_json_schema()
