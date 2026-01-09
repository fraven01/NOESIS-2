from typing import Any

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    """A chunk of knowledge used for retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    meta: dict[str, Any]
    """Metadata including tenant_id, case_id, source, hash and related fields."""

    embedding: list[float] | None = None
    parents: dict[str, dict[str, Any]] | None = None
    """Optional mapping of parent node identifiers to their metadata payloads."""
