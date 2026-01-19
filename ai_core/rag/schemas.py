from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A chunk of knowledge used for retrieval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    meta: dict[str, Any]
    """Metadata including tenant_id, case_id, source, hash and related fields."""

    embedding: list[float] | None = None
    parents: dict[str, dict[str, Any]] | None = None
    """Optional mapping of parent node identifiers to their metadata payloads."""


class SourceRef(BaseModel):
    """Reference to a source snippet with an LLM-assessed relevance score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    label: str
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM-assessed relevance score for the source.",
    )


class RagReasoning(BaseModel):
    """Structured reasoning payload for the RAG response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis: str = Field(description="Brief reasoning summary for the answer.")
    gaps: list[str] = Field(
        default_factory=list,
        description="Missing information that prevents a complete answer.",
    )


class RagResponse(BaseModel):
    """Structured RAG response with reasoning and source usage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: RagReasoning
    answer_markdown: str = Field(description="Final answer in Markdown format.")
    used_sources: list[SourceRef] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list, max_length=3)
