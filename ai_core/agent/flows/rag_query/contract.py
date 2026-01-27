from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

flow_name = "rag_query"
flow_version = "0.1.0"
required_capabilities = ["rag.retrieve", "rag.compose", "rag.evidence"]
supported_scopes = ["CASE", "TENANT"]


class InputModel(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = Field(default=10, ge=1, le=50)

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OutputModel(BaseModel):
    answer: str
    citations: list[dict]
    claim_to_citation: dict[str, list[str]]
    retrieval_matches: list[dict]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


__all__ = [
    "flow_name",
    "flow_version",
    "required_capabilities",
    "supported_scopes",
    "InputModel",
    "OutputModel",
]
