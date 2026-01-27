from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ArtifactV0(BaseModel):
    run_id: str
    inputs_hash: str
    decision_log: list[dict[str, Any]]
    stop_decision: dict[str, Any]
    retrieval: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None
    citations: list[Any] | None = None
    claim_to_citation: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


__all__ = ["ArtifactV0"]
