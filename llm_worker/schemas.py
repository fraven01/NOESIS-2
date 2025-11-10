from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, conint, field_validator


class ScoreResultInput(BaseModel):
    """Single result item supplied to the score_results task."""

    id: str = Field(..., min_length=1)
    title: str = Field(default="")
    snippet: str = Field(default="")
    url: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _ensure_id(cls, value: Any) -> str:
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
        raise ValueError("result id must be a non-empty string")

    @field_validator("title", "snippet", mode="before")
    @classmethod
    def _normalise_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("url", mode="before")
    @classmethod
    def _normalise_url(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value).strip()


class ScoreResultsData(BaseModel):
    """Payload for ranking query results."""

    query: str = Field(..., min_length=1)
    results: list[ScoreResultInput] = Field(..., min_length=1)
    criteria: list[str] | None = None
    k: conint(ge=1, le=50) = 5

    @field_validator("query", mode="before")
    @classmethod
    def _normalise_query(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        candidate = value.strip()
        if not candidate:
            raise ValueError("query must not be empty")
        return candidate

    @field_validator("criteria", mode="before")
    @classmethod
    def _normalise_criteria(cls, value: Any) -> list[str] | None:
        if value in (None, "", []):
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else None
        if isinstance(value, (list, tuple, set)):
            cleaned_items: list[str] = []
            for item in value:
                if item in (None, ""):
                    continue
                cleaned = str(item).strip()
                if cleaned:
                    cleaned_items.append(cleaned)
            return cleaned_items or None
        raise ValueError("criteria must be a sequence of strings")


class RagQueryTask(BaseModel):
    """Legacy rag_query task payload."""

    task_type: Literal["rag_query"] = "rag_query"
    control: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class ScoreResultsTask(BaseModel):
    """score_results task payload."""

    task_type: Literal["score_results"] = "score_results"
    control: dict[str, Any] = Field(default_factory=dict)
    data: ScoreResultsData


WorkerTask = Annotated[
    RagQueryTask | ScoreResultsTask, Field(discriminator="task_type")
]

__all__ = [
    "ScoreResultInput",
    "ScoreResultsData",
    "RagQueryTask",
    "ScoreResultsTask",
    "WorkerTask",
]
