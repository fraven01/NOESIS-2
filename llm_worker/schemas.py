from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Mapping
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    confloat,
    conint,
    field_validator,
)

from common.validators import normalise_str_sequence, optional_str, require_trimmed_str
from llm_worker.utils.normalisation import clamp_score, ensure_aware_utc, coerce_enum


def _ensure_facet_score(value: Any) -> float:
    return clamp_score(value, minimum=0.0, maximum=1.0)


class FreshnessMode(str, Enum):
    STANDARD = "standard"
    SOFTWARE_DOCS_STRICT = "software_docs_strict"
    LAW_EVERGREEN = "law_evergreen"


class ScoringContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1)
    jurisdiction: str = Field(default="DE", min_length=2)
    output_target: str = Field(..., min_length=1)
    preferred_sources: list[str] = Field(default_factory=list)
    disallowed_sources: list[str] = Field(default_factory=list)
    collection_scope: str = Field(..., min_length=1)
    version_target: str | None = None
    freshness_mode: FreshnessMode = Field(default=FreshnessMode.STANDARD)
    min_diversity_buckets: int | None = Field(default=None)

    @field_validator(
        "question", "purpose", "output_target", "collection_scope", mode="before"
    )
    @classmethod
    def _require_trimmed(cls, value: Any) -> str:
        return require_trimmed_str(value)

    @field_validator("jurisdiction", mode="before")
    @classmethod
    def _normalise_jurisdiction(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("jurisdiction must be a string")
        candidate = value.strip().upper()
        if len(candidate) < 2:
            raise ValueError("jurisdiction must be at least two characters")
        return candidate

    @field_validator("preferred_sources", "disallowed_sources", mode="before")
    @classmethod
    def _normalise_sources(cls, value: Any) -> list[str]:
        return normalise_str_sequence(
            value,
            field_name="sources",
            error_message="sources must be a sequence of strings",
            return_type=list,
        )

    @field_validator("version_target", mode="before")
    @classmethod
    def _normalise_optional_str(cls, value: Any) -> str | None:
        return optional_str(value)

    @field_validator("min_diversity_buckets", mode="before")
    @classmethod
    def _normalise_min_diversity(cls, value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            bucket = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_diversity_buckets must be an integer") from exc
        if bucket < 1:
            raise ValueError("min_diversity_buckets must be at least 1")
        if bucket > 10:
            raise ValueError("min_diversity_buckets must not exceed 10")
        return bucket


class CoverageDimension(str, Enum):
    LEGAL = "LEGAL"
    TECHNICAL = "TECHNICAL"
    PROCEDURAL = "PROCEDURAL"
    DATA_CATEGORIES = "DATA_CATEGORIES"
    MONITORING_SURVEILLANCE = "MONITORING_SURVEILLANCE"
    LOGGING_AUDIT = "LOGGING_AUDIT"
    ANALYTICS_REPORTING = "ANALYTICS_REPORTING"
    ACCESS_PRIVACY_SECURITY = "ACCESS_PRIVACY_SECURITY"
    API_INTEGRATION = "API_INTEGRATION"


class RAGCoverageSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    document_id: UUID
    title: str = Field(..., min_length=1)
    url: str | None = None
    key_points: list[str] = Field(..., min_length=3, max_length=5)
    coverage_facets: dict[CoverageDimension, float] = Field(default_factory=dict)
    custom_facets: dict[str, float] = Field(default_factory=dict)
    last_ingested_at: datetime

    @field_validator("title", mode="before")
    @classmethod
    def _normalise_title(cls, value: Any) -> str:
        return require_trimmed_str(value, field_name="title")

    @field_validator("url", mode="before")
    @classmethod
    def _normalise_url(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value).strip()

    @field_validator("key_points", mode="before")
    @classmethod
    def _normalise_key_points(cls, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("key_points must be a list of strings")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("key_points must only contain strings")
            candidate = item.strip()
            if not candidate:
                raise ValueError("key_points items must not be empty")
            cleaned.append(candidate)
        if not 3 <= len(cleaned) <= 5:
            raise ValueError("key_points must contain between 3 and 5 items")
        return cleaned

    @field_validator("coverage_facets", mode="before")
    @classmethod
    def _normalise_coverage_facets(cls, value: Any) -> dict[CoverageDimension, float]:
        if value in (None, {}):
            return {}
        if not isinstance(value, dict):
            raise ValueError("coverage_facets must be a mapping")
        normalised: dict[CoverageDimension, float] = {}
        for key, score in value.items():
            dimension = coerce_enum(key, CoverageDimension)
            if dimension is None:
                raise ValueError(f"invalid coverage dimension: {key}")
            normalised[dimension] = _ensure_facet_score(score)
        return normalised

    @field_validator("custom_facets", mode="before")
    @classmethod
    def _normalise_custom_facets(cls, value: Any) -> dict[str, float]:
        if value in (None, {}):
            return {}
        if not isinstance(value, dict):
            raise ValueError("custom_facets must be a mapping")
        normalised: dict[str, float] = {}
        for key, score in value.items():
            if not isinstance(key, str):
                raise ValueError("custom facet keys must be strings")
            candidate = key.strip()
            if not candidate:
                raise ValueError("custom facet keys must not be empty")
            normalised[candidate] = _ensure_facet_score(score)
        return normalised


class ScoreResultInput(BaseModel):
    """Single result item supplied to the score_results task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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


class SearchCandidate(ScoreResultInput):
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_pdf: bool | None = None
    detected_date: datetime | None = None
    version_hint: str | None = None
    domain_type: str | None = None
    trust_hint: str | None = None

    @field_validator("version_hint", "trust_hint", mode="before")
    @classmethod
    def _optional_trimmed(cls, value: Any) -> str | None:
        return optional_str(value)

    @field_validator("domain_type", mode="before")
    @classmethod
    def _normalise_domain_type(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("domain_type must be a string")
        candidate = value.strip()
        if not candidate:
            raise ValueError("domain_type must not be empty")
        return candidate

    @field_validator("detected_date", mode="before")
    @classmethod
    def _normalise_detected_date(cls, value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        normalised = ensure_aware_utc(value)
        if normalised is None:
            raise ValueError("detected_date must include timezone information")
        return normalised


class ScoreResultsData(BaseModel):
    """Payload for ranking query results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(..., min_length=1)
    results: list[SearchCandidate] = Field(..., min_length=1)
    criteria: list[str] | None = None
    k: conint(ge=1, le=50) = 5

    @field_validator("query", mode="before")
    @classmethod
    def _normalise_query(cls, value: Any) -> str:
        return require_trimmed_str(value, field_name="query")

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

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_type: Literal["rag_query"] = "rag_query"
    control: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class ScoreResultsTask(BaseModel):
    """score_results task payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_type: Literal["score_results"] = "score_results"
    control: dict[str, Any] = Field(default_factory=dict)
    data: ScoreResultsData


WorkerTaskPayload = Annotated[
    RagQueryTask | ScoreResultsTask, Field(discriminator="task_type")
]

WorkerTask = TypeAdapter(WorkerTaskPayload)
WorkerTask.model_validate = WorkerTask.validate_python


class LLMScoredItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    candidate_id: str = Field(..., min_length=1)
    score: confloat(ge=0, le=100)
    reason: str = Field(..., min_length=1)
    gap_tags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    facet_coverage: dict[CoverageDimension, float] = Field(default_factory=dict)
    custom_facets: dict[str, float] = Field(default_factory=dict)

    @field_validator("candidate_id", mode="before")
    @classmethod
    def _normalise_candidate_id(cls, value: Any) -> str:
        return require_trimmed_str(value, field_name="candidate_id")

    @field_validator("reason", mode="before")
    @classmethod
    def _normalise_reason(cls, value: Any) -> str:
        return require_trimmed_str(value, field_name="reason")

    @field_validator("gap_tags", "risk_flags", mode="before")
    @classmethod
    def _normalise_tags(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            candidate = value.strip()
            return [candidate] if candidate else []
        if isinstance(value, (list, tuple, set)):
            cleaned: list[str] = []
            for item in value:
                if item in (None, ""):
                    continue
                cleaned_item = str(item).strip()
                if cleaned_item:
                    cleaned.append(cleaned_item)
            return cleaned
        raise ValueError("tags must be a sequence of strings")

    @field_validator("facet_coverage", mode="before")
    @classmethod
    def _normalise_facet_coverage(cls, value: Any) -> dict[CoverageDimension, float]:
        if value in (None, {}):
            return {}
        if not isinstance(value, dict):
            raise ValueError("facet_coverage must be a mapping")
        normalised: dict[CoverageDimension, float] = {}
        for key, score in value.items():
            dimension = coerce_enum(key, CoverageDimension)
            if dimension is None:
                raise ValueError(f"invalid coverage dimension: {key}")
            normalised[dimension] = _ensure_facet_score(score)
        return normalised

    @field_validator("custom_facets", mode="before")
    @classmethod
    def _normalise_custom_facets(cls, value: Any) -> dict[str, float]:
        if value in (None, {}):
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("custom_facets must be a mapping")
        normalised: dict[str, float] = {}
        for key, raw in value.items():
            if key in (None, ""):
                continue
            label = str(key).strip().upper()
            if not label:
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("custom facet scores must be numeric") from exc
            if score < 0:
                score = 0.0
            if score > 1:
                score = 1.0
            normalised[label] = score
        return normalised


class RecommendedIngestItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)

    @field_validator("candidate_id", "reason", mode="before")
    @classmethod
    def _normalise_non_empty(cls, value: Any) -> str:
        return require_trimmed_str(value)


class HybridResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ranked: list[LLMScoredItem] = Field(default_factory=list)
    top_k: list[LLMScoredItem] = Field(default_factory=list)
    coverage_delta: str = Field(..., min_length=1)
    recommended_ingest: list[RecommendedIngestItem] = Field(default_factory=list)

    @field_validator("coverage_delta", mode="before")
    @classmethod
    def _normalise_coverage_delta(cls, value: Any) -> str:
        return require_trimmed_str(value, field_name="coverage_delta")


__all__ = [
    "CoverageDimension",
    "FreshnessMode",
    "HybridResult",
    "LLMScoredItem",
    "RAGCoverageSummary",
    "RecommendedIngestItem",
    "ScoringContext",
    "SearchCandidate",
    "ScoreResultInput",
    "ScoreResultsData",
    "RagQueryTask",
    "ScoreResultsTask",
    "WorkerTask",
    "WorkerTaskPayload",
]
