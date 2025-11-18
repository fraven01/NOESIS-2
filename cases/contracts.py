"""Contracts for case lifecycle configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class CaseLifecycleTransition(BaseModel):
    """Single lifecycle transition definition."""

    model_config = ConfigDict(extra="forbid")

    from_phase: str | None = None
    to_phase: str = Field(..., min_length=1)
    trigger_events: list[str] = Field(default_factory=list)

    @field_validator("from_phase", "to_phase", mode="before")
    @classmethod
    def _trim_phase(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("phase must be a string")
        candidate = value.strip()
        if not candidate:
            return None
        return candidate

    @field_validator("trigger_events", mode="before")
    @classmethod
    def _coerce_events(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            candidate = value.strip()
            return [candidate] if candidate else []
        events: list[str] = []
        if isinstance(value, (list, tuple, set)):
            for entry in value:
                if entry in (None, ""):
                    continue
                events.append(str(entry).strip())
        else:
            raise ValueError("trigger_events must be a list or string")
        return [event for event in events if event]


class CaseLifecycleDefinition(BaseModel):
    """Lifecycle definition used to compute case phases."""

    model_config = ConfigDict(extra="forbid")

    phases: list[str] = Field(default_factory=list)
    transitions: list[CaseLifecycleTransition] = Field(default_factory=list)

    @field_validator("phases", mode="before")
    @classmethod
    def _coerce_phases(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            candidate = value.strip()
            return [candidate] if candidate else []
        if isinstance(value, (list, tuple, set)):
            phases: list[str] = []
            for entry in value:
                if entry in (None, ""):
                    continue
                phases.append(str(entry).strip())
            return [phase for phase in phases if phase]
        raise ValueError("phases must be a string or list of strings")


_CASE_DEFINITION_ADAPTER: TypeAdapter[CaseLifecycleDefinition] = TypeAdapter(
    CaseLifecycleDefinition
)


def parse_case_lifecycle_definition(
    value: object | None,
) -> CaseLifecycleDefinition | None:
    """Parse *value* into a lifecycle definition or return ``None``."""

    if value in (None, "", {}):
        return None
    if isinstance(value, CaseLifecycleDefinition):
        return value
    return _CASE_DEFINITION_ADAPTER.validate_python(value)


DEFAULT_CASE_LIFECYCLE_DEFINITION = CaseLifecycleDefinition(
    phases=["intake", "evidence_collection", "external_review", "search_completed"],
    transitions=[
        CaseLifecycleTransition(
            from_phase=None,
            to_phase="intake",
            trigger_events=["ingestion_run_queued", "ingestion_run_started"],
        ),
        CaseLifecycleTransition(
            from_phase="intake",
            to_phase="evidence_collection",
            trigger_events=["ingestion_run_completed"],
        ),
        CaseLifecycleTransition(
            from_phase=None,
            to_phase="evidence_collection",
            trigger_events=[
                "ingestion_run_completed",
                "collection_search:ingest_triggered",
            ],
        ),
        CaseLifecycleTransition(
            from_phase="evidence_collection",
            to_phase="external_review",
            trigger_events=["collection_search:hitl_pending"],
        ),
        CaseLifecycleTransition(
            from_phase="external_review",
            to_phase="search_completed",
            trigger_events=["collection_search:verified"],
        ),
        CaseLifecycleTransition(
            from_phase=None,
            to_phase="search_completed",
            trigger_events=["collection_search:verified"],
        ),
    ],
)


__all__ = [
    "CaseLifecycleDefinition",
    "CaseLifecycleTransition",
    "DEFAULT_CASE_LIFECYCLE_DEFINITION",
    "parse_case_lifecycle_definition",
]
