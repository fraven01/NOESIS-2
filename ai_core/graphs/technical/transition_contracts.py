"""Shared transition contracts for LangGraph-inspired orchestrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from ai_core.api import EmbeddingResult
from ai_core.contracts.payloads import (
    DeltaPayload,
    GuardrailPayload as GuardrailStatePayload,
)
from ai_core.rag.ingestion_contracts import ChunkMeta
from documents.processing_graph import DocumentProcessingPhase

if TYPE_CHECKING:  # pragma: no cover - import cycle guards
    from ai_core.middleware.guardrails import GuardrailDecision
    from ai_core.rag.delta import DeltaDecision
    from documents.api import LifecycleStatusUpdate


GraphPhase = Literal[
    "update_status",
    "guardrails",
    "document_pipeline",
    "ingest_decision",
    "ingest",
    "finish",
    "accept_upload",
    "quarantine_scan",
    "deduplicate",
    "parse",
    "normalize",
    "delta_and_guardrails",
    "persist_document",
    "chunk_and_embed",
    "lifecycle_hook",
    "finalize",
    "framework_analysis",
]


class PipelineSection(BaseModel):
    """Details about document pipeline execution."""

    phase: Optional[str] = None
    run_until: Optional[DocumentProcessingPhase] = None
    error: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True, frozen=True)


class LifecycleSection(BaseModel):
    """Lifecycle updates emitted during a transition."""

    status: str
    policy_events: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


class _MappingSection(BaseModel):
    """Base model capturing mappings while keeping them JSON friendly."""

    attributes: Mapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @field_serializer("attributes")
    def _serialise_attributes(
        self, value: Mapping[str, Any], _: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return dict(value)


class DeltaSection(_MappingSection):
    decision: str
    reason: str


class GuardrailSection(_MappingSection):
    decision: str
    reason: str
    allowed: bool
    policy_events: tuple[str, ...] = Field(default_factory=tuple)


class EmbeddingSection(BaseModel):
    """Embedding invocation outcome."""

    result: EmbeddingResult

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @field_serializer("result")
    def _serialise_result(
        self, value: EmbeddingResult, _: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return value.to_dict()


class StandardTransitionResult(BaseModel):
    """Canonical transition payload for LangGraph nodes."""

    phase: GraphPhase
    decision: str
    reason: str
    severity: str = "info"
    pipeline: Optional[PipelineSection] = None
    lifecycle: Optional[LifecycleSection] = None
    delta: Optional[DeltaSection] = None
    guardrail: Optional[GuardrailSection] = None
    embedding: Optional[EmbeddingSection] = None
    context: Mapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @model_validator(mode="after")
    def _normalise(self) -> "StandardTransitionResult":
        decision = (self.decision or "").strip()
        if not decision:
            raise ValueError("decision_required")
        reason = (self.reason or "").strip()
        if not reason:
            raise ValueError("reason_required")
        severity = (self.severity or "info").strip().lower() or "info"
        context = dict(self.context or {})
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "context", context)
        return self

    @field_serializer("context")
    def _serialise_context(
        self, value: Mapping[str, Any], _: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return dict(value)


@dataclass(frozen=True)
class GraphTransition:
    """Typed transition wrapper used by orchestration graphs."""

    result: StandardTransitionResult

    @property
    def phase(self) -> GraphPhase:
        return self.result.phase

    @property
    def decision(self) -> str:
        return self.result.decision

    @property
    def reason(self) -> str:
        return self.result.reason

    @property
    def severity(self) -> str:
        return self.result.severity

    @property
    def context(self) -> Mapping[str, Any]:
        return self.result.context

    def with_context(self, extra: Mapping[str, Any]) -> "GraphTransition":
        merged: Dict[str, Any] = dict(self.context)
        for key, value in (extra or {}).items():
            if value is None:
                continue
            merged[str(key)] = value
        updated = self.result.model_copy(update={"context": merged})
        return GraphTransition(updated)

    def to_dict(self) -> Dict[str, Any]:
        return self.result.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphTransition":
        """Create a transition from a dictionary."""
        # Map 'attributes' to 'context' for compatibility
        if "attributes" in data and "context" not in data:
            data["context"] = data.pop("attributes")

        # Default phase if missing
        if "phase" not in data:
            data["phase"] = "framework_analysis"

        return cls(result=StandardTransitionResult(**data))


def build_lifecycle_section(
    update: Optional["LifecycleStatusUpdate"],
) -> Optional[LifecycleSection]:
    if update is None:
        return None
    policy_events: Iterable[str] = getattr(update.record, "policy_events", ())
    return LifecycleSection(status=update.status, policy_events=tuple(policy_events))


def build_delta_section(
    decision: Optional["DeltaDecision" | DeltaPayload],
) -> Optional[DeltaSection]:
    if decision is None:
        return None
    if isinstance(decision, DeltaPayload):
        return DeltaSection(
            decision=decision.decision,
            reason=decision.reason,
            attributes=dict(decision.attributes),
        )
    return DeltaSection(
        decision=decision.decision,
        reason=decision.reason,
        attributes=dict(decision.attributes),
    )


def build_guardrail_section(
    decision: Optional["GuardrailDecision" | GuardrailStatePayload],
) -> Optional[GuardrailSection]:
    if decision is None:
        return None
    if isinstance(decision, GuardrailStatePayload):
        return GuardrailSection(
            decision=decision.decision,
            reason=decision.reason,
            allowed=decision.allowed,
            policy_events=tuple(decision.policy_events),
            attributes=dict(decision.attributes),
        )
    policy_events: Iterable[str] = getattr(decision, "policy_events", ())
    attributes: Mapping[str, Any]
    raw_attributes = getattr(decision, "attributes", {})
    if isinstance(raw_attributes, Mapping):
        attributes = raw_attributes
    else:
        attributes = {}
    return GuardrailSection(
        decision=decision.decision,
        reason=decision.reason,
        allowed=getattr(decision, "allowed", False),
        policy_events=tuple(policy_events),
        attributes=dict(attributes),
    )


def build_embedding_section(
    result: Optional[EmbeddingResult],
) -> Optional[EmbeddingSection]:
    if result is None:
        return None
    if not isinstance(result.chunk_meta, ChunkMeta):
        raise TypeError("chunk_meta_required")
    return EmbeddingSection(result=result)


__all__ = [
    "DeltaSection",
    "EmbeddingSection",
    "GraphPhase",
    "GraphTransition",
    "GuardrailSection",
    "LifecycleSection",
    "PipelineSection",
    "StandardTransitionResult",
    "build_delta_section",
    "build_embedding_section",
    "build_guardrail_section",
    "build_lifecycle_section",
]
