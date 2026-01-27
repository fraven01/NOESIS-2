from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai_core.agent.capabilities.base import CapabilitySpec
from ai_core.agent.capabilities.registry import register_capability
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.tool_contracts.base import ToolContext


class RagEvidenceInput(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    mode: str = "heuristic"

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RagEvidenceOutput(BaseModel):
    claim_to_citation: dict[str, list[str]]
    ungrounded_claims: list[str] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


_WORD_RE = re.compile(r"[a-z0-9]+")


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"[.!?]+", text)
    return [part.strip() for part in raw if part and part.strip()]


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _citation_ref(citation: dict[str, Any]) -> str:
    return str(citation.get("id") or citation.get("url") or citation.get("title") or "")


def execute(
    tool_context: ToolContext,
    runtime_config: RuntimeConfig,
    input_model: RagEvidenceInput,
) -> RagEvidenceOutput:
    _ = tool_context, runtime_config
    claims = _split_sentences(input_model.answer)
    claim_map: dict[str, list[str]] = {}
    ungrounded: list[str] = []

    citation_tokens: list[tuple[str, set[str]]] = []
    for citation in input_model.citations:
        ref = _citation_ref(citation)
        text = " ".join(
            str(citation.get(key, "")) for key in ("title", "snippet", "text", "url")
        )
        citation_tokens.append((ref, _tokenize(text)))

    for claim in claims:
        claim_tokens = _tokenize(claim)
        refs = [ref for ref, tokens in citation_tokens if ref and claim_tokens & tokens]
        refs_sorted = sorted(refs)
        if refs_sorted:
            claim_map[claim] = refs_sorted
        else:
            ungrounded.append(claim)

    return RagEvidenceOutput(
        claim_to_citation=claim_map,
        ungrounded_claims=ungrounded,
        telemetry={},
    )


IO_SPEC = GraphIOSpec(
    schema_id="capability.rag.evidence",
    version=GraphIOVersion(major=0, minor=1, patch=0),
    input_model=RagEvidenceInput,
    output_model=RagEvidenceOutput,
)

CAPABILITY = CapabilitySpec(
    name="rag.evidence",
    version="0.1.0",
    io_spec_version=IO_SPEC.version_string,
    input_model=RagEvidenceInput,
    output_model=RagEvidenceOutput,
    execute=execute,
    io_spec=IO_SPEC,
)

register_capability(CAPABILITY)


__all__ = ["RagEvidenceInput", "RagEvidenceOutput", "execute", "CAPABILITY"]
