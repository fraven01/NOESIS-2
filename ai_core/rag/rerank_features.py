"""Structure-aware rerank feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ai_core.rag.evidence_graph import EvidenceGraph
from ai_core.rag.schemas import Chunk
from ai_core.tool_contracts import ToolContext


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return 0.0
    if candidate != candidate or candidate in (float("inf"), float("-inf")):
        return 0.0
    return candidate


def _coerce_score(value: object) -> float:
    return max(0.0, min(1.0, _coerce_float(value)))


def _extract_section_path(meta: Mapping[str, Any]) -> tuple[str, ...]:
    raw = meta.get("section_path")
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw if str(item).strip())
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    return ()


def _extract_doc_type(meta: Mapping[str, Any]) -> str | None:
    for key in ("doc_type", "doc_class", "document_type"):
        candidate = _coerce_str(meta.get(key))
        if candidate:
            return candidate
    return None


def _extract_parent_relevance(meta: Mapping[str, Any]) -> float:
    parents = meta.get("parents")
    if not isinstance(parents, Iterable) or isinstance(
        parents, (str, bytes, bytearray)
    ):
        return 0.0
    scores = []
    for parent in parents:
        if not isinstance(parent, Mapping):
            continue
        score = (
            parent.get("score")
            if parent.get("score") is not None
            else parent.get("relevance") or parent.get("confidence")
        )
        scores.append(_coerce_score(score))
    return max(scores) if scores else 0.0


def _resolve_quality_mode(quality_mode: str | None, context: ToolContext | None) -> str:
    if quality_mode:
        return quality_mode.strip().lower()
    if context is not None:
        meta = getattr(context, "metadata", {})
        if isinstance(meta, Mapping):
            candidate = _coerce_str(meta.get("quality_mode"))
            if candidate:
                return candidate.lower()
    return "standard"


def _resolve_doc_type(context: ToolContext | None) -> str | None:
    if context is None:
        return None
    meta = getattr(context, "metadata", {})
    if isinstance(meta, Mapping):
        return _coerce_str(meta.get("doc_class") or meta.get("doc_type"))
    return None


_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "standard": {
        "parent_relevance": 0.2,
        "section_match": 0.2,
        "confidence": 0.3,
        "adjacency_bonus": 0.2,
        "doc_type_match": 0.1,
    },
    "precision": {
        "parent_relevance": 0.2,
        "section_match": 0.3,
        "confidence": 0.3,
        "adjacency_bonus": 0.1,
        "doc_type_match": 0.1,
    },
    "recall": {
        "parent_relevance": 0.1,
        "section_match": 0.1,
        "confidence": 0.4,
        "adjacency_bonus": 0.2,
        "doc_type_match": 0.2,
    },
}


def get_static_weight_profile(quality_mode: str | None) -> dict[str, float]:
    mode = (quality_mode or "standard").strip().lower()
    return dict(_WEIGHT_PROFILES.get(mode, _WEIGHT_PROFILES["standard"]))


def resolve_weight_profile(
    quality_mode: str | None,
    *,
    context: ToolContext | None = None,
) -> dict[str, float]:
    mode = (quality_mode or "standard").strip().lower()
    base = get_static_weight_profile(mode)
    try:
        import os

        mode_flag = os.getenv("RAG_RERANK_WEIGHT_MODE", "static").strip().lower()
        if mode_flag not in {"learned", "adaptive"}:
            return base
        if context is None:
            return base
        tenant_id = context.scope.tenant_id
        if not tenant_id:
            return base
        from ai_core.rag.feedback import get_learned_weight_profile

        learned = get_learned_weight_profile(tenant_id, mode)
        if learned:
            return learned
    except Exception:
        return base
    return base


@dataclass(frozen=True)
class RerankFeatures:
    chunk_id: str
    parent_relevance: float
    section_match: float
    confidence: float
    adjacency_bonus: float
    doc_type_match: float
    weighted_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "parent_relevance": self.parent_relevance,
            "section_match": self.section_match,
            "confidence": self.confidence,
            "adjacency_bonus": self.adjacency_bonus,
            "doc_type_match": self.doc_type_match,
            "weighted_score": self.weighted_score,
        }


def _build_chunk_from_match(match: Mapping[str, Any], index: int) -> Chunk:
    meta = match.get("meta")
    meta = dict(meta) if isinstance(meta, Mapping) else {}
    chunk_id = _coerce_str(meta.get("chunk_id") or match.get("id")) or f"match-{index}"
    document_id = _coerce_str(meta.get("document_id") or match.get("id"))
    section_path = meta.get("section_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    chunk_index = meta.get("chunk_index")
    if chunk_index is None:
        chunk_index = index
    chunk_meta = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "section_path": section_path,
        "chunk_index": chunk_index,
        "parent_ids": meta.get("parent_ids") or [],
        "score": match.get("score", meta.get("score", 0.0)),
    }
    parents_payload = meta.get("parents")
    if parents_payload is not None:
        chunk_meta["parents"] = parents_payload
    doc_type = _extract_doc_type(meta)
    if doc_type:
        chunk_meta["doc_type"] = doc_type
    return Chunk(content=str(match.get("text") or ""), meta=chunk_meta)


def extract_rerank_features(
    matches: Sequence[Mapping[str, Any]],
    *,
    context: ToolContext | None = None,
    quality_mode: str | None = None,
) -> list[RerankFeatures]:
    materialized = [m for m in matches if isinstance(m, Mapping)]
    if not materialized:
        return []

    chunks = [
        _build_chunk_from_match(match, idx) for idx, match in enumerate(materialized)
    ]
    graph = EvidenceGraph.from_chunks(chunks)
    anchor_id = max(graph.nodes.values(), key=lambda node: node.score).chunk_id
    anchor_section = graph.nodes[anchor_id].section_path
    anchor_doc_type = _resolve_doc_type(context)

    weights = resolve_weight_profile(
        _resolve_quality_mode(quality_mode, context),
        context=context,
    )
    features: list[RerankFeatures] = []

    for node in graph.nodes.values():
        chunk_id = node.chunk_id
        meta = chunks[node.rank].meta or {}
        parent_relevance = _extract_parent_relevance(meta)
        section_match = 1.0 if node.section_path == anchor_section else 0.0
        confidence = _coerce_score(meta.get("score"))
        adjacency_bonus = 1.0 if chunk_id in graph.get_adjacent(anchor_id) else 0.0
        doc_type = _extract_doc_type(meta)
        doc_type_match = (
            1.0 if doc_type and anchor_doc_type and doc_type == anchor_doc_type else 0.0
        )
        weighted_score = (
            parent_relevance * weights["parent_relevance"]
            + section_match * weights["section_match"]
            + confidence * weights["confidence"]
            + adjacency_bonus * weights["adjacency_bonus"]
            + doc_type_match * weights["doc_type_match"]
        )
        features.append(
            RerankFeatures(
                chunk_id=chunk_id,
                parent_relevance=parent_relevance,
                section_match=section_match,
                confidence=confidence,
                adjacency_bonus=adjacency_bonus,
                doc_type_match=doc_type_match,
                weighted_score=weighted_score,
            )
        )

    return features


def summarise_features(
    features: Sequence[RerankFeatures],
) -> dict[str, dict[str, float]]:
    if not features:
        return {}
    keys = [
        "parent_relevance",
        "section_match",
        "confidence",
        "adjacency_bonus",
        "doc_type_match",
        "weighted_score",
    ]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [getattr(feature, key) for feature in features]
        summary[key] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    return summary


__all__ = [
    "RerankFeatures",
    "extract_rerank_features",
    "get_static_weight_profile",
    "resolve_weight_profile",
    "summarise_features",
]
