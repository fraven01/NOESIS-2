from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai_core.infra.prompts import load
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.tool_contracts import ToolContext
from ai_core.infra.observability import update_observation
from ai_core.rag.rerank_features import (
    extract_rerank_features,
    resolve_weight_profile,
    summarise_features,
)

logger = logging.getLogger(__name__)


DEFAULT_POOL_SIZE = 20
POOL_SIZE_CAP = 50


@dataclass(frozen=True)
class RerankResult:
    chunks: list[dict[str, Any]]
    mode: str
    prompt_version: str | None = None
    scores: dict[str, float] | None = None
    error: str | None = None


def _coerce_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score != score or score == float("inf") or score == float("-inf"):
        return 0.0
    return score


def _chunk_identifier(chunk: Mapping[str, Any], index: int) -> str:
    raw = chunk.get("id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    meta = chunk.get("meta")
    if isinstance(meta, Mapping):
        meta_id = meta.get("chunk_id") or meta.get("document_id")
        if isinstance(meta_id, str) and meta_id.strip():
            return meta_id.strip()
    return f"chunk-{index}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _extract_json_object(text: str) -> Mapping[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty response payload")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response payload missing JSON object")
    fragment = cleaned[start : end + 1]
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON payload") from exc
    if not isinstance(data, Mapping):
        raise ValueError("response payload must be a JSON object")
    return data


def _resolve_pool_size(value: object | None) -> int:
    try:
        candidate = int(str(value))
    except (TypeError, ValueError):
        candidate = DEFAULT_POOL_SIZE
    if candidate <= 0:
        candidate = DEFAULT_POOL_SIZE
    return min(candidate, POOL_SIZE_CAP)


def _heuristic_order(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        chunks,
        key=lambda item: (-_coerce_score(item.get("score")), str(item.get("id") or "")),
    )


def _render_candidates(chunks: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        identifier = _chunk_identifier(chunk, index - 1)
        text = str(chunk.get("text") or "")
        source = str(chunk.get("source") or "")
        snippet = _truncate(text.replace("\n", " ").strip(), 700)
        if source:
            lines.append(f"{index}. id={identifier} source={source}\ntext={snippet}")
        else:
            lines.append(f"{index}. id={identifier}\ntext={snippet}")
    return "\n".join(lines)


def _apply_ranked_order(
    base: Sequence[dict[str, Any]],
    ranked: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    id_to_chunk: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for index, chunk in enumerate(base):
        chunk_id = _chunk_identifier(chunk, index)
        if chunk_id not in id_to_chunk:
            id_to_chunk[chunk_id] = chunk
            ordered_ids.append(chunk_id)

    scores: dict[str, float] = {}
    ordered: list[dict[str, Any]] = []
    ranked_ids: set[str] = set()

    for item in ranked:
        raw_id = item.get("id")
        if not isinstance(raw_id, str):
            continue
        chunk_id = raw_id.strip()
        if not chunk_id or chunk_id in ranked_ids:
            continue
        chunk = id_to_chunk.get(chunk_id)
        if not chunk:
            continue
        score = _coerce_score(item.get("score"))
        scores[chunk_id] = score
        updated = dict(chunk)
        updated["score"] = score
        ordered.append(updated)
        ranked_ids.add(chunk_id)

    for chunk_id in ordered_ids:
        if chunk_id in ranked_ids:
            continue
        ordered.append(id_to_chunk[chunk_id])

    return ordered, scores


def rerank_chunks(
    chunks: Sequence[Mapping[str, Any]],
    query: str,
    context: ToolContext,
    *,
    top_k: int | None = None,
    candidate_pool: int | None = None,
    mode: str | None = None,
) -> RerankResult:
    materialized = [dict(chunk) for chunk in chunks if isinstance(chunk, Mapping)]
    if not materialized:
        return RerankResult(chunks=[], mode="off")

    try:
        quality_mode = os.getenv("RAG_RERANK_QUALITY_MODE")
        features = extract_rerank_features(
            materialized,
            context=context,
            quality_mode=quality_mode,
        )
        feature_summary = summarise_features(features)
        if feature_summary:
            weights = resolve_weight_profile(quality_mode, context=context)
            update_observation(
                metadata={
                    "rag.rerank_features.weights": weights,
                    "rag.rerank_features.summary": feature_summary,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "rag.rerank_features.failed",
            extra={"error": type(exc).__name__, "message": str(exc)},
        )

    resolved_mode = (
        str(mode).strip().lower()
        if mode is not None
        else str(os.getenv("RAG_RERANK_MODE", "heuristic")).strip().lower()
    )
    if resolved_mode in {"off", "disabled", "false"}:
        return RerankResult(chunks=materialized, mode="disabled")

    ordered = _heuristic_order(materialized)
    limit = top_k if top_k is not None else len(ordered)
    pool_limit = _resolve_pool_size(candidate_pool or os.getenv("RAG_RERANK_POOL"))
    pool = ordered[:pool_limit]

    if resolved_mode != "llm":
        return RerankResult(chunks=ordered[:limit], mode="heuristic")

    prompt = load("retriever/rerank")
    prompt_text = f"{prompt['text']}\n\nQuestion: {query}\n\nCandidates:\n{_render_candidates(pool)}"
    metadata = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "trace_id": context.scope.trace_id,
        "user_id": context.scope.user_id,
        "prompt_version": prompt["version"],
    }
    try:
        response = llm_client.call("analyze", prompt_text, metadata)
        payload = _extract_json_object(str(response.get("text") or ""))
        ranked = payload.get("ranked") or payload.get("results")
        if not isinstance(ranked, Sequence) or isinstance(
            ranked, (str, bytes, bytearray)
        ):
            raise ValueError("ranked list missing")
        reranked, scores = _apply_ranked_order(pool, ranked)
        if len(reranked) < len(ordered):
            pool_ids = {_chunk_identifier(chunk, idx) for idx, chunk in enumerate(pool)}
            remainder = [
                chunk
                for idx, chunk in enumerate(ordered)
                if _chunk_identifier(chunk, idx) not in pool_ids
            ]
            reranked.extend(remainder)
        return RerankResult(
            chunks=reranked[:limit],
            mode="llm",
            prompt_version=prompt.get("version"),
            scores=scores,
        )
    except (LlmClientError, RateLimitError, ValueError) as exc:
        logger.warning(
            "rag.rerank.failed",
            extra={
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        return RerankResult(
            chunks=ordered[:limit],
            mode="fallback",
            prompt_version=prompt.get("version"),
            error=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "rag.rerank.failed",
            extra={
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        return RerankResult(
            chunks=ordered[:limit],
            mode="fallback",
            prompt_version=prompt.get("version"),
            error=str(exc),
        )


__all__ = ["RerankResult", "rerank_chunks"]
