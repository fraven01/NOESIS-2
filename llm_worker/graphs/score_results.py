from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterable, Iterator, Mapping

from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError
from common.logging import get_logger

from llm_worker.schemas import ScoreResultsData

logger = get_logger(__name__)

DEFAULT_MODEL_LABEL = "fast"
DEFAULT_PROMPT_VERSION = "score-results.v1"
DEFAULT_TRACE_ID = "score-results"
DEFAULT_TEMPERATURE = 0.1


@contextmanager
def _temporary_env_var(key: str, value: str) -> Iterator[None]:
    """Temporarily override an environment variable within a worker process."""

    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _build_prompt(payload: ScoreResultsData) -> str:
    sections: list[str] = [
        "Du bewertest Suchergebnisse deterministisch und vergibst Scores zwischen 0 und 100.",
        f"Query: {payload.query}",
        "",
        "Suchergebnisse:",
    ]
    for idx, item in enumerate(payload.results, start=1):
        sections.append(f"[{idx}] id={item.id}")
        if item.title:
            sections.append(f"Title: {item.title}")
        if item.snippet:
            sections.append(f"Snippet: {item.snippet}")
        if item.url:
            sections.append(f"URL: {item.url}")
        sections.append("")
    if payload.criteria:
        sections.append("Bewertungskriterien:")
        for criterion in payload.criteria:
            sections.append(f"- {criterion}")
        sections.append("")
    instructions = (
        "Bewerte jedes Ergebnis strikt anhand der Query und Kriterien. "
        "Erlaube nur ganze Scores zwischen 0 und 100 (0 = irrelevant, 100 = perfekt). "
        "Gib ausschließlich folgenden JSON zurück: "
        '{"ranked":[{"id":"<result_id>","score":0-100,"reasons":["..."]}, ...]}'
    )
    sections.append(instructions)
    return "\n".join(sections).strip()


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    if score < 0:
        return 0.0
    if score > 100:
        return 100.0
    return score


def _normalise_reasons(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        return [cleaned] if cleaned else []
    if isinstance(raw_value, Iterable):
        reasons: list[str] = []
        for entry in raw_value:
            if entry in (None, ""):
                continue
            text = str(entry).strip()
            if text:
                reasons.append(text)
        return reasons
    return []


def _safe_json_parse(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = text[first : last + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
    return {"ranked": []}


def _normalise_rankings(raw_text: str, allowed_ids: set[str]) -> list[dict[str, Any]]:
    if not raw_text:
        return []
    parsed = _safe_json_parse(raw_text)
    ranked = parsed.get("ranked")
    if not isinstance(ranked, list):
        return []
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in ranked:
        if not isinstance(entry, Mapping):
            continue
        result_id_raw = entry.get("id")
        if not isinstance(result_id_raw, str):
            continue
        result_id = result_id_raw.strip()
        if not result_id or result_id not in allowed_ids or result_id in seen:
            continue
        score_value = _coerce_score(entry.get("score"))
        if score_value is None:
            continue
        reasons = _normalise_reasons(entry.get("reasons"))
        normalised.append(
            {
                "id": result_id,
                "score": score_value,
                "reasons": reasons,
            }
        )
        seen.add(result_id)
    normalised.sort(key=lambda item: item["score"], reverse=True)
    return normalised


def _build_metadata(
    meta: Mapping[str, Any] | None,
    control: Mapping[str, Any] | None,
    prompt_version: str,
) -> dict[str, Any]:
    def _lookup(key: str, legacy_key: str | None = None) -> Any:
        if meta and key in meta:
            return meta[key]
        if legacy_key and meta and legacy_key in meta:
            return meta[legacy_key]
        if control and key in control:
            return control[key]
        if legacy_key and control and legacy_key in control:
            return control[legacy_key]
        return None

    tenant_id = _lookup("tenant_id", "tenant")
    case_id = _lookup("case_id", "case")
    trace_id = _lookup("trace_id") or DEFAULT_TRACE_ID

    metadata: dict[str, Any] = {
        "tenant_id": tenant_id,
        "tenant": tenant_id,
        "case_id": case_id,
        "case": case_id,
        "trace_id": trace_id,
        "prompt_version": prompt_version,
    }
    key_alias = _lookup("key_alias")
    if key_alias:
        metadata["key_alias"] = key_alias
    ledger_logger = _lookup("ledger_logger")
    if ledger_logger:
        metadata["ledger_logger"] = ledger_logger
    return metadata


def _coerce_temperature(value: Any) -> str:
    if value is None:
        return f"{DEFAULT_TEMPERATURE}"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"{DEFAULT_TEMPERATURE}"
    if numeric < 0:
        numeric = 0.0
    return f"{numeric}"


def _should_retry_with_default(exc: LlmClientError) -> bool:
    """Return True when the LLM error indicates an invalid model selection."""

    if getattr(exc, "status", None) != 400:
        return False
    message = str(exc) or ""
    return "invalid model" in message.lower()


def _temperature_for_label(label: str, base_value: str) -> str | None:
    """Return a temperature compatible with the given model label."""

    if "gpt-5" in label:
        return None
    return base_value


def _prompt_metrics(payload: ScoreResultsData, prompt: str) -> dict[str, Any]:
    snippet_lengths = [len(item.snippet) for item in payload.results]
    title_lengths = [len(item.title) for item in payload.results]
    return {
        "query_chars": len(payload.query),
        "result_count": len(payload.results),
        "criteria_count": len(payload.criteria or []),
        "prompt_chars": len(prompt),
        "total_snippet_chars": sum(snippet_lengths),
        "max_snippet_chars": max(snippet_lengths or [0]),
        "total_title_chars": sum(title_lengths),
    }


def run_score_results(
    control: Mapping[str, Any] | None,
    data: Mapping[str, Any] | ScoreResultsData,
    *,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank search results via LiteLLM and return deterministic scores."""

    payload = (
        data
        if isinstance(data, ScoreResultsData)
        else ScoreResultsData.model_validate(data)
    )
    prompt_version = (
        (control or {}).get("prompt_version")
        or (meta.get("prompt_version") if meta else None)
        or DEFAULT_PROMPT_VERSION
    )
    metadata = _build_metadata(meta, control, prompt_version)
    model_label = str((control or {}).get("model_preset") or DEFAULT_MODEL_LABEL)
    prompt = _build_prompt(payload)
    allowed_ids = {item.id for item in payload.results}
    logger.info(
        "score_results.prompt_metrics",
        extra=_prompt_metrics(payload, prompt),
    )

    temperature_value = _coerce_temperature((control or {}).get("temperature"))
    labels_to_try: list[str] = []
    for candidate in (model_label, DEFAULT_MODEL_LABEL, "default"):
        candidate_text = str(candidate).strip()
        if not candidate_text:
            continue
        if candidate_text not in labels_to_try:
            labels_to_try.append(candidate_text)

    response: Mapping[str, Any] | None = None
    chosen_label: str | None = None
    last_error: Exception | None = None
    for index, candidate_label in enumerate(labels_to_try):
        is_last = index == len(labels_to_try) - 1
        candidate_temperature = _temperature_for_label(
            candidate_label, temperature_value
        )
        temp_context = (
            _temporary_env_var("LITELLM_TEMPERATURE", candidate_temperature)
            if candidate_temperature is not None
            else nullcontext()
        )
        with temp_context:
            try:
                response = llm_client.call(candidate_label, prompt, metadata)
            except ValueError:
                logger.warning(
                    "score_results.invalid_model_preset",
                    extra={"model_preset": candidate_label},
                )
                if is_last:
                    raise
                continue
            except LlmClientError as exc:
                logger.warning(
                    "score_results.llm_error",
                    extra={
                        "model_preset": candidate_label,
                        "error": str(exc),
                    },
                )
                if not is_last and _should_retry_with_default(exc):
                    logger.warning(
                        "score_results.invalid_model_runtime",
                        extra={
                            "model_preset": candidate_label,
                            "error": str(exc),
                        },
                    )
                    last_error = exc
                    continue
                raise
            else:
                chosen_label = candidate_label
                break
    else:
        if last_error:
            raise last_error
        raise RuntimeError("score_results: unable to obtain rerank response")
    assert response is not None
    assert chosen_label is not None

    ranked = _normalise_rankings(response.get("text") or "", allowed_ids)
    top_k = ranked[: payload.k]

    latency_ms = response.get("latency_ms")
    latency_s = (
        round(latency_ms / 1000.0, 4) if isinstance(latency_ms, (int, float)) else None
    )

    result = {
        "ranked": ranked,
        "top_k": top_k,
        "usage": response.get("usage"),
        "latency_s": latency_s,
        "model": response.get("model"),
    }
    logger.info(
        "score_results.response_metrics",
        extra={
            "model": result["model"],
            "latency_ms": response.get("latency_ms"),
            "text_chars": len(response.get("text") or ""),
            "ranked_count": len(ranked),
        },
    )
    if not ranked:
        logger.info(
            "score_results.empty_ranking",
            extra={
                "query": payload.query,
                "tenant_id": metadata.get("tenant_id"),
                "trace_id": metadata.get("trace_id"),
            },
        )
    return result
