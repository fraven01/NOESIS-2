from __future__ import annotations

import json
import math
import os
from contextlib import ExitStack, contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError
from common.logging import get_logger

from llm_worker.schemas import ScoreResultsData

logger = get_logger(__name__)

DEFAULT_MODEL_LABEL = "fast"
DEFAULT_PROMPT_VERSION = "score-results.v2"
DEFAULT_TRACE_ID = "score-results"
DEFAULT_TEMPERATURE = 0.3


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


def _build_prompt(
    payload: ScoreResultsData, meta: Mapping[str, Any] | None = None
) -> str:
    scoring_context = (meta or {}).get("scoring_context_payload") or {}
    rag_facets: Mapping[str, Any] = (meta or {}).get("rag_facets") or {}
    rag_gaps: Sequence[str] = (meta or {}).get("rag_gap_dimensions") or []
    rag_key_points: Sequence[str] = (meta or {}).get("rag_key_points") or []
    rag_documents: Sequence[Mapping[str, Any]] = (meta or {}).get("rag_documents") or []

    sections: list[str] = [
        "Rolle: Du bist fachkundige:r Evaluator:in für IT- und Rechtsdokumentation.",
        "Auftrag: Bewerte die Kandidaten relativ zur Frage, dem Zweck und dem vorhandenen RAG-Bestand.",
        "Kalibrierung der Scores: 85-100 = exzellente, offiziell belastbare Quelle; 70-84 = solide Quellen mit guter Deckung; 40-69 = nur teilweise relevant; <40 = ablehnen.",
        "Berücksichtige Relevanz, Deckung der Rechtsfrage, Aktualität (Version/Datum, Evergreen-Ausnahmen), Autorität und ob der Treffer Lücken im RAG schließt.",
        "Unsichere Datums- oder Versionsangaben sollen zu risk_flags wie \"uncertain_version\" führen.",
        "",
        f"Query: {payload.query}",
    ]

    if scoring_context:
        sections.append("Scoring-Kontext:")
        question = scoring_context.get("question")
        if question:
            sections.append(f"- Frage: {question}")
        purpose = scoring_context.get("purpose")
        if purpose:
            sections.append(f"- Zweck: {purpose}")
        jurisdiction = scoring_context.get("jurisdiction")
        if jurisdiction:
            sections.append(f"- Jurisdiktion: {jurisdiction}")
        output_target = scoring_context.get("output_target")
        if output_target:
            sections.append(f"- Output-Format: {output_target}")
        preferred_sources = scoring_context.get("preferred_sources") or []
        if preferred_sources:
            sections.append("- Bevorzugte Quellen: " + ", ".join(preferred_sources))
        disallowed_sources = scoring_context.get("disallowed_sources") or []
        if disallowed_sources:
            sections.append("- Ausgeschlossene Quellen: " + ", ".join(disallowed_sources))
        version_target = scoring_context.get("version_target")
        if version_target:
            sections.append(f"- Ziel-Version: {version_target}")
        freshness_mode = scoring_context.get("freshness_mode")
        if freshness_mode:
            sections.append(f"- Freshness-Modus: {freshness_mode}")
        sections.append("")

    if rag_facets or rag_key_points or rag_documents:
        sections.append("Aktueller RAG-Bestand:")
        if rag_documents:
            sections.append("- Dokumente:")
            for doc in rag_documents[:3]:
                title = doc.get("title")
                url = doc.get("url")
                if title:
                    sections.append(f"  • {title}" + (f" ({url})" if url else ""))
        if rag_key_points:
            sections.append("- Key Points:")
            for point in rag_key_points[:5]:
                sections.append(f"  • {point}")
        if rag_facets:
            sections.append("- Facet-Abdeckung (0 = fehlend, 1 = vollständig):")
            for facet, score in rag_facets.items():
                gap_marker = " (Gap)" if facet in rag_gaps else ""
                sections.append(f"  • {facet}: {score:.2f}{gap_marker}")
        if rag_gaps:
            sections.append("- Fehlende Facetten mit Priorität: " + ", ".join(rag_gaps))
        sections.append("")

    if payload.criteria:
        sections.append("Zusätzliche Bewertungskriterien:")
        for criterion in payload.criteria:
            sections.append(f"- {criterion}")
        sections.append("")

    sections.append("Kandidaten:")
    for idx, item in enumerate(payload.results, start=1):
        sections.append(f"[{idx}] id={item.id}")
        if item.title:
            sections.append(f"Titel: {item.title}")
        if item.snippet:
            sections.append(f"Snippet: {item.snippet}")
        if item.url:
            sections.append(f"URL: {item.url}")
        if item.detected_date:
            sections.append(f"Datum: {item.detected_date.isoformat()}")
        if item.version_hint:
            sections.append(f"Version: {item.version_hint}")
        if item.domain_type:
            sections.append(f"Domain-Typ: {item.domain_type}")
        if item.trust_hint:
            sections.append(f"Trust-Hint: {item.trust_hint}")
        sections.append("")

    sections.extend(
        [
            "Denke Schritt für Schritt und berücksichtige Evergreen-Ausnahmen, bevor du dein Ergebnis formulierst.",
            "Gib ausschließlich JSON im folgenden Schema zurück (keine Erklärungen außerhalb des JSON):",
            '{"evaluations":[{"candidate_id":"string","score":0,"reason":"string","gap_tags":["string"],"risk_flags":["string"],"facet_coverage":{"facet_name":0.0}}]}',
            "",
            "Gutes Beispiel:",
            '{"evaluations":[{"candidate_id":"doc-1","score":86,"reason":"Deckt Mitbestimmungs-Pflichten und ist offizielle Hersteller-Doku (Version 2024).","gap_tags":["TECHNICAL"],"risk_flags":["uncertain_version"],"facet_coverage":{"TECHNICAL":0.8,"PROCEDURAL":0.4}}]}',
            "Schlechtes Beispiel (vermeiden – falsches Schema, fehlende Begründung):",
            '{"ranked":[{"id":"doc-1","score":100}]}'
        ]
    )

    sections.append("Antworte nur mit dem JSON-Objekt und sortiere die Evaluations nach Score absteigend.")
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
    return {"evaluations": []}


def _normalise_rankings(raw_text: str, allowed_ids: set[str]) -> list[dict[str, Any]]:
    if not raw_text:
        return []
    parsed = _safe_json_parse(raw_text)
    ranked = parsed.get("evaluations")
    if not isinstance(ranked, list):
        return []
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in ranked:
        if not isinstance(entry, Mapping):
            continue
        result_id_raw = entry.get("candidate_id") or entry.get("id")
        if not isinstance(result_id_raw, str):
            continue
        result_id = result_id_raw.strip()
        if not result_id or result_id not in allowed_ids or result_id in seen:
            continue
        score_value = _coerce_score(entry.get("score"))
        if score_value is None:
            continue
        reason_raw = entry.get("reason")
        reason_text = ""
        if isinstance(reason_raw, str):
            reason_text = reason_raw.strip()
        elif isinstance(reason_raw, Iterable):
            reason_text = " ".join(str(part).strip() for part in reason_raw if part)
        reasons = [reason_text] if reason_text else []
        gap_tags = _normalise_reasons(entry.get("gap_tags"))
        risk_flags = _normalise_reasons(entry.get("risk_flags"))
        facet_payload = entry.get("facet_coverage")
        facet_coverage: dict[str, float] = {}
        if isinstance(facet_payload, Mapping):
            for key, raw_score in facet_payload.items():
                if not isinstance(key, str):
                    continue
                cleaned_key = key.strip()
                if not cleaned_key:
                    continue
                try:
                    value = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value):
                    continue
                if value < 0:
                    value = 0.0
                if value > 1:
                    value = 1.0
                facet_coverage[cleaned_key] = value
        normalised.append(
            {
                "candidate_id": result_id,
                "score": score_value,
                "reason": reasons[0] if reasons else "",
                "gap_tags": gap_tags,
                "risk_flags": risk_flags,
                "facet_coverage": facet_coverage,
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


def _coerce_max_tokens(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    if numeric > 8000:
        numeric = 8000
    return numeric


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
    prompt = _build_prompt(payload, meta)
    allowed_ids = {item.id for item in payload.results}
    logger.info(
        "score_results.prompt_metrics",
        extra=_prompt_metrics(payload, prompt),
    )

    temperature_value = _coerce_temperature((control or {}).get("temperature"))
    max_tokens_value = _coerce_max_tokens((control or {}).get("max_tokens"))
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
        with ExitStack() as stack:
            if max_tokens_value is not None:
                stack.enter_context(
                    _temporary_env_var("LITELLM_MAX_TOKENS", str(max_tokens_value))
                )
            if candidate_temperature is not None:
                stack.enter_context(
                    _temporary_env_var("LITELLM_TEMPERATURE", candidate_temperature)
                )
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
        "evaluations": ranked,
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
