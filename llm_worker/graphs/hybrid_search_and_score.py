"""Hybrid web and RAG reranking graph for the LLM worker."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from uuid import UUID, uuid5

from django.conf import settings
from django.core.cache import cache

from ai_core.infra.observability import update_observation
from ai_core.nodes import retrieve
from ai_core.tool_contracts import ToolContext
from llm_worker.graphs.score_results import run_score_results
from llm_worker.schemas import (
    CoverageDimension,
    FreshnessMode,
    HybridResult,
    LLMScoredItem,
    RAGCoverageSummary,
    RecommendedIngestItem,
    ScoringContext,
    SearchCandidate,
    ScoreResultsData,
)
from llm_worker.domain_policies import DomainPolicy, DomainPolicyAction, get_domain_policy
from llm_worker.utils.normalisation import coerce_enum, ensure_aware_utc


logger = logging.getLogger(__name__)


RAG_CACHE_TTL_S = 300
LLM_CACHE_TTL_S = 3600
MMR_LAMBDA = 0.7
MMR_LIMIT = 20
RRF_K = 90
MIN_DIVERSITY_BUCKETS = 3
MAX_KEY_POINTS = 5
MIN_KEY_POINTS = 3
DEFAULT_FRESHNESS_DAYS = 365
SOFTWARE_FRESHNESS_DAYS = 1095
MAX_REASON_LENGTH = 280
FRESHNESS_PENALTY_PER_MONTH = 2.0
DOMAIN_REDUNDANCY_PENALTY = 0.85
POLICY_PRIORITY_SCALE = 0.1


_FACET_KEYWORDS: dict[CoverageDimension, tuple[str, ...]] = {
    CoverageDimension.LEGAL: (
        "legal",
        "gesetz",
        "policy",
        "compliance",
        "regulation",
        "gdpr",
    ),
    CoverageDimension.TECHNICAL: (
        "technical",
        "system",
        "architecture",
        "api",
        "protocol",
    ),
    CoverageDimension.PROCEDURAL: (
        "process",
        "procedure",
        "workflow",
        "steps",
    ),
    CoverageDimension.DATA_CATEGORIES: (
        "data category",
        "pii",
        "personal data",
        "classification",
    ),
    CoverageDimension.MONITORING_SURVEILLANCE: (
        "monitor",
        "surveillance",
        "observe",
        "watch",
    ),
    CoverageDimension.LOGGING_AUDIT: (
        "audit",
        "logging",
        "log",
    ),
    CoverageDimension.ANALYTICS_REPORTING: (
        "analytics",
        "report",
        "metric",
        "dashboard",
    ),
    CoverageDimension.ACCESS_PRIVACY_SECURITY: (
        "privacy",
        "security",
        "access",
        "permission",
        "control",
    ),
    CoverageDimension.API_INTEGRATION: (
        "api",
        "integration",
        "endpoint",
        "sdk",
    ),
}

_URL_SPLIT_PATTERN = re.compile(r"https?://")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WORD_PATTERN = re.compile(r"[\w\u00C0-\u024F]+", re.UNICODE)
_CUSTOM_FACET_KEYS = {"custom_facets", "facet_scores", "facets"}


def _ensure_mutable(mapping: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    if isinstance(mapping, MutableMapping):
        return mapping
    return dict(mapping or {})


def _load_scoring_context(meta: Mapping[str, Any]) -> ScoringContext | None:
    candidate = meta.get("scoring_context")
    if not candidate:
        return None
    if isinstance(candidate, ScoringContext):
        return candidate
    try:
        if isinstance(candidate, str):
            return ScoringContext.model_validate_json(candidate)
        return ScoringContext.model_validate(candidate)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("hybrid.scoring_context_invalid", extra={"error": str(exc)})
        return None


def _canonicalise_url(url: str | None) -> str | None:
    if not url:
        return None
    text = str(url).strip()
    if not text:
        return None
    if not _URL_SPLIT_PATTERN.match(text):
        text = f"https://{text}"
    parsed = urlsplit(text)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if not netloc:
        return None
    hostname = parsed.hostname or netloc
    port = parsed.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = hostname
    elif port:
        netloc = f"{hostname}:{port}"
    path = quote(unquote(parsed.path or "/"), safe="/%:@")
    query = "&".join(sorted(filter(None, (parsed.query or "").split("&"))))
    return urlunsplit((scheme, netloc, path, query, ""))


def _extract_host(url: str | None) -> str | None:
    canonical = _canonicalise_url(url)
    if not canonical:
        return None
    return urlsplit(canonical).hostname

def _build_domain_policy(
    scoring_context: ScoringContext | None,
    *,
    tenant_id: str | None,
) -> DomainPolicy:
    policy = get_domain_policy(tenant_id).clone()
    if not scoring_context:
        return policy
    for source in scoring_context.preferred_sources:
        host = _extract_host(source)
        if host:
            policy.add_host(
                host,
                DomainPolicyAction.BOOST,
                priority=85,
                source="context.preferred",
            )
    for source in scoring_context.disallowed_sources:
        host = _extract_host(source)
        if host:
            policy.add_host(
                host,
                DomainPolicyAction.REJECT,
                priority=100,
                source="context.disallowed",
            )
    return policy


def _freshness_cutoff(scoring_context: ScoringContext | None) -> datetime | None:
    mode = getattr(scoring_context, "freshness_mode", FreshnessMode.STANDARD)
    if mode == FreshnessMode.LAW_EVERGREEN:
        return None
    if mode == FreshnessMode.SOFTWARE_DOCS_STRICT:
        days = SOFTWARE_FRESHNESS_DAYS
    else:
        days = DEFAULT_FRESHNESS_DAYS
    return datetime.now(timezone.utc) - timedelta(days=days)


def _hash_values(*chunks: str) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.encode("utf-8", "ignore"))
    return digest.hexdigest()


def _tokenise(text: str) -> set[str]:
    if not text:
        return set()
    return {match.group(0).lower() for match in _WORD_PATTERN.finditer(text)}


def _infer_domain_traits(url: str | None) -> tuple[str | None, str | None, str | None]:
    if not url:
        return None, None, None
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return None, None, None
    domain_type: str
    trust_hint: str
    if host.endswith(".gov") or ".gov." in host:
        domain_type = "government"
        trust_hint = "high"
    elif host.endswith(".edu") or ".ac." in host:
        domain_type = "academic"
        trust_hint = "high"
    elif host.endswith(".org"):
        domain_type = "nonprofit"
        trust_hint = "medium"
    elif host.startswith("intranet") or host.endswith(".internal") or host.endswith(
        ".corp"
    ):
        domain_type = "internal"
        trust_hint = "medium"
    else:
        domain_type = "commercial"
        trust_hint = "medium"
    return host, domain_type, trust_hint


def _mmr_light(
    candidates: Sequence[MutableMapping[str, Any]],
    *,
    lam: float = MMR_LAMBDA,
    limit: int = MMR_LIMIT,
    domain_penalty: float = DOMAIN_REDUNDANCY_PENALTY,
) -> tuple[list[MutableMapping[str, Any]], dict[str, Any]]:
    if len(candidates) <= limit:
        selected = list(candidates)
        debug = [
            {
                "id": candidate.get("id"),
                "mmr_score": float(candidate.get("base_score", 0.0)),
                "max_overlap": 0.0,
                "domain_penalty": False,
            }
            for candidate in selected
        ]
        return selected, {"selected": debug}
    remaining = list(candidates)
    selected: list[MutableMapping[str, Any]] = []
    debug_selected: list[dict[str, Any]] = []
    token_cache: dict[str, set[str]] = {}

    def _tokens(candidate: Mapping[str, Any]) -> set[str]:
        candidate_id = candidate.get("id")
        if candidate_id in token_cache:
            return token_cache[candidate_id]
        snippet = str(candidate.get("snippet") or "")
        tokens = _tokenise(snippet)
        token_cache[candidate_id] = tokens
        return tokens

    while remaining and len(selected) < limit:
        best_candidate: MutableMapping[str, Any] | None = None
        best_score = -math.inf
        best_overlap = 0.0
        best_domain_penalty = False
        for candidate in list(remaining):
            base_score = float(candidate.get("base_score", 0.0))
            candidate_tokens = _tokens(candidate)
            candidate_host = candidate.get("host")
            if selected:
                max_similarity = 0.0
                applied_penalty = False
                for other in selected:
                    similarity = _jaccard_similarity(candidate_tokens, _tokens(other))
                    if candidate_host and candidate_host == other.get("host"):
                        similarity = max(similarity, domain_penalty)
                        applied_penalty = applied_penalty or similarity >= domain_penalty
                    if similarity > max_similarity:
                        max_similarity = similarity
                penalty_applied = applied_penalty
            else:
                max_similarity = 0.0
                penalty_applied = False
            score = lam * base_score - (1.0 - lam) * max_similarity
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_overlap = max_similarity
                best_domain_penalty = penalty_applied
        assert best_candidate is not None
        selected.append(best_candidate)
        debug_selected.append(
            {
                "id": best_candidate.get("id"),
                "mmr_score": best_score,
                "max_overlap": best_overlap,
                "domain_penalty": best_domain_penalty,
            }
        )
        remaining.remove(best_candidate)
    return selected, {"selected": debug_selected}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def _extract_key_points(text: str) -> list[str]:
    sentences = [candidate.strip() for candidate in re.split(r"(?<=[.!?])\s+", text) if candidate]
    key_points = [sentence for sentence in sentences if len(sentence) > 10]
    if len(key_points) < MIN_KEY_POINTS:
        chunks = text.splitlines()
        for chunk in chunks:
            candidate = chunk.strip()
            if candidate and candidate not in key_points:
                key_points.append(candidate)
                if len(key_points) >= MIN_KEY_POINTS:
                    break
    key_points = key_points[:MAX_KEY_POINTS]
    while len(key_points) < MIN_KEY_POINTS:
        key_points.append(key_points[-1] if key_points else text[:50])
    return key_points


def _infer_facets(text: str) -> dict[CoverageDimension, float]:
    lowered = text.lower()
    facets: dict[CoverageDimension, float] = {}
    for dimension, keywords in _FACET_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits:
            facets[dimension] = min(1.0, hits / 3.0)
    return facets


def _infer_custom_facets(meta: Mapping[str, Any] | None) -> dict[str, float]:
    if not meta:
        return {}
    for key in _CUSTOM_FACET_KEYS:
        if key in meta:
            raw = meta[key]
            if isinstance(raw, Mapping):
                result: dict[str, float] = {}
                for facet, score in raw.items():
                    try:
                        numeric = float(score)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(numeric):
                        continue
                    result[str(facet)] = max(0.0, min(1.0, numeric))
                if result:
                    return result
    return {}


def _ensure_uuid(text: str) -> UUID:
    try:
        return UUID(text)
    except Exception:
        return uuid5(UUID(int=0), text)


def _aggregate_rag_facets(summaries: Sequence[RAGCoverageSummary]) -> dict[CoverageDimension, float]:
    totals: dict[CoverageDimension, list[float]] = defaultdict(list)
    for summary in summaries:
        for dimension_raw, score in summary.coverage_facets.items():
            dimension = coerce_enum(dimension_raw, CoverageDimension)
            if dimension is None:
                continue
            totals[dimension].append(score)
    averages: dict[CoverageDimension, float] = {}
    for dimension, scores in totals.items():
        if scores:
            averages[dimension] = sum(scores) / len(scores)
    return averages


def _collect_rag_key_points(
    summaries: Sequence[RAGCoverageSummary], limit: int = MAX_KEY_POINTS
) -> list[str]:
    points: list[str] = []
    for summary in summaries:
        for point in summary.key_points:
            candidate = point.strip()
            if not candidate:
                continue
            if candidate not in points:
                points.append(candidate)
            if len(points) >= limit:
                return points[:limit]
    return points[:limit]


def _dimensions_to_tags(dimensions: Iterable[CoverageDimension]) -> list[str]:
    return [dimension.value for dimension in dimensions]


def _gap_dimensions(
    rag_facets: Mapping[CoverageDimension, float], threshold: float = 0.4
) -> set[CoverageDimension]:
    gaps: set[CoverageDimension] = set()
    for dimension in CoverageDimension:
        if rag_facets.get(dimension, 0.0) < threshold:
            gaps.add(dimension)
    return gaps


def _best_reason(reasons: Sequence[str] | str | None) -> str:
    if isinstance(reasons, str):
        reasons = [reasons]
    if not reasons:
        return "Re-ranked by LLM"
    for reason in reasons:
        cleaned = reason.strip()
        if cleaned:
            if len(cleaned) > MAX_REASON_LENGTH:
                return cleaned[: MAX_REASON_LENGTH - 1].rstrip() + "…"
            return cleaned
    return "Re-ranked by LLM"


@dataclass
class HybridSearchAndScoreGraph:
    """Orchestrate hybrid web/RAG reranking for multi-tenant workloads."""

    rag_top_k: int = 12
    rerank_top_k: int = 5
    mmr_lambda: float = MMR_LAMBDA
    rag_cache_ttl: int = RAG_CACHE_TTL_S
    llm_cache_ttl: int = LLM_CACHE_TTL_S
    rrf_k: int = RRF_K
    default_min_diversity_buckets: int = MIN_DIVERSITY_BUCKETS

    _last_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def run(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        working_state = _ensure_mutable(state)
        working_meta = _ensure_mutable(meta)
        scoring_context = _load_scoring_context(working_meta)
        tenant_id = str(working_meta.get("tenant_id") or "").strip() or None
        domain_policy = _build_domain_policy(scoring_context, tenant_id=tenant_id)
        debug_info: dict[str, Any] = {}
        rrf_k_override = working_meta.get("rrf_k")
        try:
            rrf_k_value = int(rrf_k_override)
            if rrf_k_value <= 0:
                raise ValueError
        except Exception:
            rrf_k_value = self.rrf_k
        debug_info.setdefault("settings", {})["rrf_k"] = rrf_k_value

        query = str(working_state.get("query") or working_meta.get("query") or "").strip()
        if not query:
            raise ValueError("query is required for hybrid_search_and_score")

        normalised_candidates = self._normalise_candidates(working_state, scoring_context)
        if not normalised_candidates:
            raise ValueError("at least one candidate is required for reranking")

        rag_summaries, rag_flags = self._retrieve_rag_context(
            query=query,
            state=working_state,
            meta=working_meta,
            scoring_context=scoring_context,
        )
        rag_facets = _aggregate_rag_facets(rag_summaries)
        debug_info["rag_facets"] = {dimension.value: score for dimension, score in rag_facets.items()}

        filtered_candidates, pre_filter_debug = self._pre_filter_candidates(
            normalised_candidates,
            scoring_context=scoring_context,
            domain_policy=domain_policy,
        )
        debug_info["pre_filter"] = pre_filter_debug
        normalise_debug = working_state.get("_debug", {}).get("normalise")
        if normalise_debug:
            debug_info["normalise"] = normalise_debug

        rerank_payload, rerank_flags, llm_debug = self._run_llm_rerank(
            query=query,
            candidates=filtered_candidates,
            meta=working_meta,
            scoring_context=scoring_context,
            rag_facets=rag_facets,
            rag_summaries=rag_summaries,
        )
        debug_info["llm"] = llm_debug

        hybrid_result, fusion_debug = self._deduplicate_and_select(
            candidates=normalised_candidates,
            rerank_payload=rerank_payload,
            rag_summaries=rag_summaries,
            rag_facets=rag_facets,
            domain_policy=domain_policy,
            scoring_context=scoring_context,
            rrf_k=rrf_k_value,
        )
        debug_info["fusion"] = fusion_debug

        flags = {**rag_flags, **rerank_flags}
        flags["debug"] = debug_info
        self._last_debug = debug_info
        result_payload = {
            "result": hybrid_result.model_dump(mode="json"),
            "rag_summaries": [summary.model_dump(mode="json") for summary in rag_summaries],
            "flags": flags,
        }

        working_state["hybrid_result"] = result_payload
        working_state["rag_summaries"] = result_payload["rag_summaries"]
        working_state["flags"] = flags

        return working_state, result_payload

    # Normalise -----------------------------------------------------------------

    def _normalise_candidates(
        self,
        state: MutableMapping[str, Any],
        scoring_context: ScoringContext | None,
    ) -> list[MutableMapping[str, Any]]:
        raw_candidates = state.get("candidates")
        if not isinstance(raw_candidates, Sequence):
            raw_candidates = state.get("results")
        if not isinstance(raw_candidates, Sequence):
            raw_candidates = []

        normalised: list[MutableMapping[str, Any]] = []
        seen_urls: dict[str, str] = {}
        normalise_debug: list[dict[str, Any]] = []

        for index, entry in enumerate(raw_candidates):
            if isinstance(entry, SearchCandidate):
                candidate_data = entry.model_dump(mode="json")
            elif isinstance(entry, Mapping):
                candidate_data = dict(entry)
            else:
                continue

            url = candidate_data.get("url") or candidate_data.get("link")
            canonical_url = _canonicalise_url(url)
            host, domain_type, trust_hint = _infer_domain_traits(canonical_url)
            candidate_id = str(candidate_data.get("id") or "").strip()
            if not candidate_id:
                candidate_id = _hash_values(canonical_url or "candidate", str(index))[:16]
            title = str(candidate_data.get("title") or "").strip()
            snippet = str(candidate_data.get("snippet") or "").strip()
            detected_date = ensure_aware_utc(candidate_data.get("detected_date"))
            version_hint = candidate_data.get("version_hint")
            is_pdf = bool(candidate_data.get("is_pdf") or (canonical_url or "").endswith(".pdf"))
            base_score = float(candidate_data.get("score") or (100 - index))
            duplicate_of = None
            is_duplicate = False
            if canonical_url:
                if canonical_url in seen_urls:
                    duplicate_of = seen_urls[canonical_url]
                    is_duplicate = True
                else:
                    seen_urls[canonical_url] = candidate_id

            candidate_payload: MutableMapping[str, Any] = {
                "id": candidate_id,
                "url": canonical_url,
                "title": title,
                "snippet": snippet,
                "is_pdf": is_pdf,
                "detected_date": detected_date,
                "version_hint": version_hint,
                "domain_type": domain_type,
                "trust_hint": trust_hint,
                "host": host,
                "duplicate_of": duplicate_of,
                "is_duplicate": is_duplicate,
                "base_score": base_score,
            }
            normalised.append(candidate_payload)
            normalise_debug.append(
                {
                    "candidate_id": candidate_id,
                    "url_raw": url,
                    "url_canonical": canonical_url,
                    "duplicate_of": duplicate_of,
                }
            )

        normalised.sort(key=lambda item: float(item.get("base_score", 0.0)), reverse=True)
        state["normalized_candidates"] = normalised
        state.setdefault("_debug", {})["normalise"] = {"urls": normalise_debug}
        return normalised

    # RAG context ----------------------------------------------------------------

    def _retrieve_rag_context(
        self,
        *,
        query: str,
        state: MutableMapping[str, Any],
        meta: MutableMapping[str, Any],
        scoring_context: ScoringContext | None,
    ) -> tuple[list[RAGCoverageSummary], dict[str, bool]]:
        tenant_id = str(meta.get("tenant_id") or "").strip()
        case_id = str(meta.get("case_id") or "").strip()
        trace_id = str(meta.get("trace_id") or "").strip()
        if not tenant_id or not case_id:
            return [], {"rag_unavailable": True}

        collection_scope = None
        if scoring_context and scoring_context.collection_scope:
            collection_scope = scoring_context.collection_scope

        query_hash = _hash_values(query)
        cache_key = f"hybrid:rag:{tenant_id}:{collection_scope}:{query_hash}"
        cached = cache.get(cache_key)
        if cached:
            try:
                summaries = [RAGCoverageSummary.model_validate(item) for item in cached]
                update_observation(
                    metadata={
                        "hybrid.rag_cache_hit": True,
                        "hybrid.rag_cache_key": cache_key,
                    }
                )
                return summaries, {"rag_cache_hit": True, "rag_unavailable": False}
            except Exception:
                cache.delete(cache_key)

        try:
            retrieve_state = {
                "query": query,
                "collection_id": collection_scope,
                "hybrid": {
                    "alpha": 0.7,
                    "min_sim": 0.2,
                    "top_k": self.rag_top_k,
                    "vec_limit": max(self.rag_top_k * 2, 20),
                    "lex_limit": max(self.rag_top_k * 2, 20),
                    "max_candidates": max(self.rag_top_k * 4, 40),
                    "diversify_strength": 0.3,
                },
            }
            retrieve_params = retrieve.RetrieveInput.from_state(retrieve_state)
            tool_context = ToolContext(
                tenant_id=tenant_id,
                case_id=case_id,
                trace_id=trace_id or None,
                tenant_schema=meta.get("tenant_schema"),
                visibility_override_allowed=bool(
                    meta.get("visibility_override_allowed", False)
                ),
                metadata={"scoring_context": scoring_context.model_dump() if scoring_context else {}},
            )
            retrieve_output = retrieve.run(tool_context, retrieve_params)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "hybrid.rag_failure",
                extra={"tenant_id": tenant_id, "case_id": case_id, "error": str(exc)},
            )
            return [], {"rag_unavailable": True}

        summaries = self._summarise_matches(retrieve_output.matches)
        if summaries:
            payload = [summary.model_dump(mode="json") for summary in summaries]
            try:
                cache.set(cache_key, payload, timeout=self.rag_cache_ttl)
            except Exception:  # pragma: no cover - defensive
                logger.debug("hybrid.rag_cache_unavailable")
        update_observation(
            metadata={
                "hybrid.rag_cache_hit": False,
                "hybrid.rag_cache_key": cache_key,
            }
        )

        return summaries, {"rag_cache_hit": False, "rag_unavailable": False}

    def _summarise_matches(
        self, matches: Sequence[Mapping[str, Any]]
    ) -> list[RAGCoverageSummary]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for match in matches:
            meta = match.get("meta")
            if isinstance(meta, Mapping):
                doc_id = meta.get("document_id") or meta.get("id")
            else:
                doc_id = match.get("id")
            key = str(doc_id or match.get("hash") or match.get("source") or "unknown")
            grouped[key].append(match)

        summaries: list[RAGCoverageSummary] = []
        for key, group in grouped.items():
            combined_text = "\n".join(str(item.get("text") or "") for item in group)
            if not combined_text.strip():
                continue
            meta = next(
                (item.get("meta") for item in group if isinstance(item.get("meta"), Mapping)),
                {},
            )
            title = str(meta.get("title") or group[0].get("citation") or "RAG Document")
            url = meta.get("source_url") or group[0].get("source")
            key_points = _extract_key_points(combined_text)
            coverage_facets = _infer_facets(combined_text)
            custom_facets = _infer_custom_facets(meta)
            ingested_at = _parse_datetime(meta.get("ingested_at") or meta.get("updated_at"))
            if ingested_at is None:
                ingested_at = datetime.now(timezone.utc)
            try:
                document_uuid = _ensure_uuid(str(meta.get("document_id") or key))
            except Exception:
                document_uuid = uuid5(UUID(int=0), key)
            summaries.append(
                RAGCoverageSummary(
                    document_id=document_uuid,
                    title=title,
                    url=_canonicalise_url(url),
                    key_points=key_points,
                    coverage_facets=coverage_facets,
                    custom_facets=custom_facets,
                    last_ingested_at=ingested_at,
                )
            )
        return summaries

    # Pre-filter -----------------------------------------------------------------

    def _apply_policy_heuristics(
        self,
        candidates: Sequence[MutableMapping[str, Any]],
        *,
        domain_policy: DomainPolicy,
        scoring_context: ScoringContext | None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        now = datetime.now(timezone.utc)
        freshness_mode = getattr(scoring_context, "freshness_mode", FreshnessMode.STANDARD)
        debug_entries: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "")
            base_score = float(candidate.get("base_score", 0.0))
            adjustments: dict[str, Any] = {"base_before": base_score}
            host = candidate.get("host")
            decision = domain_policy.evaluate(host)
            if decision:
                adjustments["policy"] = {
                    "action": decision.action.value,
                    "priority": decision.priority,
                    "source": decision.source,
                }
                candidate["policy_decision"] = decision.action.value
                if decision.action is DomainPolicyAction.REJECT:
                    candidate["policy_reject"] = True
                elif decision.action is DomainPolicyAction.BOOST:
                    boost = max(0.0, decision.priority * POLICY_PRIORITY_SCALE)
                    candidate["policy_boost"] = boost
                    candidate["base_score"] = base_score + boost
                    base_score = candidate["base_score"]
            detected = candidate.get("detected_date")
            if (
                isinstance(detected, datetime)
                and detected.tzinfo is not None
                and freshness_mode is not FreshnessMode.LAW_EVERGREEN
            ):
                age_days = max(0, (now - detected).days)
                if age_days > SOFTWARE_FRESHNESS_DAYS:
                    months_over = (age_days - SOFTWARE_FRESHNESS_DAYS) / 30.0
                    penalty = min(25.0, max(0.0, months_over * FRESHNESS_PENALTY_PER_MONTH))
                    if penalty:
                        candidate["freshness_penalty"] = penalty
                        candidate["base_score"] = max(0.0, base_score - penalty)
                        base_score = candidate["base_score"]
                        adjustments["freshness_penalty"] = {
                            "penalty": penalty,
                            "age_days": age_days,
                        }
            adjustments["base_after"] = float(candidate.get("base_score", base_score))
            debug_entries.append(
                {
                    "id": candidate_id,
                    "host": host,
                    **adjustments,
                }
            )
        return debug_entries

    def _pre_filter_candidates(
        self,
        candidates: Sequence[MutableMapping[str, Any]],
        *,
        scoring_context: ScoringContext | None,
        domain_policy: DomainPolicy,
    ) -> tuple[list[MutableMapping[str, Any]], dict[str, Any]]:
        candidates = list(candidates)
        if not candidates:
            return [], {"dropped": [], "mmr": {"selected": []}, "heuristics": []}

        heuristics_debug = self._apply_policy_heuristics(
            candidates,
            domain_policy=domain_policy,
            scoring_context=scoring_context,
        )

        cutoff = _freshness_cutoff(scoring_context)
        filtered: list[MutableMapping[str, Any]] = []
        policy_safe: list[MutableMapping[str, Any]] = []
        dropped: list[dict[str, str]] = []

        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "")
            host = candidate.get("host")
            if candidate.get("policy_reject") or domain_policy.is_blocked(host):
                dropped.append({"id": candidate_id, "reason": "policy_block"})
                continue

            policy_safe.append(candidate)

            if candidate.get("is_duplicate"):
                dropped.append({"id": candidate_id, "reason": "duplicate"})
                continue

            snippet = str(candidate.get("snippet") or "").strip()
            if not snippet:
                dropped.append({"id": candidate_id, "reason": "empty_snippet"})
                continue

            detected = candidate.get("detected_date")
            if cutoff and isinstance(detected, datetime) and detected < cutoff:
                dropped.append({"id": candidate_id, "reason": "stale"})
                continue

            filtered.append(candidate)

        working_candidates: Sequence[MutableMapping[str, Any]]
        if filtered:
            working_candidates = filtered
        else:
            working_candidates = policy_safe

        selected, mmr_debug = _mmr_light(
            working_candidates,
            lam=self.mmr_lambda,
            limit=self.rerank_top_k * 4,
            domain_penalty=DOMAIN_REDUNDANCY_PENALTY,
        )
        return (
            selected,
            {"dropped": dropped, "mmr": mmr_debug, "heuristics": heuristics_debug},
        )

    # LLM rerank -----------------------------------------------------------------

    def _run_llm_rerank(
        self,
        *,
        query: str,
        candidates: Sequence[MutableMapping[str, Any]],
        meta: MutableMapping[str, Any],
        scoring_context: ScoringContext | None,
        rag_facets: Mapping[CoverageDimension, float],
        rag_summaries: Sequence[RAGCoverageSummary],
    ) -> tuple[list[LLMScoredItem], dict[str, bool], dict[str, Any]]:
        if not candidates:
            raise ValueError("candidates required for llm rerank")

        search_candidates: list[SearchCandidate] = []
        for candidate in candidates:
            payload = {
                "id": candidate.get("id"),
                "url": candidate.get("url"),
                "title": candidate.get("title"),
                "snippet": candidate.get("snippet"),
                "is_pdf": candidate.get("is_pdf"),
                "detected_date": candidate.get("detected_date"),
                "version_hint": candidate.get("version_hint"),
                "domain_type": candidate.get("domain_type"),
                "trust_hint": candidate.get("trust_hint"),
            }
            try:
                search_candidates.append(SearchCandidate.model_validate(payload))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("hybrid.candidate_invalid", extra={"error": str(exc)})

        rag_gap_dimensions = _dimensions_to_tags(_gap_dimensions(rag_facets))

        query_hash = _hash_values(query)
        url_hash = _hash_values(*(candidate.id for candidate in search_candidates))
        tenant_id = str(meta.get("tenant_id") or "").strip()
        context_chunks: list[str] = []
        if tenant_id:
            context_chunks.append(tenant_id)
        if scoring_context:
            context_chunks.append(
                json.dumps(
                    scoring_context.model_dump(mode="json"),
                    sort_keys=True,
                )
            )
        if rag_facets:
            context_chunks.append(
                json.dumps(
                    {
                        dimension.value: float(score)
                        for dimension, score in rag_facets.items()
                    },
                    sort_keys=True,
                )
            )
        if rag_gap_dimensions:
            context_chunks.append(json.dumps(sorted(rag_gap_dimensions)))
        if rag_summaries:
            context_chunks.append(
                json.dumps(
                    [
                        {
                            "title": summary.title,
                            "url": summary.url,
                            "document_id": str(summary.document_id),
                        }
                        for summary in rag_summaries
                    ],
                    sort_keys=True,
                )
            )
        context_hash = (
            _hash_values(*context_chunks) if context_chunks else "baseline"
        )
        cache_key = f"hybrid:llm:{query_hash}:{url_hash}:{context_hash}"
        cached = cache.get(cache_key)
        debug_payload: dict[str, Any] = {
            "cache_key": cache_key,
            "candidate_count": len(search_candidates),
        }
        if cached:
            try:
                ranked = [LLMScoredItem.model_validate(item) for item in cached]
                update_observation(
                    metadata={
                        "hybrid.llm_cache_hit": True,
                        "hybrid.llm_cache_key": cache_key,
                        "hybrid.llm_cached_items": len(ranked),
                    }
                )
                return ranked, {
                    "llm_cache_hit": True,
                    "llm_timeout": False,
                    "llm_failure": False,
                }, {
                    **debug_payload,
                    "cache_hit": True,
                    "fallback": None,
                    "llm_items": len(ranked),
                }
            except Exception:
                cache.delete(cache_key)

        payload = ScoreResultsData(
            query=query,
            results=search_candidates,
            k=max(self.rerank_top_k, 5),
            criteria=self._build_criteria(scoring_context),
        )

        control_meta = {
            "prompt_version": meta.get("prompt_version") or "hybrid-score.v1",
            "model_preset": meta.get("model_preset") or "fast",
            "max_tokens": meta.get("max_tokens") or 2000,
        }

        llm_meta: dict[str, Any] = dict(meta)
        if scoring_context:
            llm_meta["scoring_context_payload"] = scoring_context.model_dump(mode="json")
        if rag_facets:
            llm_meta["rag_facets"] = {
                dimension.value: float(score)
                for dimension, score in rag_facets.items()
            }
        if rag_gap_dimensions:
            llm_meta["rag_gap_dimensions"] = rag_gap_dimensions
        if rag_summaries:
            llm_meta["rag_key_points"] = _collect_rag_key_points(rag_summaries)
            llm_meta["rag_documents"] = [
                {
                    "title": summary.title,
                    "url": summary.url,
                }
                for summary in rag_summaries[:3]
            ]

        try:
            response = run_score_results(control_meta, payload, meta=llm_meta)
        except TimeoutError:
            logger.warning("hybrid.llm_timeout")
            fallback = self._fallback_scores(search_candidates, rag_facets)
            update_observation(
                metadata={
                    "hybrid.llm_cache_hit": False,
                    "hybrid.llm_cache_key": cache_key,
                    "hybrid.llm_timeout": True,
                }
            )
            return fallback, {
                "llm_timeout": True,
                "llm_cache_hit": False,
                "llm_failure": False,
            }, {
                **debug_payload,
                "cache_hit": False,
                "fallback": "timeout",
                "llm_items": len(fallback),
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("hybrid.llm_failure", extra={"error": str(exc)})
            fallback = self._fallback_scores(search_candidates, rag_facets)
            update_observation(
                metadata={
                    "hybrid.llm_cache_hit": False,
                    "hybrid.llm_cache_key": cache_key,
                    "hybrid.llm_failure": True,
                    "hybrid.llm_timeout": False,
                }
            )
            return fallback, {
                "llm_timeout": False,
                "llm_cache_hit": False,
                "llm_failure": True,
            }, {
                **debug_payload,
                "cache_hit": False,
                "fallback": "error",
                "llm_items": len(fallback),
            }

        ranked_payload = (
            response.get("evaluations")
            or response.get("ranked")
            or []
        )
        ranked_items = self._build_llm_items(
            ranked_payload,
            candidates,
            scoring_context,
            rag_facets=rag_facets,
        )

        if ranked_items:
            payload_dump = [item.model_dump(mode="json") for item in ranked_items]
            try:
                cache.set(cache_key, payload_dump, timeout=self.llm_cache_ttl)
            except Exception:  # pragma: no cover - defensive
                logger.debug("hybrid.llm_cache_unavailable")
        update_observation(
            metadata={
                "hybrid.llm_cache_hit": False,
                "hybrid.llm_cache_key": cache_key,
                "hybrid.llm_timeout": False,
            }
        )

        return ranked_items, {
            "llm_timeout": False,
            "llm_cache_hit": False,
            "llm_failure": False,
        }, {
            **debug_payload,
            "cache_hit": False,
            "fallback": None,
            "llm_items": len(ranked_items),
        }

    def _build_criteria(self, scoring_context: ScoringContext | None) -> list[str] | None:
        if not scoring_context:
            return None
        criteria: list[str] = [
            f"Purpose: {scoring_context.purpose}",
            f"Jurisdiction: {scoring_context.jurisdiction}",
        ]
        if scoring_context.output_target:
            criteria.append(f"Output target: {scoring_context.output_target}")
        if scoring_context.preferred_sources:
            criteria.append(
                "Preferred sources: " + ", ".join(scoring_context.preferred_sources)
            )
        if scoring_context.disallowed_sources:
            criteria.append(
                "Disallowed sources: " + ", ".join(scoring_context.disallowed_sources)
            )
        return criteria

    def _fallback_scores(
        self,
        candidates: Sequence[SearchCandidate],
        rag_facets: Mapping[CoverageDimension, float],
    ) -> list[LLMScoredItem]:
        fallback: list[LLMScoredItem] = []
        now = datetime.now(timezone.utc)
        rag_gaps = _gap_dimensions(rag_facets)
        for index, candidate in enumerate(candidates):
            detected = candidate.detected_date or now
            recency_bonus = max(0.0, 30.0 - (now - detected).days / 12)
            trust_bonus = 10.0 if candidate.trust_hint == "high" else 0.0
            score = max(1.0, 60.0 - index * 2 + recency_bonus + trust_bonus)
            reason = "Heuristic fallback ranking"
            facets = _infer_facets(candidate.snippet)
            gap_tags = _dimensions_to_tags(rag_gaps & set(facets.keys()))
            fallback.append(
                LLMScoredItem(
                    candidate_id=candidate.id,
                    score=min(100.0, score),
                    reason=reason,
                    gap_tags=gap_tags,
                    risk_flags=["low_trust"] if candidate.trust_hint == "medium" else [],
                    facet_coverage=facets,
                )
            )
        fallback.sort(key=lambda item: item.score, reverse=True)
        return fallback

    def _build_llm_items(
        self,
        ranked_payload: Sequence[Mapping[str, Any]],
        candidates: Sequence[MutableMapping[str, Any]],
        scoring_context: ScoringContext | None,
        *,
        rag_facets: Mapping[CoverageDimension, float],
    ) -> list[LLMScoredItem]:
        candidates_by_id = {candidate.get("id"): candidate for candidate in candidates}
        rag_gap_dimensions = _gap_dimensions(rag_facets)
        items: list[LLMScoredItem] = []
        for entry in ranked_payload:
            candidate_id = entry.get("candidate_id") or entry.get("id")
            if candidate_id not in candidates_by_id:
                continue
            score = entry.get("score")
            reason = _best_reason(entry.get("reason") or entry.get("reasons"))
            candidate = candidates_by_id[candidate_id]
            snippet = str(candidate.get("snippet") or "")
            facets = _infer_facets(snippet + "\n" + reason)
            custom_facets: dict[str, float] = {}
            llm_facets = entry.get("facet_coverage")
            if isinstance(llm_facets, Mapping):
                for key, value in llm_facets.items():
                    dimension = coerce_enum(key, CoverageDimension)
                    if dimension is None:
                        key_text = str(key).strip()
                        if not key_text:
                            continue
                        upper_key = key_text.upper()
                        if not upper_key.startswith("CUSTOM_"):
                            continue
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            continue
                        if numeric < 0:
                            numeric = 0.0
                        if numeric > 1:
                            numeric = 1.0
                        custom_facets[upper_key] = numeric
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if numeric < 0:
                        numeric = 0.0
                    if numeric > 1:
                        numeric = 1.0
                    facets[dimension] = numeric
            provided_gap_tags = set()
            raw_gap_tags = entry.get("gap_tags")
            if isinstance(raw_gap_tags, str):
                raw_gap_tags = [raw_gap_tags]
            if isinstance(raw_gap_tags, Iterable):
                for tag in raw_gap_tags:
                    dimension = coerce_enum(tag, CoverageDimension)
                    if dimension:
                        provided_gap_tags.add(dimension)
            facet_dimensions = set(facets.keys())
            gap_dimensions = (rag_gap_dimensions & facet_dimensions) | (
                rag_gap_dimensions & provided_gap_tags
            )
            gap_tags = _dimensions_to_tags(gap_dimensions)
            risk_flags: list[str] = []
            raw_risk_flags = entry.get("risk_flags")
            if isinstance(raw_risk_flags, str):
                raw_risk_flags = [raw_risk_flags]
            if isinstance(raw_risk_flags, Iterable):
                for flag in raw_risk_flags:
                    if flag in (None, ""):
                        continue
                    text = str(flag).strip()
                    if text and text not in risk_flags:
                        risk_flags.append(text)
            if candidate.get("trust_hint") == "medium" and "medium_trust_domain" not in risk_flags:
                risk_flags.append("medium_trust_domain")
            if candidate.get("is_pdf") and "pdf" not in risk_flags:
                risk_flags.append("pdf")
            items.append(
                LLMScoredItem(
                    candidate_id=str(candidate_id),
                    score=float(score),
                    reason=reason,
                    gap_tags=gap_tags,
                    risk_flags=risk_flags,
                    facet_coverage=facets,
                    custom_facets=custom_facets,
                )
            )
        items.sort(key=lambda item: item.score, reverse=True)
        return items

    # Deduplicate ----------------------------------------------------------------

    def _deduplicate_and_select(
        self,
        *,
        candidates: Sequence[MutableMapping[str, Any]],
        rerank_payload: Sequence[LLMScoredItem],
        rag_summaries: Sequence[RAGCoverageSummary],
        rag_facets: Mapping[CoverageDimension, float],
        domain_policy: DomainPolicy,
        scoring_context: ScoringContext | None,
        rrf_k: int,
    ) -> tuple[HybridResult, dict[str, Any]]:
        candidate_map = {candidate.get("id"): candidate for candidate in candidates}
        rag_gap_dimensions = _gap_dimensions(rag_facets)

        url_seen: set[str] = set()
        diversity_buckets: set[str] = set()
        final_ranked: list[LLMScoredItem] = []
        fused_scores: dict[str, float] = {}
        rrf_components: dict[str, dict[str, Any]] = {}
        min_diversity = (
            getattr(scoring_context, "min_diversity_buckets", None)
            or self.default_min_diversity_buckets
        )

        for position, item in enumerate(rerank_payload, start=1):
            candidate = candidate_map.get(item.candidate_id, {})
            host = candidate.get("host")
            domain_rank = 1.0 if candidate.get("domain_type") else 2.0
            base_term = 1.0 / (rrf_k + position)
            rag_bonus = 0.0
            if rag_gap_dimensions & set(item.facet_coverage.keys()):
                rag_bonus = 1.0 / (rrf_k + 1)
            domain_bonus = 1.0 / (rrf_k + domain_rank)
            policy_bonus = 0.0
            policy_decision = None
            decision = domain_policy.evaluate(host)
            if decision:
                policy_decision = {
                    "action": decision.action.value,
                    "priority": decision.priority,
                    "source": decision.source,
                }
                scale = (max(decision.priority, 0) / 100.0) / (rrf_k + 0.5)
                if decision.action is DomainPolicyAction.BOOST:
                    policy_bonus = scale
                elif decision.action is DomainPolicyAction.REJECT:
                    policy_bonus = -scale
            fused_score = base_term + rag_bonus + domain_bonus + policy_bonus
            fused_scores[item.candidate_id] = fused_score
            rrf_components[item.candidate_id] = {
                "position": position,
                "rrf_term": base_term,
                "rag_bonus": rag_bonus,
                "domain_bonus": domain_bonus,
                "policy_bonus": policy_bonus,
                "policy_decision": policy_decision,
            }

        ordered_items = sorted(
            rerank_payload,
            key=lambda item: fused_scores.get(item.candidate_id, 0.0),
            reverse=True,
        )

        for item in ordered_items:
            candidate = candidate_map.get(item.candidate_id)
            if not candidate:
                continue
            url = candidate.get("url")
            if url and url in url_seen:
                continue
            bucket = candidate.get("domain_type") or candidate.get("host") or "unknown"
            final_ranked.append(item)
            if url:
                url_seen.add(url)
            diversity_buckets.add(bucket)
            if len(final_ranked) >= self.rerank_top_k * 2:
                break

        if len(diversity_buckets) < min_diversity:
            remaining = [item for item in ordered_items if item not in final_ranked]
            for item in remaining:
                candidate = candidate_map.get(item.candidate_id)
                if not candidate:
                    continue
                bucket = candidate.get("domain_type") or candidate.get("host") or "unknown"
                if bucket in diversity_buckets:
                    continue
                final_ranked.append(item)
                diversity_buckets.add(bucket)
                if len(diversity_buckets) >= min_diversity:
                    break

        final_ranked.sort(
            key=lambda item: fused_scores.get(item.candidate_id, item.score), reverse=True
        )
        top_k = final_ranked[: self.rerank_top_k]

        coverage_delta = self._coverage_delta(top_k, rag_facets)
        recommended_ingest = self._recommended_ingest(top_k, candidate_map)
        fusion_debug = {
            "rrf_components": rrf_components,
            "fused_scores": fused_scores,
            "diversity_buckets": sorted(diversity_buckets),
            "rag_gap_dimensions": _dimensions_to_tags(rag_gap_dimensions),
            "rrf_k": rrf_k,
        }

        update_observation(
            metadata={
                "hybrid.rrf_k_used": rrf_k,
                "hybrid.diversity_buckets_final": len(diversity_buckets),
            }
        )

        return (
            HybridResult(
                ranked=final_ranked,
                top_k=top_k,
                coverage_delta=coverage_delta,
                recommended_ingest=recommended_ingest,
            ),
            fusion_debug,
        )

    def _coverage_delta(
        self,
        top_k: Sequence[LLMScoredItem],
        rag_facets: Mapping[CoverageDimension, float],
    ) -> str:
        aggregated: Counter[CoverageDimension] = Counter()
        for item in top_k:
            for dimension_raw, score in item.facet_coverage.items():
                dimension = coerce_enum(dimension_raw, CoverageDimension)
                if dimension is None:
                    continue
                if score >= 0.2:
                    aggregated[dimension] += 1
        improvements = [
            dimension
            for dimension, count in aggregated.items()
            if rag_facets.get(dimension, 0.0) < 0.4 and count
        ]
        if not improvements:
            return "Hybrid ranking maintains existing coverage"
        labels = ", ".join(dimension.value for dimension in improvements)
        return f"Adds coverage across: {labels}"

    def _recommended_ingest(
        self,
        top_k: Sequence[LLMScoredItem],
        candidate_map: Mapping[str, Mapping[str, Any]],
    ) -> list[RecommendedIngestItem]:
        recommendations: list[RecommendedIngestItem] = []
        for item in top_k:
            candidate = candidate_map.get(item.candidate_id)
            if not candidate:
                continue
            trust = candidate.get("trust_hint")
            if trust not in {"high", "medium"}:
                continue
            if "pdf" in item.risk_flags:
                continue
            if item.gap_tags:
                reason = "Addresses gaps: " + ", ".join(item.gap_tags)
            else:
                reason = "High relevance candidate"
            recommendations.append(
                RecommendedIngestItem(candidate_id=item.candidate_id, reason=reason)
            )
        return recommendations


GRAPH = HybridSearchAndScoreGraph(
    rrf_k=getattr(settings, "HYBRID_RRF_K", RRF_K),
    default_min_diversity_buckets=getattr(
        settings, "HYBRID_MIN_DIVERSITY_BUCKETS", MIN_DIVERSITY_BUCKETS
    ),
)


def build_graph() -> HybridSearchAndScoreGraph:
    return GRAPH


def run(
    state: Mapping[str, Any] | MutableMapping[str, Any],
    meta: Mapping[str, Any] | MutableMapping[str, Any],
) -> tuple[MutableMapping[str, Any], Mapping[str, Any]]:
    return GRAPH.run(state, meta)


__all__ = ["HybridSearchAndScoreGraph", "GRAPH", "build_graph", "run"]

