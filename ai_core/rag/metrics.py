"""Lightweight instrumentation primitives for the RAG vector client."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter as _PromCounter  # type: ignore
    from prometheus_client import Histogram as _PromHistogram  # type: ignore
except Exception:  # pragma: no cover - we provide a fallback used in tests
    _PromCounter = None
    _PromHistogram = None


class _FallbackCounter:
    def __init__(self) -> None:
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    @property
    def value(self) -> float:
        return self._value


class _FallbackCounterVec:
    def __init__(self) -> None:
        self._counters: Dict[Tuple[Tuple[str, str], ...], _FallbackCounter] = {}

    def labels(self, **labels: str) -> _FallbackCounter:
        key = tuple(sorted(labels.items()))
        if key not in self._counters:
            self._counters[key] = _FallbackCounter()
        return self._counters[key]

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        self.labels(**labels).inc(amount)

    def value(self, **labels: str) -> float:
        key = tuple(sorted(labels.items()))
        counter = self._counters.get(key)
        return counter.value if counter else 0.0


class _FallbackHistogram:
    def __init__(self) -> None:
        self._values: List[float] = []

    def observe(self, amount: float) -> None:
        self._values.append(amount)

    @property
    def samples(self) -> List[float]:
        return list(self._values)


class _FallbackHistogramVec:
    class _Recorder:
        def __init__(self, values: List[float]) -> None:
            self._values = values

        def observe(self, amount: float) -> None:
            self._values.append(amount)

        @property
        def samples(self) -> List[float]:
            return list(self._values)

    def __init__(self) -> None:
        self._values: Dict[Tuple[Tuple[str, str], ...], List[float]] = {}

    def labels(self, **labels: str) -> "_Recorder":
        key = tuple(sorted(labels.items()))
        if key not in self._values:
            self._values[key] = []
        return self._Recorder(self._values[key])

    def observe(self, amount: float, **labels: str) -> None:
        self.labels(**labels).observe(amount)

    def samples(self, **labels: str) -> List[float]:
        key = tuple(sorted(labels.items()))
        return list(self._values.get(key, []))


if _PromCounter is not None:  # pragma: no cover - exercised in integration
    RAG_UPSERT_CHUNKS = _PromCounter(
        "rag_upsert_chunks",
        "Number of chunks written into the pgvector store.",
    )
    RAG_EMBEDDINGS_EMPTY_TOTAL = _PromCounter(
        "rag_embeddings_empty_total",
        "Number of chunks skipped because the embedding provider returned an empty vector.",
    )
    INGESTION_DOCS_INSERTED = _PromCounter(
        "rag_ingestion_docs_inserted_total",
        "Number of documents inserted during ingestion runs.",
    )
    INGESTION_DOCS_REPLACED = _PromCounter(
        "rag_ingestion_docs_replaced_total",
        "Number of documents replaced during ingestion runs.",
    )
    INGESTION_DOCS_SKIPPED = _PromCounter(
        "rag_ingestion_docs_skipped_total",
        "Number of documents skipped during ingestion runs due to unchanged content.",
    )
    INGESTION_CHUNKS_WRITTEN = _PromCounter(
        "rag_ingestion_chunks_written_total",
        "Number of chunks written during ingestion runs.",
    )
    RAG_QUERY_TOTAL = _PromCounter(
        "rag_query_total",
        "Number of hybrid RAG queries executed.",
        ["tenant_id", "index_kind", "hybrid"],
    )
    RAG_QUERY_EMPTY_VEC_TOTAL = _PromCounter(
        "rag_query_empty_vec_total",
        "Number of queries with effectively zero embedding (dev/dummy).",
        ["tenant_id"],
    )
    RAG_QUERY_NO_HIT = _PromCounter(
        "rag_query_no_hit_total",
        "Number of hybrid RAG queries without results above the similarity threshold.",
        ["tenant_id"],
    )
    RAG_QUERY_BELOW_CUTOFF_TOTAL = _PromCounter(
        "rag_query_below_cutoff_total",
        "Number of candidates filtered by min_sim cutoff.",
        ["tenant_id"],
    )
else:  # pragma: no cover - covered via direct value inspection in tests
    RAG_UPSERT_CHUNKS = _FallbackCounter()
    RAG_EMBEDDINGS_EMPTY_TOTAL = _FallbackCounter()
    INGESTION_DOCS_INSERTED = _FallbackCounter()
    INGESTION_DOCS_REPLACED = _FallbackCounter()
    INGESTION_DOCS_SKIPPED = _FallbackCounter()
    INGESTION_CHUNKS_WRITTEN = _FallbackCounter()
    RAG_QUERY_TOTAL = _FallbackCounterVec()
    RAG_QUERY_EMPTY_VEC_TOTAL = _FallbackCounterVec()
    RAG_QUERY_NO_HIT = _FallbackCounterVec()
    RAG_QUERY_BELOW_CUTOFF_TOTAL = _FallbackCounterVec()


if _PromHistogram is not None:  # pragma: no cover - exercised in integration
    RAG_SEARCH_MS = _PromHistogram(
        "rag_search_ms",
        "Latency of pgvector similarity search in milliseconds.",
    )
    INGESTION_RUN_MS = _PromHistogram(
        "rag_ingestion_run_ms",
        "Wall-clock duration of ingestion runs in milliseconds.",
    )
    RAG_QUERY_LATENCY_MS = _PromHistogram(
        "rag_query_latency_ms",
        "Latency of hybrid RAG queries in milliseconds.",
        buckets=(5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    )
    RAG_QUERY_CANDIDATES = _PromHistogram(
        "rag_query_candidates",
        "Number of candidates considered during hybrid RAG queries.",
        ["tenant_id", "type"],
        buckets=(1, 5, 10, 20, 50, 100),
    )
    RAG_QUERY_TOP1_SIM = _PromHistogram(
        "rag_query_top1_sim",
        "Fused similarity score of the top retrieval result.",
        ["tenant_id"],
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
else:  # pragma: no cover - covered via direct value inspection in tests
    RAG_SEARCH_MS = _FallbackHistogram()
    INGESTION_RUN_MS = _FallbackHistogram()
    RAG_QUERY_LATENCY_MS = _FallbackHistogramVec()
    RAG_QUERY_CANDIDATES = _FallbackHistogramVec()
    RAG_QUERY_TOP1_SIM = _FallbackHistogramVec()


if _PromCounter is not None:  # pragma: no cover - exercised in integration
    RAG_RETRY_ATTEMPTS = _PromCounter(
        "rag_retry_attempts_total",
        "Number of retry attempts executed by pgvector operations.",
        ["operation"],
    )
    RAG_HEALTH_CHECKS = _PromCounter(
        "rag_vector_health_checks_total",
        "Health check results per vector store scope.",
        ["scope", "status"],
    )
else:  # pragma: no cover - covered via fallback value inspection in tests
    RAG_RETRY_ATTEMPTS = _FallbackCounterVec()
    RAG_HEALTH_CHECKS = _FallbackCounterVec()


__all__ = [
    "RAG_UPSERT_CHUNKS",
    "RAG_EMBEDDINGS_EMPTY_TOTAL",
    "RAG_SEARCH_MS",
    "RAG_RETRY_ATTEMPTS",
    "RAG_HEALTH_CHECKS",
    "INGESTION_DOCS_INSERTED",
    "INGESTION_DOCS_REPLACED",
    "INGESTION_DOCS_SKIPPED",
    "INGESTION_CHUNKS_WRITTEN",
    "INGESTION_RUN_MS",
    "RAG_QUERY_TOTAL",
    "RAG_QUERY_EMPTY_VEC_TOTAL",
    "RAG_QUERY_NO_HIT",
    "RAG_QUERY_BELOW_CUTOFF_TOTAL",
    "RAG_QUERY_LATENCY_MS",
    "RAG_QUERY_CANDIDATES",
    "RAG_QUERY_TOP1_SIM",
    "EvalMetrics",
    "calculate_coverage",
    "evaluate_ranking",
]


@dataclass(frozen=True)
class EvalMetrics:
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    relevant_count: int
    retrieved_count: int


def _binary_relevance(
    relevant_ids: Iterable[str], ranked_ids: Sequence[str], *, k: int
) -> list[int]:
    if k <= 0:
        return []
    relevant = {str(item) for item in relevant_ids if str(item).strip()}
    if not relevant:
        return []
    hits: list[int] = []
    for candidate in ranked_ids[:k]:
        hits.append(1 if candidate in relevant else 0)
    return hits


def _dcg(hits: Sequence[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(hits, start=1):
        if rel <= 0:
            continue
        score += rel / math.log2(idx + 1)
    return score


def evaluate_ranking(
    relevant_ids: Iterable[str],
    ranked_ids: Sequence[str],
    *,
    k: int,
) -> EvalMetrics:
    """Compute Recall@k/MRR@k/NDCG@k for binary relevance."""

    cleaned_relevant = [str(item) for item in relevant_ids if str(item).strip()]
    total_relevant = len(set(cleaned_relevant))
    hits = _binary_relevance(cleaned_relevant, ranked_ids, k=k)
    retrieved = min(len(ranked_ids), max(0, k))

    if total_relevant <= 0 or not hits:
        return EvalMetrics(
            recall_at_k=0.0,
            mrr_at_k=0.0,
            ndcg_at_k=0.0,
            relevant_count=total_relevant,
            retrieved_count=retrieved,
        )

    recall = sum(hits) / total_relevant

    mrr = 0.0
    for idx, rel in enumerate(hits, start=1):
        if rel > 0:
            mrr = 1.0 / idx
            break

    dcg = _dcg(hits)
    ideal_hits = [1] * min(total_relevant, len(hits))
    idcg = _dcg(ideal_hits)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return EvalMetrics(
        recall_at_k=recall,
        mrr_at_k=mrr,
        ndcg_at_k=ndcg,
        relevant_count=total_relevant,
        retrieved_count=retrieved,
    )


def calculate_coverage(
    retrieved: Sequence[Mapping[str, object]],
    document_id: str,
    total: int,
) -> dict[str, object]:
    if total <= 0:
        return {"coverage_ratio": 0.0, "all_covered": False}
    doc_id = str(document_id).strip()
    if not doc_id:
        return {"coverage_ratio": 0.0, "all_covered": False}

    covered_ids: set[str] = set()
    for snippet in retrieved:
        if not isinstance(snippet, Mapping):
            continue
        meta = snippet.get("meta")
        meta_payload = meta if isinstance(meta, Mapping) else {}
        snippet_doc_id = str(meta_payload.get("document_id") or "").strip()
        if snippet_doc_id != doc_id:
            continue
        chunk_id = (
            meta_payload.get("chunk_id") or meta_payload.get("id") or snippet.get("id")
        )
        if chunk_id:
            covered_ids.add(str(chunk_id))

    covered = len(covered_ids)
    ratio = min(1.0, covered / max(1, total))
    return {"coverage_ratio": ratio, "all_covered": covered >= total}
