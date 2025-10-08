"""Lightweight instrumentation primitives for the RAG vector client."""

from __future__ import annotations

from typing import Dict, List, Tuple

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
]
