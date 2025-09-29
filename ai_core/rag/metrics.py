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


if _PromCounter is not None:  # pragma: no cover - exercised in integration
    RAG_UPSERT_CHUNKS = _PromCounter(
        "rag_upsert_chunks",
        "Number of chunks written into the pgvector store.",
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
else:  # pragma: no cover - covered via direct value inspection in tests
    RAG_UPSERT_CHUNKS = _FallbackCounter()
    INGESTION_DOCS_INSERTED = _FallbackCounter()
    INGESTION_DOCS_REPLACED = _FallbackCounter()
    INGESTION_DOCS_SKIPPED = _FallbackCounter()
    INGESTION_CHUNKS_WRITTEN = _FallbackCounter()


if _PromHistogram is not None:  # pragma: no cover - exercised in integration
    RAG_SEARCH_MS = _PromHistogram(
        "rag_search_ms",
        "Latency of pgvector similarity search in milliseconds.",
    )
    INGESTION_RUN_MS = _PromHistogram(
        "rag_ingestion_run_ms",
        "Wall-clock duration of ingestion runs in milliseconds.",
    )
else:  # pragma: no cover - covered via direct value inspection in tests
    RAG_SEARCH_MS = _FallbackHistogram()
    INGESTION_RUN_MS = _FallbackHistogram()


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
    "RAG_SEARCH_MS",
    "RAG_RETRY_ATTEMPTS",
    "RAG_HEALTH_CHECKS",
    "INGESTION_DOCS_INSERTED",
    "INGESTION_DOCS_REPLACED",
    "INGESTION_DOCS_SKIPPED",
    "INGESTION_CHUNKS_WRITTEN",
    "INGESTION_RUN_MS",
]
