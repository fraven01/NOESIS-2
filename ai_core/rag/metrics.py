"""Lightweight instrumentation primitives for the RAG vector client."""

from __future__ import annotations

from typing import List

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
else:  # pragma: no cover - covered via direct value inspection in tests
    RAG_UPSERT_CHUNKS = _FallbackCounter()


if _PromHistogram is not None:  # pragma: no cover - exercised in integration
    RAG_SEARCH_MS = _PromHistogram(
        "rag_search_ms",
        "Latency of pgvector similarity search in milliseconds.",
    )
else:  # pragma: no cover - covered via direct value inspection in tests
    RAG_SEARCH_MS = _FallbackHistogram()


__all__ = ["RAG_UPSERT_CHUNKS", "RAG_SEARCH_MS"]
