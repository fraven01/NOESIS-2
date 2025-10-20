"""Prometheus-style metrics for document, asset, storage, and CLI operations."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency for production deployments
    from prometheus_client import Counter as _PromCounter  # type: ignore
    from prometheus_client import Histogram as _PromHistogram  # type: ignore
except Exception:  # pragma: no cover - fallback used in tests
    _PromCounter = None  # type: ignore[assignment]
    _PromHistogram = None  # type: ignore[assignment]


class _FallbackCounter:
    def __init__(self) -> None:
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0


class _FallbackCounterVec:
    def __init__(self) -> None:
        self._counters: Dict[Tuple[Tuple[str, str], ...], _FallbackCounter] = {}

    def labels(self, **labels: str) -> _FallbackCounter:
        key = tuple(sorted(labels.items()))
        counter = self._counters.get(key)
        if counter is None:
            counter = _FallbackCounter()
            self._counters[key] = counter
        return counter

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        self.labels(**labels).inc(amount)

    def value(self, **labels: str) -> float:
        key = tuple(sorted(labels.items()))
        counter = self._counters.get(key)
        return counter.value if counter else 0.0

    def reset(self) -> None:
        self._counters.clear()


class _FallbackHistogramRecorder:
    def __init__(self, store: List[float]) -> None:
        self._store = store

    def observe(self, amount: float) -> None:
        self._store.append(amount)

    @property
    def samples(self) -> List[float]:
        return list(self._store)

    @property
    def count(self) -> int:
        return len(self._store)


class _FallbackHistogramVec:
    def __init__(self) -> None:
        self._values: Dict[Tuple[Tuple[str, str], ...], List[float]] = {}

    def labels(self, **labels: str) -> _FallbackHistogramRecorder:
        key = tuple(sorted(labels.items()))
        store = self._values.get(key)
        if store is None:
            store = []
            self._values[key] = store
        return _FallbackHistogramRecorder(store)

    def observe(self, amount: float, **labels: str) -> None:
        self.labels(**labels).observe(amount)

    def samples(self, **labels: str) -> List[float]:
        key = tuple(sorted(labels.items()))
        store = self._values.get(key)
        return list(store) if store else []

    def count(self, **labels: str) -> int:
        key = tuple(sorted(labels.items()))
        store = self._values.get(key)
        return len(store) if store else 0

    def reset(self) -> None:
        self._values.clear()


def _normalize_workflow_label(workflow_id: str | None) -> str:
    """Return a stable label value for ``workflow_id``."""

    value = (workflow_id or "").strip()
    return value or "unknown"


if _PromCounter is not None:  # pragma: no cover - exercised in integration environments
    DOCUMENT_OPERATION_TOTAL = _PromCounter(
        "documents_operation_total",
        "Number of document repository operations by event and status.",
        ["event", "status", "workflow_id"],
    )
    ASSET_OPERATION_TOTAL = _PromCounter(
        "documents_asset_operation_total",
        "Number of asset repository operations by event and status.",
        ["event", "status", "workflow_id"],
    )
    STORAGE_OPERATION_TOTAL = _PromCounter(
        "documents_storage_operation_total",
        "Number of storage adapter operations by event and status.",
        ["event", "status", "workflow_id"],
    )
    PIPELINE_OPERATION_TOTAL = _PromCounter(
        "documents_pipeline_operation_total",
        "Number of caption pipeline operations by event and status.",
        ["event", "status", "workflow_id"],
    )
    CLI_OPERATION_TOTAL = _PromCounter(
        "documents_cli_operation_total",
        "Number of CLI invocations by event and status.",
        ["event", "status", "workflow_id"],
    )
    OTHER_OPERATION_TOTAL = _PromCounter(
        "documents_other_operation_total",
        "Number of uncategorised document operations by event and status.",
        ["event", "status", "workflow_id"],
    )
    CAPTION_RUNS_TOTAL = _PromCounter(
        "documents_caption_runs_total",
        "Number of caption pipeline runs by status.",
        ["status", "workflow_id"],
    )
    PIPELINE_BLOCKS_TOTAL = _PromCounter(
        "documents_pipeline_blocks_total",
        "Number of text blocks emitted by parser executions.",
        ["workflow_id"],
    )
    PIPELINE_ASSETS_TOTAL = _PromCounter(
        "documents_pipeline_assets_total",
        "Number of assets extracted during parser executions.",
        ["workflow_id"],
    )
    PIPELINE_OCR_TRIGGER_TOTAL = _PromCounter(
        "documents_pipeline_ocr_trigger_total",
        "Number of pages flagged for OCR follow-up during parsing.",
        ["workflow_id"],
    )
    PIPELINE_CAPTION_HITS_TOTAL = _PromCounter(
        "documents_pipeline_caption_hits_total",
        "Number of captions accepted from VLM results.",
        ["workflow_id"],
    )
    PIPELINE_CAPTION_ATTEMPTS_TOTAL = _PromCounter(
        "documents_pipeline_caption_attempts_total",
        "Number of caption attempts evaluated for acceptance.",
        ["workflow_id"],
    )
else:  # pragma: no cover - exercised through direct inspection in unit tests
    DOCUMENT_OPERATION_TOTAL = _FallbackCounterVec()
    ASSET_OPERATION_TOTAL = _FallbackCounterVec()
    STORAGE_OPERATION_TOTAL = _FallbackCounterVec()
    PIPELINE_OPERATION_TOTAL = _FallbackCounterVec()
    CLI_OPERATION_TOTAL = _FallbackCounterVec()
    OTHER_OPERATION_TOTAL = _FallbackCounterVec()
    CAPTION_RUNS_TOTAL = _FallbackCounterVec()
    PIPELINE_BLOCKS_TOTAL = _FallbackCounterVec()
    PIPELINE_ASSETS_TOTAL = _FallbackCounterVec()
    PIPELINE_OCR_TRIGGER_TOTAL = _FallbackCounterVec()
    PIPELINE_CAPTION_HITS_TOTAL = _FallbackCounterVec()
    PIPELINE_CAPTION_ATTEMPTS_TOTAL = _FallbackCounterVec()


if (
    _PromHistogram is not None
):  # pragma: no cover - exercised in integration environments
    DOCUMENT_OPERATION_DURATION_MS = _PromHistogram(
        "documents_operation_duration_ms",
        "Duration of document repository operations in milliseconds.",
        ["event", "status", "workflow_id"],
    )
    ASSET_OPERATION_DURATION_MS = _PromHistogram(
        "documents_asset_operation_duration_ms",
        "Duration of asset repository operations in milliseconds.",
        ["event", "status", "workflow_id"],
    )
    STORAGE_OPERATION_DURATION_MS = _PromHistogram(
        "documents_storage_operation_duration_ms",
        "Duration of storage adapter operations in milliseconds.",
        ["event", "status", "workflow_id"],
    )
    PIPELINE_OPERATION_DURATION_MS = _PromHistogram(
        "documents_pipeline_operation_duration_ms",
        "Duration of caption pipeline operations in milliseconds.",
        ["event", "status", "workflow_id"],
    )
    CLI_OPERATION_DURATION_MS = _PromHistogram(
        "documents_cli_operation_duration_ms",
        "Duration of CLI operations in milliseconds.",
        ["event", "status", "workflow_id"],
    )
    OTHER_OPERATION_DURATION_MS = _PromHistogram(
        "documents_other_operation_duration_ms",
        "Duration of uncategorised document operations in milliseconds.",
        ["event", "status", "workflow_id"],
    )
    CAPTION_DURATION_MS = _PromHistogram(
        "documents_caption_duration_ms",
        "Duration of caption pipeline runs in milliseconds.",
        ["status", "workflow_id"],
    )
    PIPELINE_CAPTION_HIT_RATIO = _PromHistogram(
        "documents_pipeline_caption_hit_ratio",
        "Distribution of caption hit rates produced by the captioning pipeline.",
        ["workflow_id"],
    )
else:  # pragma: no cover - exercised through direct inspection in unit tests
    DOCUMENT_OPERATION_DURATION_MS = _FallbackHistogramVec()
    ASSET_OPERATION_DURATION_MS = _FallbackHistogramVec()
    STORAGE_OPERATION_DURATION_MS = _FallbackHistogramVec()
    PIPELINE_OPERATION_DURATION_MS = _FallbackHistogramVec()
    CLI_OPERATION_DURATION_MS = _FallbackHistogramVec()
    OTHER_OPERATION_DURATION_MS = _FallbackHistogramVec()
    CAPTION_DURATION_MS = _FallbackHistogramVec()
    PIPELINE_CAPTION_HIT_RATIO = _FallbackHistogramVec()


def observe_event(
    event: str,
    status: str,
    duration_ms: float,
    *,
    workflow_id: str | None = None,
) -> None:
    """Record metrics for the given event, status, and duration."""

    workflow_label = _normalize_workflow_label(workflow_id)
    labels = {"event": event, "status": status, "workflow_id": workflow_label}
    if event.startswith("docs."):
        DOCUMENT_OPERATION_TOTAL.labels(**labels).inc()
        DOCUMENT_OPERATION_DURATION_MS.labels(**labels).observe(duration_ms)
    elif event.startswith("assets."):
        ASSET_OPERATION_TOTAL.labels(**labels).inc()
        ASSET_OPERATION_DURATION_MS.labels(**labels).observe(duration_ms)
    elif event.startswith("storage."):
        STORAGE_OPERATION_TOTAL.labels(**labels).inc()
        STORAGE_OPERATION_DURATION_MS.labels(**labels).observe(duration_ms)
    elif event.startswith("pipeline."):
        PIPELINE_OPERATION_TOTAL.labels(**labels).inc()
        PIPELINE_OPERATION_DURATION_MS.labels(**labels).observe(duration_ms)
        if event == "pipeline.assets_caption":
            CAPTION_RUNS_TOTAL.labels(status=status, workflow_id=workflow_label).inc()
            CAPTION_DURATION_MS.labels(
                status=status, workflow_id=workflow_label
            ).observe(duration_ms)
    elif event.startswith("cli."):
        CLI_OPERATION_TOTAL.labels(**labels).inc()
        CLI_OPERATION_DURATION_MS.labels(**labels).observe(duration_ms)
    else:
        OTHER_OPERATION_TOTAL.labels(**labels).inc()
        OTHER_OPERATION_DURATION_MS.labels(**labels).observe(duration_ms)


def reset_metrics() -> None:
    """Reset fallback metric state for deterministic tests."""

    for metric in (
        DOCUMENT_OPERATION_TOTAL,
        ASSET_OPERATION_TOTAL,
        STORAGE_OPERATION_TOTAL,
        PIPELINE_OPERATION_TOTAL,
        CLI_OPERATION_TOTAL,
        OTHER_OPERATION_TOTAL,
        CAPTION_RUNS_TOTAL,
        PIPELINE_BLOCKS_TOTAL,
        PIPELINE_ASSETS_TOTAL,
        PIPELINE_OCR_TRIGGER_TOTAL,
        PIPELINE_CAPTION_HITS_TOTAL,
        PIPELINE_CAPTION_ATTEMPTS_TOTAL,
    ):
        reset = getattr(metric, "reset", None)
        if callable(reset):
            reset()

    for metric in (
        DOCUMENT_OPERATION_DURATION_MS,
        ASSET_OPERATION_DURATION_MS,
        STORAGE_OPERATION_DURATION_MS,
        PIPELINE_OPERATION_DURATION_MS,
        CLI_OPERATION_DURATION_MS,
        OTHER_OPERATION_DURATION_MS,
        CAPTION_DURATION_MS,
        PIPELINE_CAPTION_HIT_RATIO,
    ):
        reset = getattr(metric, "reset", None)
        if callable(reset):
            reset()


def counter_value(metric: Any, **labels: str) -> float | None:
    """Return the counter value for ``labels`` if accessible."""

    accessor = getattr(metric, "value", None)
    if callable(accessor):
        return float(accessor(**labels))

    if hasattr(metric, "labels"):
        try:
            child = metric.labels(**labels)
        except Exception:  # pragma: no cover - defensive
            return None
        value_attr = getattr(child, "_value", None)
        if value_attr is not None and hasattr(value_attr, "get"):
            try:
                return float(value_attr.get())
            except Exception:  # pragma: no cover - defensive
                return None
    return None


def histogram_count(metric: Any, **labels: str) -> float | None:
    """Return the number of samples recorded for ``labels`` if accessible."""

    accessor = getattr(metric, "count", None)
    if callable(accessor):
        return float(accessor(**labels))

    if hasattr(metric, "labels"):
        try:
            child = metric.labels(**labels)
        except Exception:  # pragma: no cover - defensive
            return None
        count_attr = getattr(child, "_count", None)
        if count_attr is not None and hasattr(count_attr, "get"):
            try:
                return float(count_attr.get())
            except Exception:  # pragma: no cover - defensive
                return None
    return None


__all__ = [
    "ASSET_OPERATION_DURATION_MS",
    "ASSET_OPERATION_TOTAL",
    "CAPTION_DURATION_MS",
    "CAPTION_RUNS_TOTAL",
    "CLI_OPERATION_DURATION_MS",
    "CLI_OPERATION_TOTAL",
    "DOCUMENT_OPERATION_DURATION_MS",
    "DOCUMENT_OPERATION_TOTAL",
    "OTHER_OPERATION_DURATION_MS",
    "OTHER_OPERATION_TOTAL",
    "PIPELINE_OPERATION_DURATION_MS",
    "PIPELINE_OPERATION_TOTAL",
    "PIPELINE_BLOCKS_TOTAL",
    "PIPELINE_ASSETS_TOTAL",
    "PIPELINE_OCR_TRIGGER_TOTAL",
    "PIPELINE_CAPTION_HITS_TOTAL",
    "PIPELINE_CAPTION_ATTEMPTS_TOTAL",
    "PIPELINE_CAPTION_HIT_RATIO",
    "STORAGE_OPERATION_DURATION_MS",
    "STORAGE_OPERATION_TOTAL",
    "counter_value",
    "histogram_count",
    "observe_event",
    "reset_metrics",
]
