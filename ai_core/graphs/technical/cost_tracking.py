from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping


def coerce_cost_value(value: Any) -> float | None:
    """Best-effort conversion of heterogeneous inputs to a float value."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if coerced < 0:
            return None
        return coerced
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            coerced = float(candidate)
        except (TypeError, ValueError):
            return None
        if coerced < 0:
            return None
        return coerced
    return None


class GraphCostTracker:
    """Collects cost components emitted during a graph run."""

    def __init__(self, initial_total: float | None = None) -> None:
        self._total_usd = 0.0
        self._components: list[dict[str, Any]] = []
        self._reconciliation_ids: set[str] = set()
        if initial_total is not None:
            initial = coerce_cost_value(initial_total)
            if initial and initial > 0:
                self.add_component(
                    source="meta",
                    usd=initial,
                    kind="initial",
                )

    @property
    def total_usd(self) -> float:
        return self._total_usd

    @property
    def components(self) -> list[dict[str, Any]]:
        return list(self._components)

    def add_component(self, *, source: str, usd: float, **extra: Any) -> None:
        amount = coerce_cost_value(usd)
        if amount is None:
            return
        component: dict[str, Any] = {"source": source, "usd": float(amount)}
        for key, value in extra.items():
            if value is None:
                continue
            component[key] = (
                value if isinstance(value, (str, int, float, bool)) else str(value)
            )
        ledger_entry_id = component.get("ledger_entry_id")
        if ledger_entry_id:
            self._reconciliation_ids.add(str(ledger_entry_id))
        self._components.append(component)
        self._total_usd += float(amount)

    def record_ledger_meta(self, meta: Any) -> None:
        if not isinstance(meta, Mapping):
            return
        usage = meta.get("usage")
        usd = None
        if isinstance(usage, Mapping):
            cost_block = usage.get("cost")
            if isinstance(cost_block, Mapping):
                usd = cost_block.get("usd") or cost_block.get("total")
        if usd is None:
            usd = meta.get("cost_usd") or meta.get("usd")
        coerced = coerce_cost_value(usd)
        if coerced is None:
            return
        source = str(meta.get("source") or meta.get("label") or "ledger")
        component: dict[str, Any] = {
            "label": meta.get("label"),
            "model": meta.get("model"),
            "trace_id": meta.get("trace_id"),
        }
        entry_id = meta.get("id") or meta.get("ledger_id")
        if entry_id:
            component["ledger_entry_id"] = str(entry_id)
        cache_hit = meta.get("cache_hit")
        if cache_hit is not None:
            component["cache_hit"] = bool(cache_hit)
        latency = meta.get("latency_ms")
        if latency is not None:
            component["latency_ms"] = latency
        self.add_component(source=source, usd=float(coerced), **component)

    def summary(self, ledger_id: str | None = None) -> dict[str, Any] | None:
        if not self._components:
            return None
        summary: dict[str, Any] = {
            "total_usd": self.total_usd,
            "components": self.components,
        }
        reconciliation: dict[str, Any] = {}
        if ledger_id:
            reconciliation["ledger_id"] = ledger_id
        if self._reconciliation_ids:
            reconciliation["entry_ids"] = sorted(self._reconciliation_ids)
        if reconciliation:
            summary["reconciliation"] = reconciliation
        return summary


@contextmanager
def track_ledger_costs(initial_total: float | None = None):
    """Context manager that tracks ledger entries emitted during a graph run."""

    tracker = GraphCostTracker(initial_total)
    yield tracker
