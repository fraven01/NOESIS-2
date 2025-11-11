"""In-memory store for the developer HITL UI.

The store fabricates deterministic mock runs so the dev UI can
exercise approval flows, SSE updates and countdown behaviour without
impacting production data.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from random import Random
from typing import Any
from uuid import uuid4

from structlog.stdlib import get_logger

logger = get_logger(__name__)


UTC = timezone.utc
_KEEPALIVE_SECONDS = 15
_IDEMPOTENCY_WINDOW = timedelta(minutes=5)


_BASE_CANDIDATES: list[dict[str, Any]] = [
    {
        "title": "Vendor observability guide for release automation",
        "url": "https://docs.vendor.example.com/platform/observability/automation",
        "reason": (
            "Offizielles Vendor-Handbuch mit Schritt-für-Schritt-Anleitung zur Telemetrie-"
            "Konfiguration. Enthält Governance-Hinweise und verweist auf API-Limits der"
            " aktuellen Version."
        ),
        "gap_tags": ["MONITORING_SURVEILLANCE"],
        "risk_flags": ["requires_authentication"],
        "source": "web",
        "domain": "docs.vendor.example.com",
        "detected_age_days": 42,
        "version_hint": "v12.4",
        "base_score": 86,
        "base_fused": 7.2,
    },
    {
        "title": "Change management obligations under IT-SiG 2.0",
        "url": "https://bundes-it.example.de/compliance/itsig/change-management",
        "reason": (
            "Behördliche Orientierungshilfe mit Fokus auf Mitbestimmung und"
            " Dokumentationspflichten. Ergänzt bestehende RAG-Daten um explizite"
            " Reporting-Anforderungen."
        ),
        "gap_tags": ["PROCEDURAL", "LOGGING_AUDIT"],
        "risk_flags": ["legal_review"],
        "source": "web",
        "domain": "bundes-it.example.de",
        "detected_age_days": 120,
        "version_hint": None,
        "base_score": 92,
        "base_fused": 9.5,
    },
    {
        "title": "Release process checklist (tenant knowledge base)",
        "url": "https://intranet.example.local/ops/release-checklist",
        "reason": (
            "Interne Betriebsdokumentation mit technischen Prüfschritten."
            " Deckt Deploy- und Rollback-Maßnahmen ab, ergänzt Monitoring-Tasks."
        ),
        "gap_tags": ["TECHNICAL"],
        "risk_flags": ["internal_only"],
        "source": "rag",
        "domain": "intranet.example.local",
        "detected_age_days": 5,
        "version_hint": "2024-Q4",
        "base_score": 78,
        "base_fused": 6.8,
    },
    {
        "title": "Federation API logging patterns",
        "url": "https://community.vendor.example.com/articles/federation-logging",
        "reason": (
            "Community-Beitrag mit Beispielen für strukturierte Audit-Logs."
            " Liefert ergänzende JSON-Snippets und bekannte Stolperfallen."
        ),
        "gap_tags": ["LOGGING_AUDIT"],
        "risk_flags": ["community_source"],
        "source": "web",
        "domain": "community.vendor.example.com",
        "detected_age_days": 15,
        "version_hint": None,
        "base_score": 67,
        "base_fused": 5.4,
    },
    {
        "title": "Monitoring plugin configuration reference",
        "url": "https://docs.vendor.example.com/plugins/monitoring/reference",
        "reason": (
            "Aktualisierte Referenz für das Monitoring-Plugin mit neuen Telemetrie-Events"
            " und Alarm-Regeln. Enthält Matrix zu Tenant-spezifischen Overrides."
        ),
        "gap_tags": ["MONITORING_SURVEILLANCE", "ANALYTICS_REPORTING"],
        "risk_flags": [],
        "source": "web",
        "domain": "docs.vendor.example.com",
        "detected_age_days": 18,
        "version_hint": "v3.2",
        "base_score": 84,
        "base_fused": 8.1,
    },
    {
        "title": "Audit evidence export automation",
        "url": "https://kb.example.com/audit/export-automation",
        "reason": (
            "Knowledge-Base-Artikel zu Export-Jobs inklusive RBAC-Hinweisen."
            " Ergänzt RAG-Bestand um fehlende RBAC-Fallstricke und Schedulings."
        ),
        "gap_tags": ["ACCESS_PRIVACY_SECURITY", "ANALYTICS_REPORTING"],
        "risk_flags": ["outdated_ui"],
        "source": "web",
        "domain": "kb.example.com",
        "detected_age_days": 400,
        "version_hint": "v2",
        "base_score": 61,
        "base_fused": 4.6,
    },
]


@dataclass
class SubmissionRecord:
    """Record of a previously accepted submission payload."""

    payload_fingerprint: str
    response: dict[str, Any]
    created_at: datetime


class DevHitlRun:
    """Mutable run state backing the developer HITL UI."""

    def __init__(self, run_id: str, payload: dict[str, Any]):
        self.run_id = run_id
        self.payload = payload
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._events: list[dict[str, Any]] = []
        self._submissions: dict[str, SubmissionRecord] = {}
        self._last_keepalive = time.monotonic()
        self._auto_approved = False
        self._coverage_totals = {
            "total": len(payload["top_k"]),
            "ingested": 0,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def deadline(self) -> datetime:
        return datetime.fromisoformat(
            self.payload["meta"]["deadline_utc"].replace("Z", "+00:00")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def serialize(self) -> dict[str, Any]:
        """Return a JSON-safe payload describing the run."""

        with self._lock:
            self._maybe_trigger_auto_approve_locked()
            data = json.loads(json.dumps(self.payload))
            data["meta"]["ingested_count"] = self._coverage_totals["ingested"]
            data["meta"]["total_candidates"] = self._coverage_totals["total"]
            data["meta"]["auto_approved"] = self._auto_approved
            return data

    def record_submission(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Persist a submission and emit synthetic events.

        Returns ``(response, is_new)`` where ``is_new`` indicates whether
        the payload was stored (False => idempotent replay).
        """

        fingerprint = json.dumps(payload, sort_keys=True)
        now = datetime.now(tz=UTC)

        with self._lock:
            self._maybe_trigger_auto_approve_locked()
            existing = self._submissions.get(fingerprint)
            if existing and now - existing.created_at <= _IDEMPOTENCY_WINDOW:
                logger.info(
                    "hitl.dev.idempotent",
                    run_id=self.run_id,
                    approved=len(payload.get("approved_ids", [])),
                    rejected=len(payload.get("rejected_ids", [])),
                    custom=len(payload.get("custom_urls", [])),
                )
                return existing.response, False

            tasks = self._build_ingestion_tasks(payload)
            response = {
                "ingestion_task_ids": tasks,
                "eta_minutes": 5,
            }
            self._submissions[fingerprint] = SubmissionRecord(
                payload_fingerprint=fingerprint,
                response=response,
                created_at=now,
            )
            logger.info(
                "hitl.dev.approval_received",
                run_id=self.run_id,
                counts={
                    "approved": len(payload.get("approved_ids", [])),
                    "rejected": len(payload.get("rejected_ids", [])),
                    "custom": len(payload.get("custom_urls", [])),
                },
            )
            self._append_submission_events_locked(tasks, payload)
            return response, True

    def stream_events(self):
        """Yield SSE events, injecting keep-alives when idle."""

        index = 0
        while True:
            with self._condition:
                self._maybe_trigger_auto_approve_locked()
                if len(self._events) <= index:
                    now = time.monotonic()
                    remaining = self._last_keepalive + _KEEPALIVE_SECONDS - now
                    if remaining <= 0:
                        keepalive = {
                            "type": "keepalive",
                            "payload": {"ts": datetime.now(tz=UTC).isoformat()},
                        }
                        self._events.append(keepalive)
                        self._last_keepalive = now
                        self._condition.notify_all()
                    else:
                        self._condition.wait(timeout=remaining)
                    continue

                events = self._events[index:]
                index = len(self._events)

            for event in events:
                yield event

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_event_locked(self, event_type: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("ts", datetime.now(tz=UTC).isoformat())
        event = {"type": event_type, "payload": payload}
        self._events.append(event)
        self._condition.notify_all()

    def _append_submission_events_locked(
        self, task_ids: list[str], payload: dict[str, Any]
    ) -> None:
        approved = payload.get("approved_ids", [])
        custom_urls = payload.get("custom_urls", [])

        for task_id, item_id in zip(task_ids, approved + custom_urls, strict=False):
            self._append_event_locked(
                "ingestion_update",
                {"task_id": task_id, "status": "queued", "url": item_id},
            )

        # Simulate progress updates asynchronously using a background thread.
        threading.Thread(
            target=self._simulate_progress,
            args=(task_ids, payload),
            name=f"dev-hitl-progress-{self.run_id}",
            daemon=True,
        ).start()

    def _simulate_progress(self, task_ids: list[str], payload: dict[str, Any]) -> None:
        time.sleep(1.0)
        with self._condition:
            for task_id in task_ids:
                self._append_event_locked(
                    "ingestion_update",
                    {"task_id": task_id, "status": "running"},
                )

        time.sleep(1.2)
        ingested_delta = max(
            len(payload.get("approved_ids", [])) + len(payload.get("custom_urls", [])),
            0,
        )
        with self._condition:
            self._coverage_totals["ingested"] = min(
                self._coverage_totals["total"],
                self._coverage_totals["ingested"] + ingested_delta,
            )
            for task_id in task_ids:
                self._append_event_locked(
                    "ingestion_update",
                    {"task_id": task_id, "status": "done"},
                )
            self._append_event_locked(
                "coverage_update",
                {
                    "facets_after": self.payload["coverage_delta"].get(
                        "facets_after", {}
                    ),
                    "ingested_count": self._coverage_totals["ingested"],
                    "total": self._coverage_totals["total"],
                },
            )

    def _build_ingestion_tasks(self, payload: dict[str, Any]) -> list[str]:
        identifiers = list(payload.get("approved_ids", []))
        identifiers.extend(payload.get("custom_urls", []))
        if not identifiers:
            identifiers.append("noop")
        return [f"task-{uuid4().hex[:8]}" for _ in identifiers]

    def _maybe_trigger_auto_approve_locked(self) -> None:
        if self._auto_approved:
            return
        now = datetime.now(tz=UTC)
        if now < self.deadline:
            return
        self._auto_approved = True
        logger.info("hitl.dev.auto_approve_triggered", run_id=self.run_id)
        self.payload["meta"]["auto_approved"] = True
        self._append_event_locked(
            "deadline_update",
            {
                "deadline_utc": self.payload["meta"]["deadline_utc"],
                "auto_approved": True,
            },
        )


class DevHitlStore:
    """Registry for synthetic HITL runs used by the dev UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, DevHitlRun] = {}
        self._default_run_id = "demo-run"

    def default_run_id(self) -> str:
        return self._default_run_id

    def get(self, run_id: str) -> DevHitlRun:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                run = self._create_run_locked(run_id)
                self._runs[run_id] = run
            return run

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_run_locked(self, run_id: str) -> DevHitlRun:
        rng = Random(int(sha256(run_id.encode("utf-8")).hexdigest(), 16))
        top_k: list[dict[str, Any]] = []

        for index, base in enumerate(_BASE_CANDIDATES, start=1):
            candidate = dict(base)
            candidate_id = f"{run_id}-cand-{index}"
            candidate.update(
                {
                    "id": candidate_id,
                    "candidate_id": candidate_id,
                    "score": min(100, max(0, base["base_score"] + rng.randint(-5, 5))),
                    "fused_score": round(
                        base["base_fused"] + rng.uniform(-0.7, 0.7), 3
                    ),
                    "reason": base["reason"][:280],
                    "gap_tags": list(base.get("gap_tags", [])),
                    "risk_flags": list(base.get("risk_flags", [])),
                    "detected_date": self._detected_date_for(base["detected_age_days"]),
                    "domain": base["domain"],
                }
            )
            top_k.append(candidate)

        top_k.sort(key=lambda item: item["fused_score"], reverse=True)

        now = datetime.now(tz=UTC)
        deadline = now + timedelta(minutes=45)

        coverage_before = {
            "TECHNICAL": 0.52,
            "MONITORING_SURVEILLANCE": 0.3,
            "LOGGING_AUDIT": 0.25,
            "ANALYTICS_REPORTING": 0.2,
        }
        coverage_after = {
            "TECHNICAL": 0.71,
            "MONITORING_SURVEILLANCE": 0.62,
            "LOGGING_AUDIT": 0.58,
            "ANALYTICS_REPORTING": 0.44,
        }

        payload = {
            "run_id": run_id,
            "top_k": top_k,
            "coverage_delta": {
                "summary": "Monitoring- und Audit-Facetten verbessern sich signifikant.",
                "facets_before": coverage_before,
                "facets_after": coverage_after,
            },
            "meta": {
                "tenant_id": "tenant-dev",
                "case_id": f"case-{run_id}",
                "deadline_utc": deadline.isoformat().replace("+00:00", "Z"),
                "min_diversity_buckets": 3,
                "freshness_mode": "software_docs_strict",
                "rag_unavailable": False,
                "llm_timeout": False,
                "cache_hit_rag": bool(rng.getrandbits(1)),
                "cache_hit_llm": bool(rng.getrandbits(1)),
            },
        }

        run = DevHitlRun(run_id, payload)
        with run._lock:  # type: ignore[attr-defined]
            run._append_event_locked(
                "coverage_update",
                {
                    "facets_after": coverage_after,
                    "ingested_count": 0,
                    "total": len(top_k),
                },
            )
            run._append_event_locked(
                "deadline_update",
                {
                    "deadline_utc": payload["meta"]["deadline_utc"],
                    "auto_approved": False,
                },
            )
        return run

    @staticmethod
    def _detected_date_for(age_days: int | None) -> str | None:
        if age_days is None:
            return None
        dt = datetime.now(tz=UTC) - timedelta(days=age_days)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


store = DevHitlStore()


__all__ = ["store", "DevHitlStore", "DevHitlRun"]
