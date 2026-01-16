"""Locust user classes for preparing multi-tenant load tests.

The implementation focuses on staging-ready defaults without executing
anything automatically.  Header contracts follow docs/api/reference.md
and can be overridden through environment variables.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Iterable, List

from locust import HttpUser, between, task


def _load_json_env(var_name: str, fallback: Any) -> Any:
    """Return JSON from environment with sensible fallback."""

    raw = os.getenv(var_name)
    if not raw:
        return fallback

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _load_list_env(var_name: str, fallback: Iterable[str]) -> List[str]:
    raw = os.getenv(var_name)
    if not raw:
        return list(fallback)
    return [item.strip() for item in raw.split(",") if item.strip()]


class TenantHttpUser(HttpUser):
    """Base class that injects tenant headers and idempotency handling."""

    abstract = True
    wait_time = between(
        float(os.getenv("LOCUST_WAIT_MIN", 0.5)), float(os.getenv("LOCUST_WAIT_MAX", 2))
    )

    tenant_schema = os.getenv("STAGING_TENANT_SCHEMA") or os.getenv("TENANT_SCHEMA")
    tenant_id = os.getenv("STAGING_TENANT_ID") or os.getenv("TENANT_ID")
    case_id = os.getenv("STAGING_CASE_ID") or os.getenv("CASE_ID")
    bearer_token = os.getenv("STAGING_BEARER_TOKEN") or os.getenv("BEARER_TOKEN")
    key_alias = os.getenv("STAGING_KEY_ALIAS") or os.getenv("X_KEY_ALIAS")
    idempotency_prefix = os.getenv("LOAD_IDEMPOTENCY_PREFIX", "locust-chaos")

    # Allows overriding host via LOCUST_BASE_URL if not passed on the command line.
    host = (
        os.getenv("LOCUST_BASE_URL")
        or os.getenv("STAGING_WEB_URL")
        or "http://localhost:8000"
    ).rstrip("/")

    def build_headers(self) -> Dict[str, str]:
        if not self.tenant_schema or not self.tenant_id or not self.case_id:
            raise RuntimeError(
                "Tenant headers missing – set STAGING_TENANT_SCHEMA/STAGING_TENANT_ID/STAGING_CASE_ID, "
                "or TENANT_SCHEMA/TENANT_ID/CASE_ID."
            )

        headers = {
            "Content-Type": "application/json",
            "X-Tenant-Schema": self.tenant_schema,
            "X-Tenant-ID": self.tenant_id,
            "X-Case-ID": self.case_id,
            "Idempotency-Key": f"{self.idempotency_prefix}-{uuid.uuid4()}",
        }

        if self.bearer_token:
            headers["Authorization"] = (
                self.bearer_token
                if self.bearer_token.startswith("Bearer ")
                else f"Bearer {self.bearer_token}"
            )
        if self.key_alias:
            headers["X-Key-Alias"] = self.key_alias

        return headers

    def post_json(
        self, path: str, payload: Dict[str, Any], *, name: str | None = None
    ) -> None:
        """Issue a POST request with tenant headers and minimal validation."""

        headers = self.build_headers()
        # locust's client joins host automatically; path should contain trailing slash.
        with self.client.post(
            path, json=payload, headers=headers, name=name or path, catch_response=True
        ) as response:
            if response.status_code >= 400:
                response.failure(
                    f"Unexpected status {response.status_code}: {response.text}"
                )
            else:
                response.success()


class RagQueryUser(TenantHttpUser):
    """Exercises POST /v1/ai/rag/query/ with representative payloads."""

    payload = _load_json_env(
        "LOCUST_RAG_PAYLOAD",
        {
            "question": "Welche Reisekosten gelten für Consultants?",
            "filters": {"doc_class": "policy", "process": "travel"},
        },
    )

    @task
    def run_rag_query(self) -> None:
        self.post_json(
            "/v1/ai/rag/query/",
            self.payload,
            name="POST /v1/ai/rag/query/",
        )


class IntakeUser(TenantHttpUser):
    """Exercises POST /v1/ai/intake/ for info intake flows."""

    payload = _load_json_env(
        "LOCUST_INTAKE_PAYLOAD",
        {
            "prompt": "Fasse das Kundenfeedback zusammen.",
            "metadata": {"channel": "email", "source": "locust"},
        },
    )

    @task
    def run_intake(self) -> None:
        self.post_json("/v1/ai/intake/", self.payload, name="POST /v1/ai/intake/")


class IngestionRunUser(TenantHttpUser):
    """Exercises POST /rag/ingestion/run/ to trigger ingestion pipelines."""

    document_ids = _load_list_env("LOCUST_INGESTION_DOCUMENT_IDS", ["doc_demo_001"])
    payload_template: Dict[str, Any] = _load_json_env(
        "LOCUST_INGESTION_PAYLOAD",
        {
            "document_ids": document_ids,
            "priority": os.getenv("LOCUST_INGESTION_PRIORITY", "normal"),
            "embedding_profile": os.getenv("LOCUST_INGESTION_PROFILE", "standard"),
        },
    )

    @task
    def run_ingestion(self) -> None:
        payload = dict(self.payload_template)
        # Ensure list is copied per request to avoid shared state mutation.
        payload["document_ids"] = list(self.document_ids)
        self.post_json("/rag/ingestion/run/", payload, name="POST /rag/ingestion/run/")


__all__ = ["RagQueryUser", "IntakeUser", "IngestionRunUser"]
