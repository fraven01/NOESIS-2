from __future__ import annotations

import time
from uuid import uuid4

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django_tenants.utils import schema_context
from psycopg2 import sql

from ai_core.infra.reembed_progress import (
    increment_reembed_progress,
    init_reembed_progress,
    reserve_reembed_delay,
)
from ai_core.ingestion import process_document
from ai_core.contracts.business import BusinessContext
from ai_core.ids.http_scope import normalize_task_context
from ai_core.rag import vector_client
from ai_core.rag.embedding_config import build_embedding_model_version
from ai_core.rag.ingestion_contracts import resolve_ingestion_profile
from ai_core.rag.vector_schema import ensure_vector_space_schema
from ai_core.tools import InputError
from customers.models import Tenant
from documents.models import Document

REEMBED_CHUNK_RATE_LIMIT = 1000
REEMBED_RATE_KEY_PREFIX = "reembed:ratelimit"
REEMBED_PROGRESS_PREFIX = "reembed:progress"


def _iter_batches(items: list[object], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _coerce_text(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    try:
        return str(value).strip() or None
    except Exception:
        return None


def _build_task_meta(
    *,
    tenant_id: str,
    case_id: str | None,
    trace_id: str | None,
    tenant_schema: str | None,
) -> dict[str, object]:
    scope = normalize_task_context(
        tenant_id=tenant_id,
        case_id=case_id,
        service_id="celery-ingestion-worker",
        trace_id=trace_id,
        invocation_id=uuid4().hex,
        run_id=uuid4().hex,
        tenant_schema=tenant_schema,
    )
    business = BusinessContext(case_id=case_id)
    tool_context = scope.to_tool_context(business=business)
    return {
        "scope_context": scope.model_dump(mode="json", exclude_none=True),
        "business_context": business.model_dump(mode="json", exclude_none=True),
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
    }


def _fetch_chunk_counts(
    schema_name: str, tenant_id: str, document_ids: list[object]
) -> dict[object, int]:
    if not document_ids:
        return {}
    counts: dict[object, int] = {}
    with connection.cursor() as cur:
        for batch in _iter_batches(document_ids, 1000):
            cur.execute(
                sql.SQL(
                    """
                    SELECT document_id, COUNT(*)
                    FROM {}.chunks
                    WHERE document_id = ANY(%s)
                      AND (tenant_id = %s OR tenant_id IS NULL)
                    GROUP BY document_id
                    """
                ).format(sql.Identifier(schema_name)),
                (batch, tenant_id),
            )
            for row in cur.fetchall():
                counts[row[0]] = int(row[1])
    return counts


class Command(BaseCommand):
    help = "Queue background re-embedding runs for existing documents."

    def add_arguments(self, parser):
        parser.add_argument(
            "--tenant",
            type=str,
            help="Tenant schema name (default: all tenants)",
        )
        parser.add_argument(
            "--case-id",
            type=str,
            help="Optional case_id filter",
        )
        parser.add_argument(
            "--embedding-profile",
            type=str,
            help="Embedding profile to use (default: RAG_DEFAULT_EMBEDDING_PROFILE)",
        )

    def handle(self, *args, **options):
        tenant_filter = options.get("tenant")
        case_filter = options.get("case_id")
        profile_option = options.get("embedding_profile")

        if tenant_filter is not None:
            tenant_filter = str(tenant_filter).strip() or None
        if case_filter is not None:
            case_filter = str(case_filter).strip() or None

        profile_id = (
            str(profile_option).strip()
            if profile_option
            else str(getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "")).strip()
        )
        if not profile_id:
            raise CommandError("Embedding profile is required; set --embedding-profile")

        try:
            profile_binding = resolve_ingestion_profile(profile_id)
        except InputError as exc:
            raise CommandError(str(exc)) from exc

        resolved_profile_id = profile_binding.profile_id
        embedding_model_version = build_embedding_model_version(
            profile_binding.resolution.profile
        )

        ensure_vector_space_schema(profile_binding.resolution.vector_space)

        if tenant_filter:
            tenants = Tenant.objects.filter(schema_name=tenant_filter)
            if not tenants.exists():
                raise CommandError(f"Tenant '{tenant_filter}' not found")
        else:
            tenants = Tenant.objects.all()

        source_schema = vector_client.get_default_schema()
        run_id = uuid4().hex

        for tenant in tenants:
            with schema_context(tenant.schema_name):
                docs_query = Document.objects.filter(tenant=tenant)
                if case_filter:
                    docs_query = docs_query.filter(case_id=case_filter)
                docs = list(docs_query.values_list("id", "case_id", "trace_id"))

            if not docs:
                self.stdout.write(
                    self.style.WARNING(
                        f"No documents found for tenant={tenant.schema_name}"
                    )
                )
                continue

            document_ids = [doc_id for doc_id, _, _ in docs]
            chunk_counts = _fetch_chunk_counts(
                source_schema, str(tenant.id), document_ids
            )
            total_chunks = sum(
                max(1, chunk_counts.get(doc_id, 0)) for doc_id in document_ids
            )
            progress_key = f"{REEMBED_PROGRESS_PREFIX}:{tenant.schema_name}:{run_id}"
            init_reembed_progress(
                progress_key,
                total_documents=len(document_ids),
                total_chunks=total_chunks,
                embedding_profile=resolved_profile_id,
                model_version=embedding_model_version,
                metadata={"tenant_schema": tenant.schema_name, "case_id": case_filter},
            )

            rate_key = f"{REEMBED_RATE_KEY_PREFIX}:{tenant.schema_name}"
            queued_docs = 0
            queued_chunks = 0

            for document_id, document_case, trace_id in docs:
                chunk_count = max(1, chunk_counts.get(document_id, 0))
                delay_seconds = reserve_reembed_delay(
                    rate_key,
                    chunk_count=chunk_count,
                    quota_per_minute=REEMBED_CHUNK_RATE_LIMIT,
                )
                case_value = case_filter if case_filter is not None else document_case
                if case_value == "":
                    case_value = None
                trace_value = _coerce_text(trace_id)
                meta = _build_task_meta(
                    tenant_id=tenant.schema_name,
                    case_id=case_value,
                    trace_id=trace_value,
                    tenant_schema=tenant.schema_name,
                )
                state = {
                    "tenant_id": tenant.schema_name,
                    "case_id": case_value,
                    "document_id": str(document_id),
                    "embedding_profile": resolved_profile_id,
                    "tenant_schema": tenant.schema_name,
                    "trace_id": trace_value,
                }
                process_document.apply_async(
                    kwargs={
                        "state": state,
                        "meta": meta,
                        "reembed_progress_key": progress_key,
                    },
                    queue="ingestion-bulk",
                    countdown=max(0.0, float(delay_seconds)),
                )
                queued_docs += 1
                queued_chunks += chunk_count
                increment_reembed_progress(
                    progress_key,
                    queued_documents=1,
                    queued_chunks=chunk_count,
                )

            self.stdout.write(
                self.style.SUCCESS(
                    (
                        "Queued re-embedding for tenant=%s documents=%d "
                        "chunks=%d progress_key=%s"
                    )
                    % (tenant.schema_name, queued_docs, queued_chunks, progress_key)
                )
            )

            # Small delay to avoid throttling across tenants in tight loops.
            time.sleep(0.05)
