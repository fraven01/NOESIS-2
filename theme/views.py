import json
from pprint import pformat
from typing import Any
from uuid import uuid4

from django import forms as django_forms
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

from ai_core.ingestion_status import get_latest_ingestion_run
from ai_core.rag.routing_rules import (
    get_routing_table,
    is_collection_routing_enabled,
)
from ai_core.services import handle_document_upload, start_ingestion_run
from ai_core.views import RagQueryViewV1
from rest_framework.response import Response as DRFResponse
from rest_framework.test import APIRequestFactory
from structlog.stdlib import get_logger

from .forms import (
    RagIngestionForm,
    RagQueryForm,
    RagStatusForm,
    RagUploadForm,
)


logger = get_logger(__name__)


_request_factory = APIRequestFactory()


def home(request):
    """Render the homepage."""

    return render(request, "theme/home.html")


def _derive_tenant_context(request) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    host = request.get_host() or ""
    hostname = host.split(":", maxsplit=1)[0]

    tenant_id = hostname or "dev.localhost"

    tenant_schema = None
    if hostname and "." in hostname:
        candidate_schema = hostname.split(".", maxsplit=1)[0] or None
        if candidate_schema and any(char.isalpha() for char in candidate_schema):
            tenant_schema = candidate_schema

    if not tenant_schema:
        tenant_schema = getattr(settings, "DEFAULT_TENANT_SCHEMA", None) or "dev"

    return tenant_id, tenant_schema


def _build_meta(
    tenant_id: str,
    tenant_schema: str,
    case_id: str,
    trace_id: str,
    collection_id: str | None,
) -> dict[str, Any]:
    meta = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "case_id": case_id,
        "trace_id": trace_id,
    }
    if collection_id:
        meta["collection_id"] = collection_id
    return meta


def _style_form(form):  # type: ignore[no-untyped-def]
    base_class = (
        "block w-full rounded-md border border-slate-300 px-3 py-2 text-sm "
        "shadow-sm focus:border-slate-500 focus:outline-none focus:ring-1 focus:ring-slate-500"
    )
    for field in form.fields.values():
        widget = field.widget
        if isinstance(widget, django_forms.widgets.FileInput):
            widget.attrs.setdefault(
                "class",
                "block w-full text-sm text-slate-700 file:mr-4 file:rounded-md "
                "file:border-0 file:bg-slate-900 file:px-3 file:py-2 file:text-sm "
                "file:font-medium file:text-white hover:file:bg-slate-700",
            )
        elif isinstance(widget, django_forms.widgets.Textarea):
            widget.attrs.setdefault("class", base_class)
            widget.attrs.setdefault("rows", widget.attrs.get("rows", 3))
        elif isinstance(widget, (django_forms.widgets.Select, django_forms.widgets.Input)):
            widget.attrs.setdefault("class", base_class)
        else:
            widget.attrs.setdefault("class", base_class)
    return form


def _format_payload(payload: Any) -> str:
    if payload in (None, ""):
        return "–"
    if isinstance(payload, (dict, list)):
        try:
            return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    try:
        return pformat(payload, sort_dicts=False)
    except Exception:
        return str(payload)


def _normalise_response(response: Any) -> dict[str, Any]:
    status_code: int | None = None
    payload: Any = None

    if isinstance(response, DRFResponse):
        status_code = response.status_code
        payload = response.data
    elif isinstance(response, HttpResponse):
        status_code = response.status_code
        try:
            payload = json.loads(response.content)
        except Exception:
            payload = response.content.decode()
    else:
        payload = response

    formatted = _format_payload(payload)

    return {
        "status": status_code,
        "payload": payload,
        "formatted": formatted,
    }


def rag_tools(request):
    """Render the RAG operations workbench with server-side orchestration."""

    tenant_id, tenant_schema = _derive_tenant_context(request)
    default_case = request.GET.get("case") or "manual-workbench"

    collection_options: list[str] = []
    resolver_profile_hint: str | None = None
    resolver_collection_hint: str | None = None

    try:
        routing_table = get_routing_table()
    except Exception:
        logger.warning("rag_tools.routing_table.unavailable", exc_info=True)
        routing_table = None
    else:
        unique_collections = {
            rule.collection_id
            for rule in routing_table.rules
            if getattr(rule, "collection_id", None)
        }
        collection_options = sorted(
            value for value in unique_collections if isinstance(value, str)
        )

        try:
            resolution = routing_table.resolve_with_metadata(
                tenant=tenant_id,
                process=None,
                collection_id=None,
                workflow_id=None,
                doc_class=None,
            )
        except Exception:
            logger.info("rag_tools.profile_resolution.failed", exc_info=True)
        else:
            resolver_profile_hint = resolution.profile
            if resolution.rule and resolution.rule.collection_id:
                resolver_collection_hint = resolution.rule.collection_id

    upload_form = _style_form(
        RagUploadForm(
            initial={
                "case_id": default_case,
                "collection_id": resolver_collection_hint or "",
            }
        )
    )
    ingestion_form = _style_form(
        RagIngestionForm(
            initial={
                "case_id": default_case,
                "embedding_profile": getattr(
                    settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
                ),
                "collection_id": resolver_collection_hint or "",
            }
        )
    )
    status_form = _style_form(RagStatusForm(initial={"case_id": default_case}))
    query_form = _style_form(
        RagQueryForm(
            initial={
                "case_id": default_case,
                "collection_id": resolver_collection_hint or "",
            }
        )
    )
    query_form.fields["collection_id"].widget.attrs.setdefault(
        "list", "query-collection-options"
    )

    upload_result: dict[str, Any] | None = None
    ingestion_result: dict[str, Any] | None = None
    status_result: dict[str, Any] | None = None
    query_result: dict[str, Any] | None = None
    query_summary: dict[str, Any] | None = None

    if request.method == "POST":
        action = request.POST.get("action") or ""
        trace_id = uuid4().hex

        if action == "upload":
            upload_form = _style_form(RagUploadForm(request.POST, request.FILES))
            if upload_form.is_valid():
                collection_id = upload_form.cleaned_data.get("collection_id")
                meta = _build_meta(
                    tenant_id,
                    tenant_schema,
                    upload_form.cleaned_data["case_id"],
                    trace_id,
                    collection_id,
                )
                metadata = dict(upload_form.cleaned_data.get("metadata", {}))
                if collection_id and not metadata.get("collection_id"):
                    metadata["collection_id"] = collection_id
                metadata_raw = (
                    json.dumps(metadata)
                    if metadata
                    else ""
                )
                response = handle_document_upload(
                    upload_form.cleaned_data["file"],
                    metadata_raw,
                    meta,
                    uuid4().hex,
                )
                upload_result = _normalise_response(response)
            else:
                upload_result = {
                    "status": None,
                    "payload": {"errors": upload_form.errors},
                    "formatted": _format_payload({"errors": upload_form.errors}),
                }

        elif action == "ingestion":
            ingestion_form = _style_form(RagIngestionForm(request.POST))
            if ingestion_form.is_valid():
                collection_id = ingestion_form.cleaned_data.get("collection_id")
                meta = _build_meta(
                    tenant_id,
                    tenant_schema,
                    ingestion_form.cleaned_data["case_id"],
                    trace_id,
                    collection_id,
                )
                payload = {
                    "document_ids": ingestion_form.cleaned_data["document_ids"],
                    "embedding_profile": ingestion_form.cleaned_data["embedding_profile"],
                }
                if collection_id:
                    payload["collection_id"] = collection_id
                response = start_ingestion_run(payload, meta, uuid4().hex)
                ingestion_result = _normalise_response(response)
            else:
                ingestion_result = {
                    "status": None,
                    "payload": {"errors": ingestion_form.errors},
                    "formatted": _format_payload({"errors": ingestion_form.errors}),
                }

        elif action == "status":
            status_form = _style_form(RagStatusForm(request.POST))
            if status_form.is_valid():
                case_id = status_form.cleaned_data["case_id"]
                status_payload = get_latest_ingestion_run(tenant_id, case_id)
                if status_payload is None:
                    status_result = {
                        "status": 404,
                        "payload": {
                            "detail": "No ingestion runs recorded for the current tenant/case.",
                        },
                        "formatted": _format_payload(
                            {
                                "detail": "No ingestion runs recorded for the current tenant/case.",
                            }
                        ),
                    }
                else:
                    status_result = _normalise_response(status_payload)
                    status_result["status"] = status_result.get("status") or 200
            else:
                status_result = {
                    "status": None,
                    "payload": {"errors": status_form.errors},
                    "formatted": _format_payload({"errors": status_form.errors}),
                }

        elif action == "query":
            query_form = _style_form(RagQueryForm(request.POST))
            query_form.fields["collection_id"].widget.attrs.setdefault(
                "list", "query-collection-options"
            )
            if query_form.is_valid():
                case_id = query_form.cleaned_data["case_id"]
                collection_id = query_form.cleaned_data.get("collection_id")
                payload = query_form.build_payload()
                headers = {
                    "HTTP_X_TENANT_ID": tenant_id,
                    "HTTP_X_CASE_ID": case_id,
                    "HTTP_X_TRACE_ID": trace_id,
                }
                if tenant_schema:
                    headers["HTTP_X_TENANT_SCHEMA"] = tenant_schema
                if collection_id:
                    headers["HTTP_X_COLLECTION_ID"] = collection_id

                drf_request = _request_factory.post(
                    "/ai/rag/query/",
                    payload,
                    format="json",
                    **headers,
                )
                response = RagQueryViewV1.as_view()(drf_request)
                query_result = _normalise_response(response)

                payload_obj = query_result.get("payload")
                if (
                    isinstance(query_result.get("status"), int)
                    and 200 <= query_result["status"] < 300
                    and isinstance(payload_obj, dict)
                ):
                    query_summary = {
                        "answer": payload_obj.get("answer"),
                        "prompt_version": payload_obj.get("prompt_version"),
                        "retrieval": payload_obj.get("retrieval"),
                        "snippets": payload_obj.get("snippets"),
                    }
            else:
                query_result = {
                    "status": None,
                    "payload": {"errors": query_form.errors},
                    "formatted": _format_payload({"errors": query_form.errors}),
                }

    return render(
        request,
        "theme/rag_tools.html",
        {
            "tenant_id": tenant_id,
            "tenant_schema": tenant_schema,
            "default_embedding_profile": getattr(
                settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
            ),
            "collection_options": collection_options,
            "collection_alias_enabled": is_collection_routing_enabled(),
            "resolver_profile_hint": resolver_profile_hint,
            "resolver_collection_hint": resolver_collection_hint,
            "upload_form": upload_form,
            "ingestion_form": ingestion_form,
            "status_form": status_form,
            "query_form": query_form,
            "upload_result": upload_result,
            "ingestion_result": ingestion_result,
            "status_result": status_result,
            "query_result": query_result,
            "query_summary": query_summary,
        },
    )
