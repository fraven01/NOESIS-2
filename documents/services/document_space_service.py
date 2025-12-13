from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from django.urls import reverse
from django_tenants.utils import schema_context
from structlog.stdlib import get_logger

from customers.models import Tenant
from documents.collection_service import CollectionService
from documents.models import DocumentCollection
from documents.repository import DocumentsRepository

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentSpaceRequest:
    requested_collection: str | None
    limit: int
    latest_only: bool
    cursor: str | None
    workflow_filter: str | None
    search_term: str


@dataclass
class DocumentSpaceResult:
    collections: list[dict[str, object]]
    selected_collection: dict[str, object] | None
    selected_collection_identifier: str
    documents: list[dict[str, object]]
    document_summary: dict[str, object]
    summaries: dict[str, list[dict[str, object]]]
    documents_error: str | None
    collection_warning: bool
    has_collections: bool
    next_cursor: str | None

    def as_context(self) -> dict[str, object]:
        return {
            "collections": self.collections,
            "selected_collection": self.selected_collection,
            "selected_collection_identifier": self.selected_collection_identifier,
            "documents": self.documents,
            "document_summary": self.document_summary,
            "summaries": self.summaries,
            "documents_error": self.documents_error,
            "collection_warning": self.collection_warning,
            "has_collections": self.has_collections,
            "next_cursor": self.next_cursor,
        }


class DocumentSpaceService:
    """Aggregate collection and document metadata for the document-space view."""

    def __init__(
        self,
        *,
        collection_service: CollectionService | None = None,
    ) -> None:
        self._collection_service = collection_service or CollectionService()

    def build_context(
        self,
        *,
        tenant_context: str,
        tenant_obj: Tenant | None,
        params: DocumentSpaceRequest,
        repository: DocumentsRepository,
    ) -> DocumentSpaceResult:
        tenant_schema = tenant_context
        with schema_context(tenant_schema):

            self._ensure_manual_collection(tenant_obj)
            collections = self._load_collections(tenant_obj, tenant_schema)
            serialized_collections = [
                self._serialize_collection(item) for item in collections
            ]

            selected_collection = self._match_collection_identifier(
                collections, params.requested_collection
            )
            collection_warning = bool(
                params.requested_collection and not selected_collection
            )
            requested_identifier = params.requested_collection or ""
            if selected_collection is None and collections:
                selected_collection = collections[0]
                requested_identifier = str(selected_collection.id)

            documents_payload: list[dict[str, object]] = []
            documents_error: str | None = None
            next_cursor: str | None = None

            if selected_collection:
                list_fn = (
                    repository.list_latest_by_collection
                    if params.latest_only
                    else repository.list_by_collection
                )
                try:
                    document_refs, next_cursor = list_fn(
                        tenant_id=tenant_schema,
                        collection_id=selected_collection.collection_id,
                        limit=params.limit,
                        cursor=params.cursor or None,
                        workflow_id=params.workflow_filter or None,
                    )
                except Exception as e:
                    logger.exception(
                        "document_space.list_failed",
                        extra={
                            "tenant_id": tenant_schema,
                            "collection_id": str(selected_collection.collection_id),
                        },
                    )
                    documents_error = f"Fehler beim Laden: {str(e)}"
                else:
                    fetched_docs = self._fetch_documents(
                        repository=repository,
                        tenant_id=tenant_schema,
                        document_refs=document_refs,
                        latest_only=params.latest_only,
                    )
                    for doc in fetched_docs:
                        payload = self._serialize_document_payload(doc)
                        documents_payload.append(payload)

        filtered_documents = self._filter_documents(
            documents_payload, params.search_term
        )
        summaries = self._summaries_for_documents(filtered_documents)
        document_summary = {
            "fetched": len(documents_payload),
            "displayed": len(filtered_documents),
            "limit": params.limit,
        }
        selected_payload = self._select_serialized_collection(
            serialized_collections, selected_collection
        )

        return DocumentSpaceResult(
            collections=serialized_collections,
            selected_collection=selected_payload,
            selected_collection_identifier=requested_identifier,
            documents=filtered_documents,
            document_summary=document_summary,
            summaries=summaries,
            documents_error=documents_error,
            collection_warning=collection_warning,
            has_collections=bool(collections),
            next_cursor=next_cursor,
        )

    def _ensure_manual_collection(self, tenant: Tenant | None) -> None:
        if tenant is None:
            return
        try:
            self._collection_service.ensure_manual_collection(tenant)
        except Exception:
            logger.warning(
                "document_space.ensure_manual_collection_failed",
                exc_info=True,
            )

    def _load_collections(
        self,
        tenant: Tenant | None,
        tenant_schema: str,
    ) -> list[DocumentCollection]:
        collections_qs = DocumentCollection.objects.select_related("case")
        if tenant is not None:
            collections_qs = collections_qs.filter(tenant=tenant)
        else:
            collections_qs = collections_qs.filter(tenant__schema_name=tenant_schema)
        return list(collections_qs.order_by("name", "created_at"))

    def _fetch_documents(
        self,
        *,
        repository: DocumentsRepository,
        tenant_id: str,
        document_refs: Sequence,
        latest_only: bool,
    ) -> list:
        logger.info(f"DEBUG: _fetch_documents called with {len(document_refs)} refs")
        fetched_docs = []
        for idx, ref in enumerate(document_refs):
            logger.info(
                f"DEBUG: Fetching ref {idx}: document_id={ref.document_id}, workflow_id={ref.workflow_id}, version={ref.version}"
            )
            try:
                doc = repository.get(
                    tenant_id=tenant_id,
                    document_id=ref.document_id,
                    version=ref.version,
                    prefer_latest=latest_only or ref.version is None,
                    workflow_id=ref.workflow_id,
                )
                if doc is None:
                    logger.warning(
                        f"DEBUG: repository.get returned None for document_id={ref.document_id}"
                    )
                else:
                    logger.info(
                        f"DEBUG: Successfully fetched document {ref.document_id}"
                    )
            except Exception as e:
                logger.warning(
                    "document_space.document_fetch_failed",
                    exc_info=True,
                    extra={
                        "tenant_id": tenant_id,
                        "document_id": str(getattr(ref, "document_id", "")),
                        "error": str(e),
                    },
                )
                continue
            if doc is None:
                continue
            fetched_docs.append(doc)
        logger.info(f"DEBUG: _fetch_documents returning {len(fetched_docs)} documents")
        return fetched_docs

    def _serialize_collection(
        self, collection: DocumentCollection
    ) -> dict[str, object]:
        metadata = collection.metadata or {}
        metadata_items = [
            {"key": str(key), "value": self._stringify_metadata_value(value)}
            for key, value in sorted(metadata.items(), key=lambda item: str(item[0]))
        ]
        case_obj = collection.case
        case_info = None
        if case_obj is not None:
            case_info = {
                "id": str(case_obj.id),
                "external_id": getattr(case_obj, "external_id", ""),
                "title": getattr(case_obj, "title", ""),
                "status": getattr(case_obj, "status", ""),
            }

        return {
            "id": str(collection.id),
            "name": collection.name,
            "key": collection.key,
            "collection_id": str(collection.collection_id),
            "type": collection.type or "",
            "visibility": collection.visibility or "",
            "metadata": metadata_items,
            "case": case_info,
            "created_at": collection.created_at,
            "updated_at": collection.updated_at,
            "selector": str(collection.id),
        }

    @staticmethod
    def _select_serialized_collection(
        serialized_collections: list[dict[str, object]],
        selected_collection: DocumentCollection | None,
    ) -> dict[str, object] | None:
        if not selected_collection:
            return None
        target_id = str(selected_collection.id)
        for entry in serialized_collections:
            if entry["id"] == target_id:
                return entry
        return None

    @staticmethod
    def _match_collection_identifier(
        collections: Sequence[DocumentCollection], requested: object
    ) -> DocumentCollection | None:
        if not requested:
            return None
        requested_text = str(requested).strip().lower()
        if not requested_text:
            return None
        for collection in collections:
            if str(collection.id).lower() == requested_text:
                return collection
            if str(collection.collection_id).lower() == requested_text:
                return collection
            if collection.key and collection.key.lower() == requested_text:
                return collection
        return None

    @staticmethod
    def _stringify_metadata_value(value: object) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(
                    value, ensure_ascii=False, sort_keys=True, default=str
                )
            except TypeError:
                return str(value)
        return str(value)

    @staticmethod
    def _describe_blob(blob) -> dict[str, object]:
        size_bytes = getattr(blob, "size", 0) or 0
        size_display = "0 B"
        if size_bytes > 0:
            if size_bytes < 1024:
                size_display = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_display = f"{size_bytes / 1024:.1f} KB"
            else:
                size_display = f"{size_bytes / (1024 * 1024):.1f} MB"

        return {
            "type": getattr(blob, "type", None),
            "size": size_bytes,
            "size_display": size_display,
            "sha256": getattr(blob, "sha256", None),
            "media_type": getattr(blob, "media_type", None),
            "uri": getattr(blob, "uri", None),
        }

    def _serialize_document_payload(
        self,
        doc,
    ) -> dict[str, object]:
        external_ref = getattr(doc.meta, "external_ref", None) or {}
        lifecycle_state = getattr(doc, "lifecycle_state", "")

        # Extract lifecycle metadata from document.metadata['lifecycle']
        lifecycle_meta = {}
        doc_metadata = getattr(doc, "metadata", None)
        if isinstance(doc_metadata, dict):
            lifecycle_meta = doc_metadata.get("lifecycle", {})
            if not isinstance(lifecycle_meta, dict):
                lifecycle_meta = {}

        # Read changed_at from lifecycle metadata or lifecycle_updated_at field
        changed_at = None
        changed_str = lifecycle_meta.get("changed_at")
        if changed_str:
            try:
                from datetime import datetime, timezone as dt_timezone

                parsed = datetime.fromisoformat(str(changed_str))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=dt_timezone.utc)
                changed_at = parsed
            except (ValueError, AttributeError):
                pass

        if changed_at is None:
            lifecycle_updated_at = getattr(doc, "lifecycle_updated_at", None)
            if lifecycle_updated_at:
                changed_at = lifecycle_updated_at

        ingestion_payload = {
            "state": lifecycle_state,
            "changed_at": changed_at,
            "trace_id": lifecycle_meta.get("trace_id", "") or "",
            "run_id": lifecycle_meta.get("run_id", "") or "",
            "ingestion_run_id": lifecycle_meta.get("ingestion_run_id", "") or "",
            "reason": lifecycle_meta.get("reason", "") or "",
            "policy_events": (
                list(lifecycle_meta.get("policy_events", []))
                if lifecycle_meta.get("policy_events")
                else []
            ),
        }

        # Enhanced title extraction with improved URL handling
        # Normalize None and empty string to ""
        title = (doc.meta.title or "").strip()

        # Try URL-based title extraction from origin_uri OR external_ref['url']
        # Support both crawler (origin_uri) and other sources (external_ref)
        url = None
        if doc.meta.origin_uri:
            url = doc.meta.origin_uri
        elif external_ref and external_ref.get("url"):
            url = external_ref.get("url")

        if not title and url:
            try:
                from urllib.parse import urlparse, unquote

                parsed = urlparse(url)
                domain = parsed.netloc or ""
                path = parsed.path.strip("/")

                if path:
                    # Split path into segments and decode URL encoding
                    segments = [unquote(s) for s in path.split("/") if s]

                    if segments:
                        # Get last meaningful segment (skip index.html)
                        candidate = segments[-1]

                        # If last segment is index.html or similar, use parent directory
                        if candidate.lower() in [
                            "index.html",
                            "index.htm",
                            "default.html",
                            "default.htm",
                        ]:
                            if len(segments) > 1:
                                candidate = segments[-2]
                            else:
                                # Use domain if only index.html
                                candidate = domain.replace("www.", "").split(".")[0]

                        # Remove file extensions for cleaner titles
                        if "." in candidate:
                            name_part = candidate.rsplit(".", 1)[0]
                            # Only use name part if it's substantial
                            if len(name_part) > 2:
                                candidate = name_part

                        # Clean up: replace separators with spaces
                        title = candidate.replace("_", " ").replace("-", " ")

                        # Smart capitalization: capitalize each word but preserve acronyms
                        words = title.split()
                        capitalized = []
                        for word in words:
                            # Keep short words (likely acronyms) uppercase if already uppercase
                            if len(word) <= 3 and word.isupper():
                                capitalized.append(word)
                            # Otherwise title case
                            else:
                                capitalized.append(word.capitalize())
                        title = " ".join(capitalized)

                if not title and domain:
                    # Fallback to domain name
                    domain_name = domain.replace("www.", "").split(".")[0]
                    title = domain_name.replace("-", " ").title()
            except Exception:
                pass

        if not title:
            title = "Untitled Document"

        # Enhanced blob description with better media_type
        blob_info = self._describe_blob(doc.blob)

        # Try to infer media type if missing
        if not blob_info.get("media_type"):
            source_url = external_ref.get("url", "")
            if source_url:
                if source_url.endswith(".html") or "wiki" in source_url.lower():
                    blob_info["media_type"] = "text/html"
                elif source_url.endswith(".pdf"):
                    blob_info["media_type"] = "application/pdf"

        # Extract URL for display
        origin_uri = doc.meta.origin_uri or external_ref.get("url", "")

        payload = {
            "document_id": str(doc.ref.document_id),
            "workflow_id": doc.ref.workflow_id,
            "version": doc.ref.version or "",
            "collection_id": (
                str(doc.ref.collection_id) if doc.ref.collection_id else ""
            ),
            "document_collection_id": (
                str(doc.meta.document_collection_id)
                if doc.meta.document_collection_id
                else ""
            ),
            "title": title,  # Enhanced title
            "language": doc.meta.language or "",
            "tags": list(doc.meta.tags or []),
            "origin_uri": origin_uri,  # Enhanced URL
            "external_ref_items": self._dict_items(external_ref),
            "external_provider": external_ref.get("provider"),
            "external_id": external_ref.get("external_id"),
            "created_at": doc.created_at,
            "source": doc.source or "",
            "checksum": doc.checksum,
            "lifecycle_state": doc.lifecycle_state,
            "blob": blob_info,  # Enhanced blob info
            "download_url": reverse("documents:download", args=[doc.ref.document_id]),
            "ingestion": ingestion_payload,
            "meta": {
                "crawl_timestamp": doc.meta.crawl_timestamp,
                "pipeline_config": doc.meta.pipeline_config or {},
                "parse_stats": doc.meta.parse_stats or {},
            },
            "assets": self._serialize_assets(doc),
        }
        payload["search_blob"] = self._build_search_blob(payload)
        return payload

    def _serialize_assets(self, doc) -> list[dict[str, object]]:
        """Serialize document assets for template display."""
        assets = getattr(doc, "assets", None) or []
        serialized = []
        for asset in assets:
            asset_ref = getattr(asset, "ref", None)
            if not asset_ref:
                continue

            asset_id = getattr(asset_ref, "asset_id", None)
            document_id = getattr(asset_ref, "document_id", None)
            if not asset_id or not document_id:
                continue

            blob = getattr(asset, "blob", None)
            blob_info = {}
            if blob:
                if isinstance(blob, dict):
                    blob_info = {
                        "type": blob.get("type", ""),
                        "size": blob.get("size", 0),
                        "uri": blob.get("uri", ""),
                    }
                else:
                    blob_info = {
                        "type": getattr(blob, "type", ""),
                        "size": getattr(blob, "size", 0),
                        "uri": getattr(blob, "uri", ""),
                    }

            media_type = getattr(asset, "media_type", "application/octet-stream")
            is_image = media_type.startswith("image/")

            serialized.append(
                {
                    "asset_id": str(asset_id),
                    "document_id": str(document_id),
                    "media_type": media_type,
                    "is_image": is_image,
                    "blob": blob_info,
                    "origin_uri": getattr(asset, "origin_uri", ""),
                    "text_description": getattr(asset, "text_description", ""),
                    "caption_source": getattr(asset, "caption_source", ""),
                    "caption_method": getattr(asset, "caption_method", ""),
                    "context_before": (getattr(asset, "context_before", "") or "")[
                        :100
                    ],
                    "context_after": (getattr(asset, "context_after", "") or "")[:100],
                    "serve_url": reverse(
                        "documents:asset_serve", args=[document_id, asset_id]
                    ),
                }
            )
        return serialized

    def _dict_items(self, mapping: Mapping[str, object] | None) -> list[dict[str, str]]:
        if not isinstance(mapping, Mapping):
            return []
        return [
            {"key": str(key), "value": self._stringify_metadata_value(value)}
            for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
        ]

    def _build_search_blob(self, payload: Mapping[str, object]) -> str:
        helpers = []
        ingestion = payload.get("ingestion", {}) if isinstance(payload, Mapping) else {}
        helpers.extend(
            [
                payload.get("document_id"),
                payload.get("title"),
                payload.get("workflow_id"),
                payload.get("version"),
                payload.get("collection_id"),
                payload.get("document_collection_id"),
                payload.get("origin_uri"),
                payload.get("language"),
                payload.get("source"),
                payload.get("external_provider"),
                payload.get("external_id"),
                ingestion.get("state") if isinstance(ingestion, Mapping) else None,
                ingestion.get("trace_id") if isinstance(ingestion, Mapping) else None,
                ingestion.get("run_id") if isinstance(ingestion, Mapping) else None,
                (
                    ingestion.get("ingestion_run_id")
                    if isinstance(ingestion, Mapping)
                    else None
                ),
            ]
        )
        helpers.extend(payload.get("tags", []))
        for item in payload.get("external_ref_items", []):
            helpers.append(item.get("value"))
        normalized_tokens = [
            str(value).strip().lower()
            for value in helpers
            if isinstance(value, str) and value.strip()
        ]
        return " ".join(normalized_tokens)

    @staticmethod
    def _filter_documents(
        documents: list[dict[str, object]], query: str
    ) -> list[dict[str, object]]:
        normalized = str(query or "").strip().lower()
        if not normalized:
            return documents
        tokens = [token for token in normalized.split() if token]
        if not tokens:
            return documents
        filtered: list[dict[str, object]] = []
        for doc in documents:
            search_blob = doc.get("search_blob", "")
            if not isinstance(search_blob, str):
                continue
            blob = search_blob.lower()
            if all(token in blob for token in tokens):
                filtered.append(doc)
        return filtered

    @staticmethod
    def _summaries_for_documents(
        documents: list[dict[str, object]],
    ) -> dict[str, list[dict[str, object]]]:
        source_counter: Counter[str] = Counter()
        lifecycle_counter: Counter[str] = Counter()
        for doc in documents:
            source_counter[doc.get("source") or ""] += 1
            ingestion = doc.get("ingestion", {}) if isinstance(doc, Mapping) else {}
            lifecycle_counter[ingestion.get("state") or ""] += 1

        def _serialize(counter: Counter[str]) -> list[dict[str, object]]:
            entries = []
            for key, count in counter.items():
                label = key or "unknown"
                entries.append({"label": label, "count": count})
            entries.sort(key=lambda item: item["label"])
            return entries

        return {
            "sources": _serialize(source_counter),
            "lifecycle": _serialize(lifecycle_counter),
        }
