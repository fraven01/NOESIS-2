from __future__ import annotations

import time
import json
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from contextlib import nullcontext
from uuid import UUID

from celery import group, shared_task
from celery.exceptions import TimeoutError as CeleryTimeoutError
from common.celery import ScopedTask
from common.logging import get_logger
from django.utils import timezone
from django.conf import settings

from documents import (
    DocumentPipelineConfig,
    DocxDocumentParser,
    HtmlDocumentParser,
    ImageDocumentParser,
    MarkdownDocumentParser,
    ParserDispatcher,
    ParserRegistry,
    PdfDocumentParser,
    PptxDocumentParser,
    ParsedResult,
    ParsedTextBlock,
)

from . import tasks as pipe
from .case_events import emit_ingestion_case_event
from .infra import object_store
from .ingestion_status import (
    mark_ingestion_run_completed,
    mark_ingestion_run_running,
)
from ai_core.tools import InputError
from cases.models import Case

from .rag.ingestion_contracts import resolve_ingestion_profile
from .rag.vector_schema import ensure_vector_space_schema

log = get_logger(__name__)


def _load_case_observability_context(
    tenant_id: str, case_id: str | None
) -> dict[str, str]:
    if not tenant_id or not case_id:
        return {}
    for filter_kwargs in (
        {"tenant_id": tenant_id, "external_id": case_id},
        {"tenant__schema_name": tenant_id, "external_id": case_id},
    ):
        try:
            record = (
                Case.objects.filter(**filter_kwargs).values("status", "phase").first()
            )
        except Exception:
            continue
        if record:
            return dict(record)
    return {}


def _meta_store_path(tenant: str, case: str | None, document_id: str) -> str:
    return "/".join(
        (
            object_store.sanitize_identifier(tenant),
            object_store.sanitize_identifier(case or "upload"),
            "uploads",
            f"{document_id}.meta.json",
        )
    )


def _status_store_path(tenant: str, case: str | None, document_id: str) -> str:
    return "/".join(
        (
            object_store.sanitize_identifier(tenant),
            object_store.sanitize_identifier(case or "upload"),
            "uploads",
            f"{document_id}.status.json",
        )
    )


def _load_pipeline_state(
    tenant: str, case: str | None, document_id: str
) -> Dict[str, object]:
    status_path = _status_store_path(tenant, case, document_id)
    try:
        raw_state = object_store.read_json(status_path)
    except (json.decoder.JSONDecodeError, ValueError):
        return {"steps": {}, "attempts": 0}
    except FileNotFoundError:
        return {
            "steps": {},
            "attempts": 0,
        }
    if not isinstance(raw_state, dict):
        return {"steps": {}, "attempts": 0}
    raw_state.setdefault("steps", {})
    if not isinstance(raw_state["steps"], dict):
        raw_state["steps"] = {}
    return raw_state


def _write_pipeline_state(
    tenant: str, case: str | None, document_id: str, state: Dict[str, object]
) -> None:
    status_path = _status_store_path(tenant, case, document_id)
    serialized: Dict[str, object] = {}
    for key, value in state.items():
        if key == "steps" and isinstance(value, dict):
            normalized_steps: Dict[str, object] = {}
            for step_name, step_data in value.items():
                if not isinstance(step_data, dict):
                    continue
                normalized: Dict[str, object] = {}
                for field, field_value in step_data.items():
                    if field == "path" and field_value:
                        normalized[field] = str(field_value)
                    elif isinstance(field_value, Path):
                        normalized[field] = str(field_value)
                    else:
                        normalized[field] = field_value
                normalized_steps[step_name] = normalized
            serialized[key] = normalized_steps
        else:
            serialized[key] = value
    object_store.write_json(status_path, serialized)


def _normalize_step_result(result: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in result.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def _ensure_step(
    tenant: str,
    case: str | None,
    document_id: str,
    state: Dict[str, object],
    step_name: str,
    run_step: Callable[[], Dict[str, object]],
) -> tuple[Dict[str, object], bool]:
    steps = state.setdefault("steps", {})
    cached = steps.get(step_name)
    if isinstance(cached, dict):
        cached_path = cached.get("path")
        if cached_path:
            if (object_store.BASE_PATH / str(cached_path)).exists():
                cached_result = {
                    key: cached[key]
                    for key in cached
                    if key not in {"completed_at", "cleaned"}
                }
                return cached_result, True
        elif "path" not in cached:
            cached_result = {
                key: cached[key]
                for key in cached
                if key not in {"completed_at", "cleaned"}
            }
            return cached_result, True

    result = _normalize_step_result(run_step())
    steps[step_name] = {
        **result,
        "completed_at": time.time(),
        "cleaned": False,
    }
    _write_pipeline_state(tenant, case, document_id, state)
    return result, False


def _normalize_object_store_relative_path(path: str, base_path: Path) -> Path:
    """Validate *path* is within the object store and return the normalized value."""

    candidate = Path(path)
    if candidate.is_absolute() or candidate.anchor:
        raise ValueError("payload_path_invalid")

    try:
        resolved = (base_path / candidate).resolve()
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("payload_path_invalid") from exc

    if not resolved.is_relative_to(base_path):
        raise ValueError("payload_path_invalid")

    try:
        relative_path = resolved.relative_to(base_path)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("payload_path_invalid") from exc

    if str(relative_path) == ".":
        raise ValueError("payload_path_invalid")

    return relative_path


def _cleanup_artifacts(paths: Iterable[Optional[str]]) -> List[str]:
    removed: List[str] = []
    seen: set[str] = set()
    base_path = object_store.BASE_PATH.resolve()
    for path in paths:
        if not path:
            continue

        candidate = str(path).strip()
        if not candidate:
            continue

        try:
            normalized_path = _normalize_object_store_relative_path(
                candidate, base_path
            )
        except ValueError:
            log.warning(
                "Skipping invalid object store path during cleanup",
                extra={"path": candidate},
            )
            continue

        normalized_key = normalized_path.as_posix()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)

        target = base_path / normalized_path
        try:
            if target.exists():
                target.unlink()
                removed.append(normalized_key)
        except Exception:
            log.exception("Failed to cleanup artifact", extra={"path": normalized_key})
    return removed


def cleanup_raw_payload_artifact(raw_payload_path: Optional[str]) -> List[str]:
    """Remove the persisted raw payload, returning the removed paths."""

    if not raw_payload_path:
        return []
    return _cleanup_artifacts([raw_payload_path])


def _mark_cleaned(
    tenant: str,
    case: str | None,
    document_id: str,
    state: Dict[str, object],
    removed_paths: Iterable[str],
) -> None:
    removed_set = {str(path) for path in removed_paths}
    steps = state.setdefault("steps", {})
    updated = False
    for step_name, step_data in list(steps.items()):
        if not isinstance(step_data, dict):
            continue
        if step_data.get("path") in removed_set:
            step_data["path"] = None
            step_data["cleaned"] = True
            updated = True
    if updated:
        _write_pipeline_state(tenant, case, document_id, state)


_CONFIG_FIELD_NAMES = {
    "pdf_safe_mode",
    "caption_min_confidence_default",
    "caption_min_confidence_by_collection",
    "enable_ocr",
    "enable_notes_in_pptx",
    "emit_empty_slides",
    "enable_asset_captions",
    "ocr_fallback_confidence",
    "use_readability_html_extraction",
    "ocr_renderer",
}


def _extract_pipeline_config_overrides(
    source: Mapping[str, object] | None,
    *,
    source_name: str,
) -> Dict[str, object]:
    """Return validated pipeline config overrides from ``source``.

    ``source`` may contain an optional ``pipeline_config`` mapping. Only keys that
    are present in :data:`_CONFIG_FIELD_NAMES` are accepted. Any malformed input
    results in :class:`InputError`.
    """

    if not isinstance(source, Mapping):
        return {}

    candidate = source.get("pipeline_config")
    if candidate is None:
        return {}
    if not isinstance(candidate, Mapping):
        raise InputError(
            "invalid_pipeline_config_override",
            "Pipeline config overrides must be provided as an object.",
            context={"source": source_name, "received_type": type(candidate).__name__},
        )

    invalid_keys = [key for key in candidate.keys() if key not in _CONFIG_FIELD_NAMES]
    if invalid_keys:
        raise InputError(
            "invalid_pipeline_config_override",
            "Unsupported pipeline config overrides provided.",
            context={
                "source": source_name,
                "invalid_keys": sorted(str(key) for key in invalid_keys),
            },
        )

    overrides: Dict[str, object] = {}
    for key in candidate:
        if not isinstance(key, str):
            raise InputError(
                "invalid_pipeline_config_override",
                "Pipeline config override keys must be strings.",
                context={"source": source_name, "invalid_key_type": type(key).__name__},
            )
        overrides[key] = candidate[key]
    return overrides


def _build_document_pipeline_config(
    *,
    state: Mapping[str, object] | None = None,
    meta: Mapping[str, object] | None = None,
) -> DocumentPipelineConfig:
    """Build a :class:`DocumentPipelineConfig` instance.

    The resulting configuration respects global Django settings overrides and
    optional runtime overrides supplied via ``state`` or ``meta``. Runtime
    overrides must be provided under a ``pipeline_config`` mapping and may only
    contain keys declared in :data:`_CONFIG_FIELD_NAMES`.
    """

    config_kwargs: Dict[str, object] = {}

    mapping = {
        "pdf_safe_mode": "DOCUMENT_PIPELINE_PDF_SAFE_MODE",
        "caption_min_confidence_default": "DOCUMENT_PIPELINE_CAPTION_MIN_CONFIDENCE_DEFAULT",
        "caption_min_confidence_by_collection": "DOCUMENT_PIPELINE_CAPTION_MIN_CONFIDENCE_BY_COLLECTION",
        "enable_ocr": "DOCUMENT_PIPELINE_ENABLE_OCR",
        "enable_notes_in_pptx": "DOCUMENT_PIPELINE_ENABLE_NOTES_IN_PPTX",
        "emit_empty_slides": "DOCUMENT_PIPELINE_EMIT_EMPTY_SLIDES",
        "enable_asset_captions": "DOCUMENT_PIPELINE_ENABLE_ASSET_CAPTIONS",
        "ocr_fallback_confidence": "DOCUMENT_PIPELINE_OCR_FALLBACK_CONFIDENCE",
        "use_readability_html_extraction": "DOCUMENT_PIPELINE_USE_READABILITY_HTML_EXTRACTION",
        "ocr_renderer": "DOCUMENT_PIPELINE_OCR_RENDERER",
    }

    for field, setting_name in mapping.items():
        if hasattr(settings, setting_name):
            value = getattr(settings, setting_name)
            if value is not None:
                config_kwargs[field] = value

    overrides = getattr(settings, "DOCUMENT_PIPELINE_CONFIG", None)
    if isinstance(overrides, Mapping):
        for field in _CONFIG_FIELD_NAMES:
            if field in overrides:
                config_kwargs[field] = overrides[field]

    runtime_overrides: Dict[str, object] = {}
    runtime_overrides.update(
        _extract_pipeline_config_overrides(state, source_name="state")
    )
    if isinstance(state, Mapping):
        runtime_overrides.update(
            _extract_pipeline_config_overrides(
                state.get("meta"), source_name="state.meta"
            )
        )
    runtime_overrides.update(
        _extract_pipeline_config_overrides(meta, source_name="meta")
    )

    if runtime_overrides:
        config_kwargs.update(runtime_overrides)

    return DocumentPipelineConfig(**config_kwargs)


def _build_parser_dispatcher() -> ParserDispatcher:
    """Return a dispatcher with the configured parser order."""

    registry = ParserRegistry()
    registry.register(MarkdownDocumentParser())
    registry.register(HtmlDocumentParser())
    registry.register(DocxDocumentParser())
    registry.register(PptxDocumentParser())
    registry.register(PdfDocumentParser())
    registry.register(ImageDocumentParser())
    text_parser_cls = getattr(
        import_module("documents.parsers_text"), "TextDocumentParser"
    )
    registry.register(text_parser_cls())
    return ParserDispatcher(registry)


def _render_parsed_text(parsed: ParsedResult) -> str:
    """Flatten parsed text blocks into a single string for downstream tasks."""

    if parsed.text_blocks:
        return "\n\n".join(block.text for block in parsed.text_blocks)
    return ""


def _parsed_text_path(tenant: str, case: str | None, document_identifier: UUID) -> str:
    tenant_segment = object_store.sanitize_identifier(str(tenant))
    case_segment = object_store.sanitize_identifier(case or "upload")
    document_segment = object_store.sanitize_identifier(str(document_identifier))
    return "/".join(
        [tenant_segment, case_segment, "text", f"{document_segment}.parsed.txt"]
    )


def _parsed_blocks_path(
    tenant: str, case: str | None, document_identifier: UUID
) -> str:
    tenant_segment = object_store.sanitize_identifier(str(tenant))
    case_segment = object_store.sanitize_identifier(case or "upload")
    document_segment = object_store.sanitize_identifier(str(document_identifier))
    return "/".join(
        [tenant_segment, case_segment, "text", f"{document_segment}.parsed.json"]
    )


def _serialise_text_block(block: ParsedTextBlock) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "text": block.text,
        "kind": str(block.kind),
        "section_path": list(block.section_path) if block.section_path else None,
        "page_index": block.page_index,
        "table_meta": dict(block.table_meta) if block.table_meta is not None else None,
        "language": block.language,
    }
    metadata = getattr(block, "metadata", None)
    if isinstance(metadata, Mapping):
        parent_ref = metadata.get("parent_ref")
        if parent_ref:
            payload["parent_ref"] = parent_ref
        locator = metadata.get("locator")
        if locator:
            payload["locator"] = locator
    return payload


def _persist_parsed_text(
    tenant: str,
    case: str | None,
    document_identifier: UUID,
    parsed: ParsedResult,
) -> Dict[str, object]:
    text_content = _render_parsed_text(parsed)
    text_path = _parsed_text_path(tenant, case, document_identifier)
    object_store.put_bytes(text_path, text_content.encode("utf-8"))

    blocks_path = _parsed_blocks_path(tenant, case, document_identifier)
    serialised_blocks = [
        {"index": index, **_serialise_text_block(block)}
        for index, block in enumerate(parsed.text_blocks)
    ]
    payload = {
        "blocks": serialised_blocks,
        "statistics": dict(parsed.statistics),
        "text": text_content,
    }
    object_store.write_json(blocks_path, payload)

    return {
        "path": text_path,
        "text_path": text_path,
        "blocks_path": blocks_path,
        "statistics": dict(parsed.statistics),
    }


def _extract_blob_payload_bytes(document: object) -> bytes | None:
    blob = getattr(document, "blob", None)
    if hasattr(blob, "decoded_payload"):
        try:
            payload = blob.decoded_payload()
        except Exception:  # pragma: no cover - defensive guard
            return None
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
    payload_attr = getattr(blob, "payload", None)
    if isinstance(payload_attr, (bytes, bytearray)):
        return bytes(payload_attr)
    return None


def _extract_external_id(document: object) -> str | None:
    meta = getattr(document, "meta", None)
    external_ref = getattr(meta, "external_ref", None)
    if isinstance(external_ref, Mapping):
        candidate = external_ref.get("external_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _extract_meta_media_type(document: object) -> str | None:
    media_type = getattr(document, "media_type", None)
    if isinstance(media_type, str) and media_type.strip():
        return media_type.strip().lower()
    return None


def _hydrate_blob_payload(repository: object, document: object) -> bytes | None:
    blob = getattr(document, "blob", None)
    if blob is None:
        return None
    payload = _extract_blob_payload_bytes(document)
    if payload is not None:
        return payload
    uri = getattr(blob, "uri", None)
    storage = getattr(repository, "_storage", None)
    if uri and storage is not None:
        try:
            payload = storage.get(uri)  # type: ignore[call-arg]
        except Exception:
            return None
        object.__setattr__(blob, "payload", payload)
        return payload
    return None


def partition_document_ids(
    tenant: str, case: str, document_ids: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """Split document identifiers into existing and missing uploads."""

    existing: List[str] = []
    missing: List[str] = []

    services = import_module("ai_core.services")
    repository = services._get_documents_repository()  # type: ignore[attr-defined]

    for document_id in document_ids:
        try:
            document_uuid = UUID(str(document_id))
        except (TypeError, ValueError, AttributeError):
            missing.append(document_id)
            continue

        document = repository.get(tenant, document_uuid, prefer_latest=True)
        if document is None:
            missing.append(document_id)
        else:
            existing.append(document_id)

    return existing, missing


@shared_task(
    base=ScopedTask,
    queue="ingestion",
    bind=True,
    max_retries=3,
    accepts_scope=True,
)
def process_document(
    self,
    tenant: str,
    case: str | None,
    document_id: str,
    embedding_profile: str,
    tenant_schema: Optional[str] = None,
    trace_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, object]:
    kwargs.pop("session_salt", None)
    started = time.perf_counter()

    # Graph Migration: Load minimal state primarily for error tracking
    state = _load_pipeline_state(tenant, case, document_id)
    state["attempts"] = int(state.get("attempts", 0)) + 1
    state["last_attempt_started_at"] = time.time()
    _write_pipeline_state(tenant, case, document_id, state)

    # Ensure correct tenant schema context for DB operations
    from django.db import connection
    from django_tenants.utils import schema_context
    from ai_core.services import get_document_components
    from documents.processing_graph import (
        build_document_processing_graph,
        DocumentProcessingState,
    )
    from documents.pipeline import DocumentPipelineConfig, DocumentProcessingContext
    from ai_core.api import (
        trigger_embedding,
        decide_delta,
        enforce_guardrails,
    )

    target_schema = tenant_schema or tenant

    try:
        with schema_context(target_schema) if target_schema else nullcontext():
            log.info(
                "process_document graph execution started",
                extra={
                    "tenant": tenant,
                    "tenant_schema": tenant_schema,
                    "active_schema": connection.schema_name,
                    "document_id": document_id,
                },
            )

            services = import_module("ai_core.services")
            repository = services._get_documents_repository()

            try:
                document_uuid = UUID(str(document_id))
            except (TypeError, ValueError, AttributeError) as exc:
                raise InputError(
                    "invalid_document_id",
                    "Document identifier is not a valid UUID.",
                    context={"document_id": document_id},
                ) from exc

            # 1. Load Document
            normalized_document = repository.get(
                tenant, document_uuid, prefer_latest=True
            )
            if normalized_document is None:
                raise InputError(
                    "document_not_found",
                    "Document payload unavailable for ingestion.",
                    context={"document_id": document_id},
                )

            # 2. Build Components
            components = get_document_components()

            # Storage
            storage_candidate = components.storage
            try:
                storage = storage_candidate()
            except Exception:
                storage = storage_candidate

            # Captioner
            captioner_cls = components.captioner
            try:
                captioner = captioner_cls()
            except TypeError:
                captioner = captioner_cls

            # Parser
            # We use the default parser dispatcher from services/documents
            from documents.parsers import create_default_parser_dispatcher

            parser = create_default_parser_dispatcher()

            # Chunker
            from documents.cli import SimpleDocumentChunker

            chunker = SimpleDocumentChunker()

            # 3. Build Graph
            graph = build_document_processing_graph(
                parser=parser,
                repository=repository,
                storage=storage,
                captioner=captioner,
                chunker=chunker,
                embedder=trigger_embedding,
                delta_decider=decide_delta,
                guardrail_enforcer=enforce_guardrails,
                propagate_errors=True,
            )

            # 4. Prepare State
            # Config
            config = DocumentPipelineConfig(
                enable_embedding=True,
                enable_asset_captions=False,  # Default behavior from legacy
            )

            # Context
            context = DocumentProcessingContext.from_document(
                normalized_document,
                case_id=case,
                trace_id=trace_id,
            )

            # Initial State
            doc_state = DocumentProcessingState(
                document=normalized_document,
                config=config,
                context=context,
                storage=storage,
            )

            # 5. Invoke Graph
            try:
                result_state = graph.invoke(doc_state)
                if isinstance(result_state, dict):
                    result_state = DocumentProcessingState(**result_state)
            except Exception as exc:
                # Map exception to legacy error reporting
                state["last_error"] = {
                    "step": "graph_execution",
                    "message": str(exc),
                    "retry": getattr(self.request, "retries", 0),
                    "failed_at": time.time(),
                }
                _write_pipeline_state(tenant, case, document_id, state)
                raise

            # 6. Map Results
            chunks_generated = 0

            # Extract chunk stats
            if result_state.chunk_artifact:
                chunks_generated = len(result_state.chunk_artifact.chunks)

            # Prefer embedding.inserted if available (reflects actual DB insertions)
            if "embedding.inserted" in result_state.statistics:
                chunks_generated = int(result_state.statistics["embedding.inserted"])

            # Extract embedding/upsert stats
            # The API trigger_embedding calls upsert and result isn't explicitly passed back
            # in DocumentProcessingState except via observability side-channels or if we inspect artifacts.
            # DocumentProcessingGraph doesn't store EmbeddingResult in state explicitly in the built-in node.
            # However, we can assume if successful, it worked.
            # For simplicity in this migration, we report success.

            duration_ms = (time.perf_counter() - started) * 1000

            log.info(
                "Ingested document via graph",
                extra={
                    "tenant": tenant,
                    "case": case,
                    "document_id": document_id,
                    "chunk_count": chunks_generated,
                    "duration_ms": duration_ms,
                    "embedding_profile": embedding_profile,
                },
            )

            # Clean up state on success
            state["completed_at"] = time.time()
            state["last_error"] = None
            _write_pipeline_state(tenant, case, document_id, state)

            return {
                "document_id": document_id,
                "written": chunks_generated,  # Approximation
                "action": "inserted" if chunks_generated > 0 else "skipped",
                "chunk_count": chunks_generated,
                "duration_ms": duration_ms,
                "embedding_profile": embedding_profile,
                "inserted": chunks_generated,
                "replaced": 0,
                "skipped": 0,
            }

    except InputError:
        # Input validation errors are non-retriable - fail fast without wasting
        # queue capacity on futile retries. These include invalid document IDs,
        # missing payloads, and other user-correctable input issues.
        raise

    except Exception as exc:
        retries = getattr(self.request, "retries", 0)
        countdown = min(300, 5 * (2**retries or 1))

        state["last_error"] = {
            "step": "graph_execution",
            "message": str(exc),
            "retry": retries,
            "failed_at": time.time(),
        }
        _write_pipeline_state(tenant, case, document_id, state)

        log.warning(
            "Retrying ingestion document task after graph failure",
            extra={
                "tenant": tenant,
                "case": case,
                "document_id": document_id,
                "retries": retries,
                "error": str(exc),
            },
        )
        raise self.retry(exc=exc, countdown=countdown)


@shared_task(base=ScopedTask, queue="ingestion", accepts_scope=True)
def run_ingestion(
    tenant: str,
    case: str | None,
    document_ids: List[str],
    embedding_profile: str,
    *,
    run_id: str,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    tenant_schema: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    dead_letter_queue: Optional[str] = None,
    session_salt: Optional[str] = None,  # noqa: ARG001 - propagated by ScopedTask
    session_scope: Optional[Tuple[str, str, str]] = None,  # noqa: ARG001 - unused hook
) -> Dict[str, object]:
    if not tenant:
        raise InputError("missing_tenant_id", "tenant_id is required")
    if not run_id:
        raise InputError("missing_run_id", "run_id is required")
    if not trace_id:
        raise InputError("missing_trace_id", "trace_id is required")

    # Set tenant schema for django-tenants routing
    from customers.models import Tenant
    from django.db import connection

    if tenant_schema:
        try:
            tenant_obj = Tenant.objects.get(schema_name=tenant_schema)
            connection.set_tenant(tenant_obj)
            log.debug(
                "Tenant schema set for ingestion",
                extra={"tenant_schema": tenant_schema},
            )
        except Tenant.DoesNotExist:
            log.warning(
                "Tenant not found for schema",
                extra={"tenant_schema": tenant_schema},
            )

    # Use schema name for partitioning if available, as repository expects schema_name
    partition_tenant = tenant_schema if tenant_schema else tenant
    valid_ids, invalid_ids = partition_document_ids(
        partition_tenant, case or "", document_ids
    )
    dispatch_ids = list(valid_ids if valid_ids else document_ids)
    doc_count = len(valid_ids)
    start_case_context = _load_case_observability_context(tenant, case)
    try:
        binding = resolve_ingestion_profile(embedding_profile)
    except InputError as exc:
        log.error(
            "Failed to resolve ingestion profile",
            extra={
                "tenant": tenant,
                "case": case,
                "run_id": run_id,
                "trace_id": trace_id,
                "embedding_profile": embedding_profile,
                "error": str(exc),
            },
        )
        raise

    resolved_profile_id = binding.profile_id
    resolved_space = binding.resolution.vector_space
    vector_space_id = resolved_space.id
    vector_space_schema = resolved_space.schema
    vector_space_backend = resolved_space.backend
    vector_space_dimension = resolved_space.dimension

    schema_initialised = ensure_vector_space_schema(resolved_space)
    if schema_initialised:
        log.info(
            "Initialised vector schema for ingestion run",
            extra={
                "tenant": tenant,
                "case": case,
                "run_id": run_id,
                "embedding_profile": resolved_profile_id,
                "vector_space_id": vector_space_id,
                "vector_space_schema": vector_space_schema,
            },
        )

    pipe.log_ingestion_run_start(
        tenant=tenant,
        case=case,
        run_id=run_id,
        doc_count=doc_count,
        trace_id=trace_id,
        idempotency_key=idempotency_key,
        embedding_profile=resolved_profile_id,
        vector_space_id=vector_space_id,
        case_status=start_case_context.get("status"),
        case_phase=start_case_context.get("phase"),
    )
    mark_ingestion_run_running(
        tenant_id=tenant,
        case=case,
        run_id=run_id,
        started_at=timezone.now().isoformat(),
        document_ids=dispatch_ids,
    )
    emit_ingestion_case_event(
        tenant,
        case,
        run_id=run_id,
        context="running",
    )
    started = time.perf_counter()

    async_result = None
    failure: Optional[BaseException] = None
    results: List[Dict[str, object]] = []
    failed_ids: List[str] = []
    duration_ms = 0.0
    inserted = replaced = skipped = total_chunks = 0

    try:
        if valid_ids:
            job_group = group(
                process_document.s(
                    tenant,
                    case,
                    doc_id,
                    resolved_profile_id,
                    tenant_schema,
                    trace_id,
                )
                for doc_id in valid_ids
            )
            try:
                async_result = job_group.apply_async()
            except Exception as exc:  # pragma: no cover - defensive path
                failure = exc
                log.exception(
                    "Failed to dispatch ingestion group",
                    extra={
                        "tenant": tenant,
                        "case": case,
                        "run_id": run_id,
                        "trace_id": trace_id,
                    },
                )
            else:
                try:
                    results = async_result.get(
                        timeout=timeout_seconds, disable_sync_subtasks=False
                    )
                except CeleryTimeoutError as exc:
                    failure = exc
                    log.error(
                        "Timed out waiting for ingestion group",
                        extra={
                            "tenant": tenant,
                            "case": case,
                            "run_id": run_id,
                            "trace_id": trace_id,
                            "timeout_seconds": timeout_seconds,
                        },
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    failure = exc
                    log.exception(
                        "Ingestion group failed",
                        extra={
                            "tenant": tenant,
                            "case": case,
                            "run_id": run_id,
                            "trace_id": trace_id,
                        },
                    )

        if failure is not None:
            if async_result:
                results = _collect_partial_results(async_result)
                failed_ids = _determine_failed_documents(valid_ids, results)
                _revoke_pending(async_result)
            elif valid_ids:
                failed_ids = list(valid_ids)
            _safe_dispatch_dead_letters(
                dead_letter_queue,
                tenant,
                case,
                run_id,
                trace_id,
                failed_ids,
                failure,
                embedding_profile=resolved_profile_id,
                vector_space_id=vector_space_id,
                vector_space_schema=vector_space_schema,
                vector_space_backend=vector_space_backend,
                vector_space_dimension=vector_space_dimension,
            )

    except BaseException as exc:
        if failure is None:
            failure = exc
        if async_result and not failed_ids:
            results = _collect_partial_results(async_result)
            failed_ids = _determine_failed_documents(valid_ids, results)
            _revoke_pending(async_result)
            _safe_dispatch_dead_letters(
                dead_letter_queue,
                tenant,
                case,
                run_id,
                trace_id,
                failed_ids,
                failure,
                embedding_profile=resolved_profile_id,
                vector_space_id=vector_space_id,
                vector_space_schema=vector_space_schema,
                vector_space_backend=vector_space_backend,
                vector_space_dimension=vector_space_dimension,
            )
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        inserted = sum(int(result.get("inserted", 0)) for result in results)
        replaced = sum(int(result.get("replaced", 0)) for result in results)
        skipped = sum(int(result.get("skipped", 0)) for result in results)
        total_chunks = sum(
            int(result.get("chunk_count", result.get("written", 0)))
            for result in results
        )
        try:
            end_case_context = _load_case_observability_context(tenant, case)
        except Exception:  # pragma: no cover - defensive logging path
            end_case_context = {}
        try:
            pipe.log_ingestion_run_end(
                tenant=tenant,
                case=case,
                run_id=run_id,
                doc_count=doc_count,
                inserted=inserted,
                replaced=replaced,
                skipped=skipped,
                total_chunks=total_chunks,
                duration_ms=duration_ms,
                trace_id=trace_id,
                idempotency_key=idempotency_key,
                embedding_profile=resolved_profile_id,
                vector_space_id=vector_space_id,
                case_status=end_case_context.get("status"),
                case_phase=end_case_context.get("phase"),
            )
        except Exception:  # pragma: no cover - defensive logging path
            pass

    log.info(
        "Dispatched ingestion group",
        extra={
            "tenant": tenant,
            "case": case,
            "trace_id": trace_id,
            "count": doc_count,
            "invalid_ids": invalid_ids,
            "embedding_profile": resolved_profile_id,
            "vector_space_id": vector_space_id,
            "failed_ids": (
                failed_ids
                if failed_ids
                else (
                    _determine_failed_documents(valid_ids, results) if valid_ids else []
                )
            ),
        },
    )
    finished_at = timezone.now().isoformat()

    response: Dict[str, object] = {
        "status": "dispatched",
        "count": doc_count,
        "invalid_ids": invalid_ids,
        "inserted": inserted,
        "replaced": replaced,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "duration_ms": duration_ms,
    }
    if failure is not None:
        response["status"] = "failed"
        response["error"] = str(failure)

    mark_ingestion_run_completed(
        tenant_id=tenant,
        case=case,
        run_id=run_id,
        finished_at=finished_at,
        duration_ms=duration_ms,
        inserted_documents=inserted,
        replaced_documents=replaced,
        skipped_documents=skipped,
        inserted_chunks=total_chunks,
        invalid_document_ids=invalid_ids,
        document_ids=dispatch_ids,
        error=response.get("error"),
    )
    emit_ingestion_case_event(
        tenant,
        case,
        run_id=run_id,
        context="completed",
    )

    return response


def _collect_partial_results(async_result) -> List[Dict[str, object]]:
    collected: List[Dict[str, object]] = []
    for result in getattr(async_result, "results", []) or []:
        try:
            partial = result.get(timeout=0, propagate=False)
        except CeleryTimeoutError:
            continue
        except Exception:
            continue
        if isinstance(partial, dict):
            collected.append(partial)
    return collected


def _revoke_pending(async_result) -> None:
    for result in getattr(async_result, "results", []) or []:
        if not result.ready():
            try:
                result.revoke(terminate=True)
            except Exception:
                continue


def _determine_failed_documents(
    document_ids: Iterable[str], results: Iterable[Dict[str, object]]
) -> List[str]:
    processed_ids = {
        str(result.get("document_id"))
        for result in results
        if isinstance(result, dict) and result.get("document_id")
    }
    return [doc_id for doc_id in document_ids if doc_id not in processed_ids]


def _dispatch_dead_letters(
    dead_letter_queue: Optional[str],
    tenant: str,
    case: str | None,
    run_id: str,
    trace_id: Optional[str],
    failed_ids: Iterable[str],
    failure: BaseException,
    *,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
    vector_space_schema: Optional[str] = None,
    vector_space_backend: Optional[str] = None,
    vector_space_dimension: Optional[int] = None,
) -> None:
    failed_list = list(failed_ids)
    if not failed_list:
        return
    payload = {
        "tenant": tenant,
        "case": case,
        "run_id": run_id,
        "trace_id": trace_id,
        "error": str(failure),
    }
    if embedding_profile:
        payload["embedding_profile"] = embedding_profile
    if vector_space_id:
        payload["vector_space_id"] = vector_space_id
    if vector_space_schema:
        payload["vector_space_schema"] = vector_space_schema
    if vector_space_backend:
        payload["vector_space_backend"] = vector_space_backend
    if vector_space_dimension is not None:
        payload["vector_space_dimension"] = vector_space_dimension
    context = getattr(failure, "context", None)
    if isinstance(context, dict):
        for key, value in context.items():
            if key in payload and key not in {"process", "workflow_id"}:
                continue
            if value is not None or key in {"process", "workflow_id"}:
                payload[key] = value
    payload.setdefault("process", None)
    payload.setdefault("workflow_id", None)
    payload.pop("doc_class", None)

    for document_id in failed_list:
        message = {**payload, "document_id": document_id}
        record_dead_letter.apply_async(
            args=[message],
            queue=dead_letter_queue if dead_letter_queue else "ingestion_dead_letter",
        )


def _safe_dispatch_dead_letters(
    dead_letter_queue: Optional[str],
    tenant: str,
    case: str | None,
    run_id: str,
    trace_id: Optional[str],
    failed_ids: Iterable[str],
    failure: BaseException,
    *,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
    vector_space_schema: Optional[str] = None,
    vector_space_backend: Optional[str] = None,
    vector_space_dimension: Optional[int] = None,
) -> None:
    try:
        _dispatch_dead_letters(
            dead_letter_queue,
            tenant,
            case,
            run_id,
            trace_id,
            failed_ids,
            failure,
            embedding_profile=embedding_profile,
            vector_space_id=vector_space_id,
            vector_space_schema=vector_space_schema,
            vector_space_backend=vector_space_backend,
            vector_space_dimension=vector_space_dimension,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.exception(
            "Failed to dispatch ingestion dead letters",
            extra={
                "tenant": tenant,
                "case": case,
                "run_id": run_id,
                "trace_id": trace_id,
                "failed_ids": list(failed_ids),
                "embedding_profile": embedding_profile,
                "vector_space_id": vector_space_id,
            },
        )


@shared_task(
    base=ScopedTask,
    queue="ingestion_dead_letter",
    name="ai_core.ingestion.dead_letter",
    accepts_scope=True,
)
def record_dead_letter(payload: Dict[str, object]) -> None:
    log.error("Ingestion dead letter", extra=payload)
