"""Command line interface for document and asset smoke tests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import UUID, uuid4

from pydantic import ValidationError

from common.logging import log_context

from .contracts import (
    Asset,
    AssetRef,
    DocumentMeta,
    DocumentRef,
    FileBlob,
    InlineBlob,
    NormalizedDocument,
    blob_locator_schema,
    document_meta_schema,
    document_ref_schema,
    normalized_document_schema,
    asset_ref_schema,
    asset_schema,
)
from .logging_utils import (
    asset_log_fields,
    document_log_fields,
    log_call,
    log_extra_entry,
    log_extra_exit,
)
from .repository import DocumentsRepository, InMemoryDocumentsRepository
from .storage import InMemoryStorage, Storage


@dataclass
class CLIContext:
    """Holds repository and storage instances for CLI operations."""

    repository: DocumentsRepository
    storage: Storage


def _default_context() -> CLIContext:
    storage = InMemoryStorage()
    repository = InMemoryDocumentsRepository(storage=storage)
    return CLIContext(repository=repository, storage=storage)


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {key: _serialize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, tuple):
        return [_serialize(item) for item in obj]
    return obj


def _print_success(
    args: argparse.Namespace, payload: Any, *, warning: Optional[str] = None
) -> int:
    data = _serialize(payload)
    if warning and getattr(args, "json", False):
        if isinstance(data, dict):
            if "warning" in data and data["warning"] != warning:
                data = {"warning": warning, "result": data}
            else:
                data["warning"] = warning
        else:
            data = {"warning": warning, "result": data}
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(json.dumps(data, indent=2))
    return 0


def _print_error(args: argparse.Namespace, message: str) -> int:
    log_extra_exit(status="error", error_code=message)
    if getattr(args, "json", False):
        print(json.dumps({"error": message}, indent=2, sort_keys=True))
    else:
        print(f"error: {message}", file=sys.stderr)
    return 1


def _validation_message(error: ValidationError) -> str:
    parts = []
    for err in error.errors():
        loc = ".".join(str(item) for item in err.get("loc", ()))
        msg = err.get("msg", "validation_error")
        if loc:
            parts.append(f"{loc}: {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts) if parts else "validation_error"


def _exclusive_args(
    inline: Optional[str], inline_file: Optional[str], file_uri: Optional[str]
) -> None:
    provided = sum(value is not None for value in (inline, inline_file, file_uri))
    if provided != 1:
        raise ValueError("blob_source_required")


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # pragma: no cover - delegation to caller
        raise ValueError("base64_invalid") from exc


def _inline_blob(inline: str, media_type: str) -> tuple[InlineBlob, str]:
    payload = _decode_base64(inline)
    sha = hashlib.sha256(payload).hexdigest()
    blob = InlineBlob(
        type="inline",
        media_type=media_type,
        base64=inline,
        sha256=sha,
        size=len(payload),
    )
    return blob, sha


def _file_blob(storage: Storage, uri: str) -> tuple[FileBlob, str]:
    try:
        payload = storage.get(uri)
    except KeyError as exc:
        raise ValueError("storage_uri_missing") from exc
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    sha = hashlib.sha256(payload).hexdigest()
    blob = FileBlob(type="file", uri=uri, sha256=sha, size=len(payload))
    return blob, sha


def _inline_from_file(path: str) -> str:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:  # pragma: no cover - propagated as CLI error
        raise ValueError("inline_file_unreadable") from exc
    return base64.b64encode(payload).decode("ascii")


def _parse_external_ref(raw: Optional[str]) -> Optional[dict[str, str]]:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("external_ref_invalid") from exc
    if not isinstance(data, dict):
        raise ValueError("external_ref_invalid")
    normalized: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("external_ref_invalid")
        normalized[key] = value
    return normalized


def _parse_bbox(raw: Optional[str]) -> Optional[list[float]]:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("bbox_invalid") from exc
    if not isinstance(data, list):
        raise ValueError("bbox_invalid")
    # Further range validation for bounding boxes occurs within the model layer.
    return [float(item) for item in data]


def _schema_payload(kind: str) -> Any:
    mapping = {
        "normalized-document": normalized_document_schema,
        "document-ref": document_ref_schema,
        "blob": blob_locator_schema,
        "asset": asset_schema,
        "asset-ref": asset_ref_schema,
        "document-meta": document_meta_schema,
    }
    if kind == "all":
        return {name: factory() for name, factory in mapping.items()}
    try:
        factory = mapping[kind]
    except KeyError as exc:
        raise ValueError("schema_kind_invalid") from exc
    return factory()


@log_call("cli.schema.print")
def _handle_schema_print(args: argparse.Namespace) -> int:
    log_extra_entry(kind=args.kind)
    payload = _schema_payload(args.kind)
    log_extra_exit(rendered=True)
    return _print_success(args, payload)


@log_call("cli.docs.add")
def _handle_docs_add(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    _exclusive_args(args.inline, args.inline_file, args.file_uri)
    inline_payload = args.inline
    if args.inline_file:
        inline_payload = _inline_from_file(args.inline_file)
    media_type = args.media_type or "application/octet-stream"
    if inline_payload is not None:
        blob, checksum = _inline_blob(inline_payload, media_type)
    else:
        blob, checksum = _file_blob(context.storage, args.file_uri)
    document_id = args.doc_id or uuid4()
    ref = DocumentRef(
        tenant_id=args.tenant,
        workflow_id=args.workflow,
        document_id=document_id,
        collection_id=args.collection,
        version=args.version,
    )
    meta = DocumentMeta(
        tenant_id=args.tenant,
        workflow_id=args.workflow,
        title=args.title,
        language=args.lang,
        tags=args.tag or [],
        origin_uri=args.origin_uri,
        crawl_timestamp=None,
        external_ref=_parse_external_ref(args.external_ref),
    )
    document = NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source=args.source,
        assets=[],
    )
    with log_context(
        tenant=args.tenant,
        collection_id=str(args.collection) if args.collection else None,
    ):
        log_extra_entry(**document_log_fields(document))
        stored = context.repository.upsert(document)
        log_extra_exit(**document_log_fields(stored))
    return _print_success(args, stored)


@log_call("cli.docs.get")
def _handle_docs_get(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    document_id = UUID(args.doc_id)
    with log_context(tenant=args.tenant, workflow_id=args.workflow):
        log_extra_entry(
            tenant_id=args.tenant,
            document_id=document_id,
            version=args.version,
            prefer_latest=args.prefer_latest,
            workflow_id=args.workflow,
        )
        doc = context.repository.get(
            args.tenant,
            document_id,
            version=args.version,
            prefer_latest=args.prefer_latest,
            workflow_id=args.workflow,
        )
        if doc is None:
            log_extra_exit(status="error", error_code="document_not_found")
            return _print_error(args, "document_not_found")
        log_extra_exit(**document_log_fields(doc))
        return _print_success(args, doc)


@log_call("cli.docs.list")
def _handle_docs_list(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    collection_id = UUID(args.collection)
    with log_context(
        tenant=args.tenant,
        collection_id=str(collection_id),
        workflow_id=args.workflow,
    ):
        log_extra_entry(
            tenant_id=args.tenant,
            collection_id=collection_id,
            limit=args.limit,
            cursor_present=bool(args.cursor),
            latest_only=args.latest_only,
            workflow_id=args.workflow,
        )
        refs, cursor = context.repository.list_by_collection(
            args.tenant,
            collection_id,
            limit=args.limit,
            cursor=args.cursor,
            latest_only=args.latest_only,
            workflow_id=args.workflow,
        )
        payload = {"items": refs, "next_cursor": cursor}
        log_extra_exit(item_count=len(refs), next_cursor_present=bool(cursor))
        return _print_success(args, payload)


@log_call("cli.docs.delete")
def _handle_docs_delete(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    document_id = UUID(args.doc_id)
    with log_context(tenant=args.tenant, workflow_id=args.workflow):
        log_extra_entry(
            tenant_id=args.tenant,
            document_id=document_id,
            hard=args.hard,
            workflow_id=args.workflow,
        )
        deleted = context.repository.delete(
            args.tenant,
            document_id,
            workflow_id=args.workflow,
            hard=args.hard,
        )
        if not deleted:
            log_extra_exit(status="error", error_code="document_not_found")
            return _print_error(args, "document_not_found")
        log_extra_exit(deleted=True)
        return _print_success(args, {"deleted": True})


@log_call("cli.assets.add")
def _handle_assets_add(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    _exclusive_args(args.inline, args.inline_file, args.file_uri)
    inline_payload = args.inline
    if args.inline_file:
        inline_payload = _inline_from_file(args.inline_file)
    if inline_payload is not None:
        blob, checksum = _inline_blob(inline_payload, args.media_type)
    else:
        blob, checksum = _file_blob(context.storage, args.file_uri)
    asset_id = args.asset_id or uuid4()
    ref = AssetRef(
        tenant_id=args.tenant,
        workflow_id=args.workflow,
        asset_id=asset_id,
        document_id=UUID(args.document),
    )
    asset = Asset(
        ref=ref,
        media_type=args.media_type,
        blob=blob,
        origin_uri=args.origin_uri,
        page_index=args.page_index,
        bbox=_parse_bbox(args.bbox),
        context_before=args.context_before,
        context_after=args.context_after,
        ocr_text=args.ocr_text,
        text_description=args.text_description,
        caption_method=args.caption_method,
        caption_model=args.caption_model,
        caption_confidence=args.caption_confidence,
        created_at=datetime.now(timezone.utc),
        checksum=checksum,
    )
    with log_context(tenant=args.tenant, workflow_id=args.workflow):
        log_extra_entry(**asset_log_fields(asset))
        try:
            stored = context.repository.add_asset(asset)
        except ValueError as exc:
            log_extra_exit(status="error", error_code=str(exc))
            return _print_error(args, str(exc))
        warning = None
        if (
            args.caption_method == "ocr_only"
            and not args.ocr_text
            and getattr(args, "json", False)
        ):
            warning = "ocr_text_missing_for_ocr_only"
        log_extra_exit(**asset_log_fields(stored))
        return _print_success(args, stored, warning=warning)


@log_call("cli.assets.get")
def _handle_assets_get(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    asset_id = UUID(args.asset_id)
    with log_context(tenant=args.tenant, workflow_id=args.workflow):
        log_extra_entry(
            tenant_id=args.tenant,
            asset_id=asset_id,
            workflow_id=args.workflow,
        )
        asset = context.repository.get_asset(
            args.tenant, asset_id, workflow_id=args.workflow
        )
        if asset is None:
            log_extra_exit(status="error", error_code="asset_not_found")
            return _print_error(args, "asset_not_found")
        log_extra_exit(**asset_log_fields(asset))
        return _print_success(args, asset)


@log_call("cli.assets.list")
def _handle_assets_list(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    document_id = UUID(args.document)
    with log_context(tenant=args.tenant, workflow_id=args.workflow):
        log_extra_entry(
            tenant_id=args.tenant,
            document_id=document_id,
            limit=args.limit,
            cursor_present=bool(args.cursor),
            workflow_id=args.workflow,
        )
        refs, cursor = context.repository.list_assets_by_document(
            args.tenant,
            document_id,
            limit=args.limit,
            cursor=args.cursor,
            workflow_id=args.workflow,
        )
        payload = {"items": refs, "next_cursor": cursor}
        log_extra_exit(item_count=len(refs), next_cursor_present=bool(cursor))
        return _print_success(args, payload)


@log_call("cli.assets.delete")
def _handle_assets_delete(args: argparse.Namespace) -> int:
    context: CLIContext = args.context
    asset_id = UUID(args.asset_id)
    with log_context(tenant=args.tenant, workflow_id=args.workflow):
        log_extra_entry(
            tenant_id=args.tenant,
            asset_id=asset_id,
            hard=args.hard,
            workflow_id=args.workflow,
        )
        deleted = context.repository.delete_asset(
            args.tenant,
            asset_id,
            workflow_id=args.workflow,
            hard=args.hard,
        )
        if not deleted:
            log_extra_exit(status="error", error_code="asset_not_found")
            return _print_error(args, "asset_not_found")
        log_extra_exit(deleted=True)
        return _print_success(args, {"deleted": True})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document repository CLI")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit command output (and errors) as JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("schema", help="Schema operations")
    schema_sub = schema_parser.add_subparsers(dest="schema_command", required=True)
    schema_print = schema_sub.add_parser("print", help="Print JSON schema")
    schema_print.add_argument(
        "--kind",
        required=True,
        choices=[
            "normalized-document",
            "document-ref",
            "blob",
            "asset",
            "asset-ref",
            "document-meta",
            "all",
        ],
        help="Schema kind to render (use 'all' to print every schema).",
    )
    schema_print.set_defaults(func=_handle_schema_print)

    docs_parser = subparsers.add_parser("docs", help="Document operations")
    docs_sub = docs_parser.add_subparsers(dest="docs_command", required=True)

    docs_add = docs_sub.add_parser("add", help="Add or update a document")
    docs_add.add_argument("--tenant", required=True)
    docs_add.add_argument("--workflow", required=True)
    docs_add.add_argument("--doc-id")
    docs_add.add_argument("--collection")
    docs_add.add_argument("--version")
    docs_add.add_argument("--title")
    docs_add.add_argument("--lang")
    docs_add.add_argument("--origin-uri")
    docs_add.add_argument("--external-ref")
    docs_add.add_argument("--inline")
    docs_add.add_argument(
        "--inline-file",
        help="Read inline payload from file path and base64-encode it.",
    )
    docs_add.add_argument("--file-uri")
    docs_add.add_argument(
        "--media-type",
        help="Media type for inline payloads (only used with --inline/--inline-file).",
    )
    docs_add.add_argument(
        "--source",
        required=True,
        choices=["upload", "crawler", "integration", "other"],
    )
    docs_add.add_argument("--tag", action="append")
    docs_add.set_defaults(func=_handle_docs_add)

    docs_get = docs_sub.add_parser("get", help="Fetch a document")
    docs_get.add_argument("--tenant", required=True)
    docs_get.add_argument("--doc-id", required=True)
    docs_get.add_argument("--version")
    docs_get.add_argument("--prefer-latest", action="store_true")
    docs_get.add_argument("--workflow")
    docs_get.set_defaults(func=_handle_docs_get)

    docs_list = docs_sub.add_parser(
        "list",
        help=(
            "List documents by collection (use --latest-only to return the newest "
            "version per document)."
        ),
    )
    docs_list.add_argument("--tenant", required=True)
    docs_list.add_argument("--collection", required=True)
    docs_list.add_argument("--limit", type=int, default=100)
    docs_list.add_argument("--cursor")
    docs_list.add_argument("--latest-only", action="store_true")
    docs_list.add_argument("--workflow")
    docs_list.set_defaults(func=_handle_docs_list)

    docs_delete = docs_sub.add_parser("delete", help="Delete a document")
    docs_delete.add_argument("--tenant", required=True)
    docs_delete.add_argument("--doc-id", required=True)
    docs_delete.add_argument("--hard", action="store_true")
    docs_delete.add_argument("--workflow")
    docs_delete.set_defaults(func=_handle_docs_delete)

    assets_parser = subparsers.add_parser("assets", help="Asset operations")
    assets_sub = assets_parser.add_subparsers(dest="assets_command", required=True)

    assets_add = assets_sub.add_parser("add", help="Add an asset to a document")
    assets_add.add_argument("--tenant", required=True)
    assets_add.add_argument("--workflow", required=True)
    assets_add.add_argument("--asset-id")
    assets_add.add_argument("--document", required=True)
    assets_add.add_argument("--media-type", required=True)
    assets_add.add_argument("--inline")
    assets_add.add_argument(
        "--inline-file",
        help="Read inline payload from file path and base64-encode it.",
    )
    assets_add.add_argument("--file-uri")
    assets_add.add_argument("--origin-uri")
    assets_add.add_argument("--page-index", type=int)
    assets_add.add_argument("--bbox")
    assets_add.add_argument("--context-before")
    assets_add.add_argument("--context-after")
    assets_add.add_argument("--ocr-text")
    assets_add.add_argument("--text-description")
    assets_add.add_argument(
        "--caption-method",
        required=True,
        choices=["vlm_caption", "ocr_only", "manual", "none"],
    )
    assets_add.add_argument("--caption-model")
    assets_add.add_argument("--caption-confidence", type=float)
    assets_add.set_defaults(func=_handle_assets_add)

    assets_get = assets_sub.add_parser("get", help="Fetch an asset")
    assets_get.add_argument("--tenant", required=True)
    assets_get.add_argument("--asset-id", required=True)
    assets_get.add_argument("--workflow")
    assets_get.set_defaults(func=_handle_assets_get)

    assets_list = assets_sub.add_parser(
        "list", help="List assets for a document"
    )
    assets_list.add_argument("--tenant", required=True)
    assets_list.add_argument("--document", required=True)
    assets_list.add_argument("--limit", type=int, default=100)
    assets_list.add_argument("--cursor")
    assets_list.add_argument("--workflow")
    assets_list.set_defaults(func=_handle_assets_list)

    assets_delete = assets_sub.add_parser("delete", help="Delete an asset")
    assets_delete.add_argument("--tenant", required=True)
    assets_delete.add_argument("--asset-id", required=True)
    assets_delete.add_argument("--hard", action="store_true")
    assets_delete.add_argument("--workflow")
    assets_delete.set_defaults(func=_handle_assets_delete)

    return parser


@log_call("cli.main")
def main(argv: Optional[Iterable[str]] = None, *, context: Optional[CLIContext] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    active_context = context or _default_context()
    setattr(args, "context", active_context)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        command = getattr(args, "command", None)
        sub_command = getattr(args, f"{command}_command", None) if command else None
        log_extra_entry(command=command, sub_command=sub_command)
        result = handler(args)
        log_extra_exit(exit_code=result)
        return result
    except ValidationError as exc:
        log_extra_exit(status="error", error_code="validation_error")
        return _print_error(args, _validation_message(exc))
    except ValueError as exc:
        log_extra_exit(status="error", error_code=str(exc))
        return _print_error(args, str(exc))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
