import base64
import json
from pathlib import Path
from uuid import uuid4


from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage


WORKFLOW_ID = "workflow-1"


def _context() -> CLIContext:
    storage = InMemoryStorage()
    repository = InMemoryDocumentsRepository(storage=storage)
    return CLIContext(repository=repository, storage=storage)


def _run(args: list[str], context: CLIContext, capsys) -> tuple[int, str, str]:
    exit_code = main(["--json", *args], context=context)
    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


def _encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _write_inline_file(tmp_path: Path, name: str, payload: bytes) -> Path:
    file_path = tmp_path / name
    file_path.write_bytes(payload)
    return file_path


def test_docs_add_requires_workflow_id(capsys):
    context = _context()
    payload = _encode(b"doc")

    code, out, err = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--collection",
            str(uuid4()),
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )

    assert code == 1
    assert json.loads(out)["error"] == "workflow_id_required"


def test_assets_add_requires_workflow_id(capsys):
    context = _context()
    payload = _encode(b"asset")

    code, out, err = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--document",
            str(uuid4()),
            "--media-type",
            "image/png",
            "--inline",
            payload,
            "--caption-method",
            "none",
        ],
        context,
        capsys,
    )

    assert code == 1
    assert json.loads(out)["error"] == "workflow_id_required"


def test_docs_add_get_list_delete_roundtrip(capsys):
    context = _context()
    collection_id = str(uuid4())
    payload = _encode(b"hello world")

    code, out, err = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--title",
            "Greeting",
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    assert code == 0
    assert err == ""
    added = json.loads(out)
    doc_id = added["ref"]["document_id"]
    file_uri = added["blob"]["uri"]

    code, out, err = _run(
        [
            "docs",
            "get",
            "--tenant",
            "tenant-a",
            "--doc-id",
            doc_id,
            "--workflow-id",
            WORKFLOW_ID,
            "--prefer-latest",
        ],
        context,
        capsys,
    )
    assert code == 0
    fetched = json.loads(out)
    assert fetched["ref"]["document_id"] == doc_id
    assert fetched["blob"]["uri"] == file_uri

    code, out, err = _run(
        [
            "docs",
            "list",
            "--tenant",
            "tenant-a",
            "--collection",
            collection_id,
            "--limit",
            "5",
            "--workflow-id",
            WORKFLOW_ID,
        ],
        context,
        capsys,
    )
    assert code == 0
    listing = json.loads(out)
    assert listing["items"][0]["document_id"] == doc_id
    assert listing["next_cursor"] is None
    assert listing["workflow_id"] == WORKFLOW_ID

    code, out, err = _run(
        [
            "docs",
            "delete",
            "--tenant",
            "tenant-a",
            "--doc-id",
            doc_id,
            "--workflow-id",
            WORKFLOW_ID,
        ],
        context,
        capsys,
    )
    assert code == 0
    assert json.loads(out) == {"deleted": True, "workflow_id": WORKFLOW_ID}

    code, out, err = _run(
        [
            "docs",
            "get",
            "--tenant",
            "tenant-a",
            "--doc-id",
            doc_id,
            "--workflow-id",
            WORKFLOW_ID,
        ],
        context,
        capsys,
    )
    assert code == 1
    assert json.loads(out)["error"] == "document_not_found"


def test_docs_get_without_workflow_id(capsys):
    context = _context()
    payload = _encode(b"hello world")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    stored = json.loads(out)

    code, out, err = _run(
        [
            "docs",
            "get",
            "--tenant",
            "tenant-a",
            "--doc-id",
            stored["ref"]["document_id"],
        ],
        context,
        capsys,
    )

    assert code == 0
    fetched = json.loads(out)
    assert fetched["ref"]["document_id"] == stored["ref"]["document_id"]


def test_docs_delete_requires_workflow_id(capsys):
    context = _context()
    payload = _encode(b"hello world")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    stored = json.loads(out)

    code, out, err = _run(
        [
            "docs",
            "delete",
            "--tenant",
            "tenant-a",
            "--doc-id",
            stored["ref"]["document_id"],
        ],
        context,
        capsys,
    )

    assert code == 1
    assert json.loads(out)["error"] == "workflow_id_required"


def test_docs_add_with_existing_file_uri(capsys):
    context = _context()
    collection_id = str(uuid4())
    payload = _encode(b"inline bytes")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    assert code == 0
    doc = json.loads(out)
    uri = doc["blob"]["uri"]

    code, out, err = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--file-uri",
            uri,
            "--doc-id",
            doc["ref"]["document_id"],
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    assert code == 0
    updated = json.loads(out)
    assert updated["blob"]["uri"] == uri


def test_docs_add_inline_file(tmp_path, capsys):
    context = _context()
    collection_id = str(uuid4())
    inline_file = _write_inline_file(tmp_path, "payload.bin", b"inline-bytes")

    code, out, err = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--inline-file",
            str(inline_file),
            "--media-type",
            "application/pdf",
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    assert code == 0
    assert err == ""
    added = json.loads(out)
    assert added["blob"]["type"] == "file"


def test_assets_cli_flow(capsys):
    context = _context()
    collection_id = str(uuid4())
    doc_payload = _encode(b"document body")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--inline",
            doc_payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    doc = json.loads(out)
    doc_id = doc["ref"]["document_id"]

    asset_payload = _encode(b"asset")
    code, out, err = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--document",
            doc_id,
            "--media-type",
            "image/png",
            "--inline",
            asset_payload,
            "--caption-method",
            "none",
        ],
        context,
        capsys,
    )
    assert code == 0
    asset = json.loads(out)
    asset_id = asset["ref"]["asset_id"]
    asset_uri = asset["blob"]["uri"]

    code, out, err = _run(
        [
            "assets",
            "list",
            "--tenant",
            "tenant-a",
            "--document",
            doc_id,
        ],
        context,
        capsys,
    )
    assert code == 0
    asset_list = json.loads(out)
    assert asset_list["items"][0]["asset_id"] == asset_id

    code, out, err = _run(
        [
            "assets",
            "list",
            "--tenant",
            "tenant-a",
            "--document",
            doc_id,
            "--workflow-id",
            WORKFLOW_ID,
        ],
        context,
        capsys,
    )
    assert code == 0
    filtered_assets = json.loads(out)
    assert filtered_assets["items"][0]["asset_id"] == asset_id
    assert filtered_assets["workflow_id"] == WORKFLOW_ID

    code, out, err = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--document",
            doc_id,
            "--media-type",
            "image/png",
            "--file-uri",
            asset_uri,
            "--caption-method",
            "none",
        ],
        context,
        capsys,
    )
    assert code == 0

    code, out, err = _run(
        [
            "assets",
            "get",
            "--tenant",
            "tenant-a",
            "--asset-id",
            asset_id,
        ],
        context,
        capsys,
    )
    assert code == 0
    fetched = json.loads(out)
    assert fetched["ref"]["asset_id"] == asset_id

    code, out, err = _run(
        [
            "assets",
            "delete",
            "--tenant",
            "tenant-a",
            "--asset-id",
            asset_id,
            "--workflow-id",
            WORKFLOW_ID,
        ],
        context,
        capsys,
    )
    assert code == 0
    assert json.loads(out) == {"deleted": True, "workflow_id": WORKFLOW_ID}

    code, out, err = _run(
        [
            "assets",
            "get",
            "--tenant",
            "tenant-a",
            "--asset-id",
            asset_id,
        ],
        context,
        capsys,
    )
    assert code == 1
    assert json.loads(out)["error"] == "asset_not_found"


def test_assets_get_optional_workflow_id(capsys):
    context = _context()
    collection_id = str(uuid4())
    payload = _encode(b"document")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    doc = json.loads(out)

    asset_payload = _encode(b"asset")
    code, out, _ = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--document",
            doc["ref"]["document_id"],
            "--media-type",
            "image/png",
            "--inline",
            asset_payload,
            "--caption-method",
            "none",
        ],
        context,
        capsys,
    )
    stored_asset = json.loads(out)
    asset_id = stored_asset["ref"]["asset_id"]

    code, out, err = _run(
        [
            "assets",
            "get",
            "--tenant",
            "tenant-a",
            "--asset-id",
            asset_id,
            "--workflow-id",
            WORKFLOW_ID,
        ],
        context,
        capsys,
    )
    assert code == 0
    assert json.loads(out)["ref"]["asset_id"] == asset_id

    code, out, err = _run(
        [
            "assets",
            "get",
            "--tenant",
            "tenant-a",
            "--asset-id",
            asset_id,
        ],
        context,
        capsys,
    )
    assert code == 0
    assert json.loads(out)["ref"]["asset_id"] == asset_id


def test_assets_delete_requires_workflow_id(capsys):
    context = _context()
    asset_id = str(uuid4())

    code, out, err = _run(
        [
            "assets",
            "delete",
            "--tenant",
            "tenant-a",
            "--asset-id",
            asset_id,
        ],
        context,
        capsys,
    )

    assert code == 1
    assert json.loads(out)["error"] == "workflow_id_required"


def test_assets_add_inline_file(tmp_path, capsys):
    context = _context()
    collection_id = str(uuid4())
    doc_payload = _encode(b"document body")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--inline",
            doc_payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    doc = json.loads(out)
    doc_id = doc["ref"]["document_id"]

    inline_file = _write_inline_file(tmp_path, "asset.bin", b"asset-bytes")
    code, out, err = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--document",
            doc_id,
            "--media-type",
            "image/png",
            "--inline-file",
            str(inline_file),
            "--caption-method",
            "none",
        ],
        context,
        capsys,
    )
    assert code == 0
    asset = json.loads(out)
    assert asset["blob"]["type"] == "file"


def test_error_on_missing_blob_source(capsys):
    context = _context()
    code, out, err = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    assert code == 1
    assert json.loads(out)["error"] == "blob_source_required"


def test_invalid_inline_payload(capsys):
    context = _context()
    code, out, err = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--inline",
            "not-base64",
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    assert code == 1
    assert json.loads(out)["error"] == "base64_invalid"


def test_asset_add_missing_document(capsys):
    context = _context()
    payload = _encode(b"asset")
    code, out, err = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--document",
            str(uuid4()),
            "--media-type",
            "image/png",
            "--inline",
            payload,
            "--caption-method",
            "none",
        ],
        context,
        capsys,
    )
    assert code == 1
    assert json.loads(out)["error"] == "document_missing"


def test_asset_add_ocr_only_warns_without_ocr_text(capsys):
    context = _context()
    collection_id = str(uuid4())
    payload = _encode(b"document")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            collection_id,
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    doc = json.loads(out)

    asset_payload = _encode(b"asset")
    code, out, err = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--document",
            doc["ref"]["document_id"],
            "--media-type",
            "image/png",
            "--inline",
            asset_payload,
            "--caption-method",
            "ocr_only",
        ],
        context,
        capsys,
    )
    assert code == 0
    result = json.loads(out)
    assert result["warning"] == "ocr_text_missing_for_ocr_only"


def test_asset_add_rejects_parameterized_media_type(capsys):
    context = _context()
    payload = _encode(b"document")

    code, out, _ = _run(
        [
            "docs",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--collection",
            str(uuid4()),
            "--inline",
            payload,
            "--source",
            "upload",
        ],
        context,
        capsys,
    )
    doc = json.loads(out)

    asset_payload = _encode(b"asset")
    code, out, err = _run(
        [
            "assets",
            "add",
            "--tenant",
            "tenant-a",
            "--workflow-id",
            WORKFLOW_ID,
            "--document",
            doc["ref"]["document_id"],
            "--media-type",
            "text/html; charset=utf-8",
            "--inline",
            asset_payload,
            "--caption-method",
            "none",
        ],
        context,
        capsys,
    )
    assert code == 1
    assert "media_type_invalid" in json.loads(out)["error"]


def test_schema_print_command(capsys):
    context = _context()
    code, out, err = _run(
        ["schema", "print", "--kind", "normalized-document"],
        context,
        capsys,
    )
    assert code == 0
    schema = json.loads(out)
    assert "$defs" in schema or "properties" in schema


def test_schema_print_all_command(capsys):
    context = _context()
    code, out, err = _run(
        ["schema", "print", "--kind", "all"],
        context,
        capsys,
    )
    assert code == 0
    schemas = json.loads(out)
    expected_keys = {
        "normalized-document",
        "document-ref",
        "blob",
        "asset",
        "asset-ref",
        "document-meta",
    }
    assert set(schemas.keys()) == expected_keys


def test_docs_list_empty_with_cursor(capsys):
    context = _context()
    code, out, err = _run(
        [
            "docs",
            "list",
            "--tenant",
            "tenant-a",
            "--collection",
            str(uuid4()),
            "--cursor",
            "invalid-cursor",
        ],
        context,
        capsys,
    )
    assert code == 1
    assert json.loads(out)["error"] == "cursor_invalid"
