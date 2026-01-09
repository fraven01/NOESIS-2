# Documents CLI How-To

Die Dokumenten-CLI dient als schlankes Werkzeug für Smoke-Tests ohne API. Jede Ausführung erzeugt ein frisches In-Memory-Repository samt Storage; führe zusammenhängende Befehle deshalb innerhalb eines Python-Skripts aus, das einen gemeinsamen `CLIContext` verwendet.

## Session-Template
Verwende folgendes Template als Ausgangspunkt. Es kapselt `documents.cli.main`, sammelt JSON-Ausgaben und gibt Exit-Codes aus.

```bash
python - <<'PY'
import io
import json
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    print(f"$ documents.cli {' '.join(args)}")
    print(payload)
    print(f"exit_code={exit_code}\n")
    return exit_code, json.loads(payload)

# Weitere run(...) Aufrufe hier ergänzen.
PY
```

> Hinweis: Das Flag `--media-type` bei `docs add` ist nur relevant für Inline-Payloads (`--inline` oder `--inline-file`). Bei `--file-uri` wird der Media-Type aus dem gespeicherten Blob übernommen.

## Aufgabenrezepte

### Dokument hinzufügen (Inline)
```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    print(payload)
    return exit_code, json.loads(payload)

inline_payload = b64encode(b"Inline hello world").decode()
code, doc = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "11111111-1111-1111-1111-111111111111",
    "--title", "Inline Demo",
    "--inline", inline_payload,
    "--media-type", "text/plain",
    "--source", "upload",
)
print("document_id=", doc["ref"]["document_id"])
PY
```

**Beispielausgabe**
```json
{
  "assets": [],
  "blob": {
    "sha256": "bd3f49afdf47717d8dbd141e5018ed1b1d0e789ef046c03b8959d1dffed0f8c1",
    "size": 18,
    "type": "file",
    "uri": "memory://blob-1"
  },
  "checksum": "bd3f49afdf47717d8dbd141e5018ed1b1d0e789ef046c03b8959d1dffed0f8c1",
  "created_at": "2024-05-02T10:15:00Z",
  "meta": {
    "language": null,
    "tags": [],
    "tenant_id": "tenant-a",
    "title": "Inline Demo"
  },
  "ref": {
    "collection_id": "11111111-1111-1111-1111-111111111111",
    "document_id": "c7f8b4f4-1b7b-4ad2-9da6-0f8df1d96c90",
    "tenant_id": "tenant-a",
    "version": null
  },
  "source": "upload"
}
```

### Dokument hinzufügen (File-URI)
1. Lege zuerst Bytes direkt im Storage ab.
2. Verwende die erzeugte URI in `docs add --file-uri`.

```bash
python - <<'PY'
import io
import json
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

uri, sha256, size = storage.put(b"Persisted payload")

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    print(payload)
    return exit_code, json.loads(payload)

code, doc = run(
    "docs", "add",
    "--tenant", "tenant-b",
    "--collection", "22222222-2222-2222-2222-222222222222",
    "--file-uri", uri,
    "--source", "upload",
)
print("uri=", doc["blob"]["uri"], "checksum=", doc["checksum"])
PY
```

**Beispielausgabe**
```json
{
  "assets": [],
  "blob": {
    "sha256": "2aa7d1af4d61b061fd7e1a6ebb75556d7d8aa996f66d01a6c82bfa342e401e54",
    "size": 17,
    "type": "file",
    "uri": "memory://blob-1"
  },
  "checksum": "2aa7d1af4d61b061fd7e1a6ebb75556d7d8aa996f66d01a6c82bfa342e401e54",
  "created_at": "2024-05-02T10:15:00Z",
  "meta": {
    "language": null,
    "tags": [],
    "tenant_id": "tenant-b",
    "title": null
  },
  "ref": {
    "collection_id": "22222222-2222-2222-2222-222222222222",
    "document_id": "43994302-ef39-49ac-a2a0-0b06da9eec1d",
    "tenant_id": "tenant-b",
    "version": null
  },
  "source": "upload"
}
```

### Dokument anzeigen
```bash
python - <<'PY'
import io
import json
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return exit_code, json.loads(payload)

code, added = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "33333333-3333-3333-3333-333333333333",
    "--inline", "SGVsbG8=",
    "--media-type", "text/plain",
    "--source", "upload",
)
doc_id = added["ref"]["document_id"]
code, fetched = run(
    "docs", "get",
    "--tenant", "tenant-a",
    "--doc-id", doc_id,
    "--prefer-latest",
)
print(json.dumps(fetched, indent=2))
PY
```

### Liste nach Collection (inkl. `--latest-only`)
```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)
collection = "44444444-4444-4444-4444-444444444444"

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return exit_code, json.loads(payload)

for version in ("v1", "v2"):
    run(
        "docs", "add",
        "--tenant", "tenant-a",
        "--collection", collection,
        "--doc-id", "11111111-aaaa-bbbb-cccc-000000000000",
        "--version", version,
        "--inline", b64encode(b"Hello").decode(),
        "--media-type", "text/plain",
        "--source", "upload",
    )

code, listing = run(
    "docs", "list",
    "--tenant", "tenant-a",
    "--collection", collection,
    "--latest-only",
    "--limit", "10",
)
print(json.dumps(listing, indent=2))
PY
```

### Dokument löschen (soft & hard)
```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return exit_code, json.loads(payload)

code, added = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "55555555-5555-5555-5555-555555555555",
    "--inline", b64encode(b"Soft delete").decode(),
    "--media-type", "text/plain",
    "--source", "upload",
)
doc_id = added["ref"]["document_id"]
print(run("docs", "delete", "--tenant", "tenant-a", "--doc-id", doc_id))
print(run("docs", "delete", "--tenant", "tenant-a", "--doc-id", doc_id, "--hard"))
PY
```

### Asset hinzufügen (mit BBox & Kontext)
```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return exit_code, json.loads(payload)

code, doc = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "66666666-6666-6666-6666-666666666666",
    "--inline", b64encode(b"base document").decode(),
    "--media-type", "application/pdf",
    "--source", "upload",
)
doc_id = doc["ref"]["document_id"]
img_payload = b64encode(b"fake image bytes").decode()
code, asset = run(
    "assets", "add",
    "--tenant", "tenant-a",
    "--document", doc_id,
    "--media-type", "image/png",
    "--inline", img_payload,
    "--bbox", "[0.1, 0.1, 0.5, 0.6]",
    "--context-before", "Figure 1",
    "--context-after", "Details follow",
    "--caption-method", "vlm_caption",
    "--caption-model", "stub-v1",
    "--caption-confidence", "0.9",
)
print(json.dumps(asset, indent=2))
PY
```

### Asset hinzufügen mit OCR-Warnung (`ocr_only`)
```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return exit_code, json.loads(payload)

code, doc = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "77777777-7777-7777-7777-777777777777",
    "--inline", b64encode(b"doc").decode(),
    "--media-type", "application/pdf",
    "--source", "upload",
)
doc_id = doc["ref"]["document_id"]
code, asset = run(
    "assets", "add",
    "--tenant", "tenant-a",
    "--document", doc_id,
    "--media-type", "image/png",
    "--inline", b64encode(b"img").decode(),
    "--caption-method", "ocr_only",
)
print(json.dumps(asset, indent=2))
PY
```
→ Die Ausgabe enthält zusätzlich `"warning": "ocr_text_missing_for_ocr_only"`.

### Asset anzeigen und listen
```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return exit_code, json.loads(payload)

code, doc = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "88888888-8888-8888-8888-888888888888",
    "--inline", b64encode(b"doc").decode(),
    "--media-type", "application/pdf",
    "--source", "upload",
)
doc_id = doc["ref"]["document_id"]
code, asset = run(
    "assets", "add",
    "--tenant", "tenant-a",
    "--document", doc_id,
    "--media-type", "image/png",
    "--inline", b64encode(b"img").decode(),
    "--caption-method", "none",
)
asset_id = asset["ref"]["asset_id"]
print(json.dumps(run("assets", "get", "--tenant", "tenant-a", "--asset-id", asset_id)[1], indent=2))
print(json.dumps(run("assets", "list", "--tenant", "tenant-a", "--document", doc_id)[1], indent=2))
PY
```

### Asset löschen (soft & hard)
```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return exit_code, json.loads(payload)

code, doc = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "99999999-9999-9999-9999-999999999999",
    "--inline", b64encode(b"doc").decode(),
    "--media-type", "application/pdf",
    "--source", "upload",
)
doc_id = doc["ref"]["document_id"]
code, asset = run(
    "assets", "add",
    "--tenant", "tenant-a",
    "--document", doc_id,
    "--media-type", "image/png",
    "--inline", b64encode(b"img").decode(),
    "--caption-method", "none",
)
asset_id = asset["ref"]["asset_id"]
print(run("assets", "delete", "--tenant", "tenant-a", "--asset-id", asset_id))
print(run("assets", "delete", "--tenant", "tenant-a", "--asset-id", asset_id, "--hard"))
PY
```

### Schema inspizieren (`--kind all`)
```bash
python - <<'PY'
import io
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

buf = io.StringIO()
with redirect_stdout(buf):
    exit_code = main(["--json", "schema", "print", "--kind", "all"], context=context)
print(buf.getvalue())
print("exit_code=", exit_code)
PY
```

### Einfache Textsuche auf Assets
Nutze `jq`, um JSON-Ausgaben nach Textmustern zu filtern.

```bash
python - <<'PY'
import io
import json
from base64 import b64encode
from contextlib import redirect_stdout
from documents.cli import CLIContext, main
from documents.repository import InMemoryDocumentsRepository
from documents.storage import InMemoryStorage

storage = InMemoryStorage()
context = CLIContext(repository=InMemoryDocumentsRepository(storage=storage), storage=storage)

def run(*args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["--json", *args], context=context)
    payload = buf.getvalue().strip()
    return json.loads(payload)

doc = run(
    "docs", "add",
    "--tenant", "tenant-a",
    "--collection", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "--inline", b64encode(b"doc").decode(),
    "--media-type", "application/pdf",
    "--source", "upload",
)
doc_id = doc["ref"]["document_id"]
asset = run(
    "assets", "add",
    "--tenant", "tenant-a",
    "--document", doc_id,
    "--media-type", "image/png",
    "--inline", b64encode(b"img").decode(),
    "--text-description", "Invoice overview for March",
    "--caption-method", "manual",
)
listing = run(
    "assets", "list",
    "--tenant", "tenant-a",
    "--document", doc_id,
)
with open("assets.json", "w", encoding="utf-8") as handle:
    json.dump(listing, handle, indent=2)
print(json.dumps(asset, indent=2))
print("assets.json geschrieben")
PY
```

Filter Beispiel:
```bash
jq '.items[] | select(.text_description | test("invoice"; "i"))' assets.json
```

## Troubleshooting
| Fehlercode | Ursache | Lösung |
| --- | --- | --- |
| `blob_source_required` | Weder `--inline`/`--inline-file` noch `--file-uri` gesetzt | Genau eine Quelle auswählen |
| `base64_invalid` | Inline-Payload ist keine gültige Base64-Codierung | Payload korrigieren oder `--inline-file` nutzen |
| `storage_uri_missing` | Angegebene `--file-uri` existiert nicht im Storage | URI aus vorherigem Output übernehmen oder Bytes per `storage.put` erstellen |
| `validation_error` | Pydantic-Validierung (z. B. Bounding Box, Sprache) fehlgeschlagen | Fehlermeldung prüfen, Werte anpassen |
| `document_not_found` | Dokument-ID ist im aktuellen In-Memory-Repository nicht vorhanden | Dokument im selben Kontext anlegen oder ID korrigieren |
| `asset_not_found` | Asset-ID existiert nicht | Asset-ID prüfen oder Asset zuvor anlegen |
| `schema_kind_invalid` | Ungültiger Wert für `schema print --kind` | Einen der erlaubten Werte (`normalized-document`, `all`, …) verwenden |
| `ocr_text_missing_for_ocr_only` (Warning) | `--caption-method ocr_only` ohne `--ocr-text` | OCR-Text nachreichen oder Warnung bewusst ignorieren |

## Quick-Checks vor Nutzung
- LOG-Level via `LOG_LEVEL` konfigurieren, um CLI-Logs zu reduzieren.
- Bei Bedarf `OTEL_EXPORTER_OTLP_ENDPOINT` setzen, damit Spans weitergereicht werden.
- Kontext pro Run beibehalten, damit Storage/Repository synchron bleiben.
- Keine Base64- oder Textinhalte in Logs kopieren – nur IDs und Checksummen verwenden.
