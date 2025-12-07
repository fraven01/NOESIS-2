# Repository Divergence Analysis: InMemory vs. ObjectStore vs. DB

**Datum**: 2025-12-07
**Scope**: Developer Workbench (rag-tools) → Dokumenten Explorer
**Ziel**: Risiken, Abweichungen, Inkonsistenzen zwischen den drei DocumentsRepository-Implementierungen identifizieren

---

## Executive Summary

Die NOESIS 2 Plattform nutzt drei verschiedene `DocumentsRepository`-Implementierungen:

1. **InMemoryDocumentsRepository** (`documents/repository.py`) – Thread-safe, für Tests/Dev
2. **ObjectStoreDocumentsRepository** (`ai_core/adapters/object_store_repository.py`) – File-backed, minimal
3. **DbDocumentsRepository** (`ai_core/adapters/db_documents_repository.py`) – PostgreSQL, produktionsreif

**Kritische Erkenntnisse**:
- ✅ **Default Production Config**: `DbDocumentsRepository` ist korrekt als Default konfiguriert
- ⚠️ **Semantic Drift**: Erhebliche Unterschiede in Collection-ID-Handling, Lifecycle, Assets
- ⚠️ **ObjectStore**: Unvollständig implementiert (nur `upsert`, `get`, `list_*`), fehlt: `delete`, `add_asset`, etc.
- ✅ **Workbench Dataflow**: Document Explorer nutzt konsistent `_get_documents_repository()` → DB-Store aktiv

---

## 1. Repository-Auswahl-Mechanismus

### 1.1 Konfiguration & Fallback-Logik

**Settings Default** ([noesis2/settings/base.py:196-199](noesis2/settings/base.py#L196-L199)):
```python
DOCUMENTS_REPOSITORY_CLASS = env(
    "DOCUMENTS_REPOSITORY_CLASS",
    default="ai_core.adapters.db_documents_repository.DbDocumentsRepository",
)
```

**Fallback Chain** ([ai_core/services/__init__.py:250-266](ai_core/services/__init__.py#L250-L266)):
```python
def _get_documents_repository() -> DocumentsRepository:
    # 1. Check ai_core.views.DOCUMENTS_REPOSITORY (Test Monkey-Patch)
    try:
        views = import_module("ai_core.views")
        repo = getattr(views, "DOCUMENTS_REPOSITORY", None)
        if isinstance(repo, DocumentsRepository):
            return repo
    except Exception:
        pass

    # 2. Global singleton
    global _DOCUMENTS_REPOSITORY
    if _DOCUMENTS_REPOSITORY is None:
        _DOCUMENTS_REPOSITORY = _build_documents_repository()
    return _DOCUMENTS_REPOSITORY
```

**Factory** ([ai_core/services/__init__.py:227-247](ai_core/services/__init__.py#L227-L247)):
```python
def _build_documents_repository() -> DocumentsRepository:
    # 1. Prüfe settings.DOCUMENTS_REPOSITORY (Instanz oder Callable)
    repository_setting = getattr(settings, "DOCUMENTS_REPOSITORY", None)
    if isinstance(repository_setting, DocumentsRepository):
        return repository_setting
    if callable(repository_setting):
        candidate = repository_setting()
        if isinstance(candidate, DocumentsRepository):
            return candidate

    # 2. Prüfe settings.DOCUMENTS_REPOSITORY_CLASS (String Import)
    repository_class_setting = getattr(settings, "DOCUMENTS_REPOSITORY_CLASS", None)
    if repository_class_setting:
        repository_class = import_string(repository_class_setting)
        candidate = repository_class()
        if not isinstance(candidate, DocumentsRepository):
            raise TypeError("documents_repository_invalid_instance")
        return candidate

    # 3. Fallback: InMemoryDocumentsRepository
    return InMemoryDocumentsRepository()
```

### 1.2 Workbench Integration

**Upload Path** ([ai_core/services/__init__.py:1464-1885](ai_core/services/__init__.py#L1464-L1885)):
```python
def handle_document_upload(...) -> Response:
    # ...
    repository = _get_documents_repository()  # ✅ Zeile 1717
    repository.upsert(normalized_document)
    # ...
```

**Explorer Path** ([theme/views.py:589-670](theme/views.py#L589-L670)):
```python
def document_explorer(request):
    # ...
    repository = _get_documents_repository()  # ✅ Zeile 615
    params = DocumentSpaceRequest(...)
    result = DOCUMENT_SPACE_SERVICE.build_context(
        tenant_id=tenant_id,
        tenant_schema=tenant_schema,
        tenant=tenant_obj,
        params=params,
        repository=repository,  # ✅ Übergeben an Service
    )
    # ...
```

**DocumentSpaceService** ([documents/services/document_space_service.py:69-168](documents/services/document_space_service.py#L69-L168)):
```python
def build_context(self, *, repository: DocumentsRepository, ...) -> DocumentSpaceResult:
    # ...
    list_fn = (
        repository.list_latest_by_collection if params.latest_only
        else repository.list_by_collection  # ✅ Zeile 102-106
    )
    document_refs, next_cursor = list_fn(
        tenant_id=tenant_id,
        collection_id=selected_collection.collection_id,
        limit=params.limit,
        cursor=params.cursor or None,
        workflow_id=params.workflow_filter or None,
    )
    # ...
    for ref in document_refs:
        doc = repository.get(  # ✅ Zeile 204-210
            tenant_id=tenant_id,
            document_id=ref.document_id,
            version=ref.version,
            prefer_latest=latest_only or ref.version is None,
            workflow_id=ref.workflow_id,
        )
    # ...
```

**✅ Ergebnis**: Workbench nutzt konsistent `_get_documents_repository()` → DB-Store ist produktiv aktiv (kein Fallback auf InMemory/ObjectStore).

---

## 2. API-Signatur & Methoden-Vergleich

### 2.1 Interface-Vollständigkeit

| Methode                          | InMemory | ObjectStore | DB | Bemerkung                                      |
|----------------------------------|----------|-------------|----|------------------------------------------------|
| `upsert(doc, workflow_id)`       | ✅       | ✅          | ✅ | Core-API, alle implementiert                   |
| `get(tenant, doc_id, version, …)`| ✅       | ✅          | ✅ | Signatur identisch                             |
| `list_by_collection(…)`          | ✅       | ✅          | ✅ | Cursor/Filter-Semantik **divergiert** (siehe 2.2) |
| `list_latest_by_collection(…)`   | ✅       | ✅          | ✅ | Logik unterschiedlich (siehe 2.3)              |
| `delete(tenant, doc_id, …)`      | ✅       | ❌          | ❌ | **ObjectStore/DB: NICHT implementiert**        |
| `add_asset(asset, workflow_id)`  | ✅       | ❌          | ❌ | **Nur InMemory**                               |
| `get_asset(tenant, asset_id, …)` | ✅       | ❌          | ❌ | **Nur InMemory**                               |
| `list_assets_by_document(…)`     | ✅       | ❌          | ❌ | **Nur InMemory**                               |
| `delete_asset(…)`                | ✅       | ❌          | ❌ | **Nur InMemory**                               |

**❌ KRITISCH**: ObjectStore & DB implementieren **nicht** das vollständige Repository-Interface. Falls Code Assets verwendet oder Dokumente löscht, schlagen Aufrufe mit `NotImplementedError` fehl.

### 2.2 `list_by_collection` – Cursor & Filter-Semantik

#### InMemory ([documents/repository.py:995-1054](documents/repository.py#L995-L1054))

```python
def list_by_collection(
    self,
    tenant_id: str,
    collection_id: UUID,
    limit: int = 100,
    cursor: Optional[str] = None,
    latest_only: bool = False,
    *,
    workflow_id: Optional[str] = None,
) -> Tuple[List[DocumentRef], Optional[str]]:
    # FILTER: collection_id in doc.ref.collection_id (In-Memory-Feld)
    with self._lock:
        entries = [
            self._document_entry(doc)
            for doc in self._iter_documents_locked(
                tenant_id, collection_id, workflow_id
            )
        ]
        entries.sort(key=lambda entry: entry[0])
        start = self._cursor_start(entries, cursor)
        sliced = entries[start : start + limit]
        refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
        next_cursor = self._next_cursor(entries, start, limit)
        return refs, next_cursor
```

**Cursor-Encoding** ([documents/repository.py:1490-1504](documents/repository.py#L1490-L1504)):
```python
def _encode_cursor(parts: Iterable[str]) -> str:
    payload = "|".join(parts)
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8"))
    return encoded.decode("ascii")

# Format: "created_at|document_id|workflow_id|version"
parts = [
    doc.created_at.isoformat(),
    str(doc.ref.document_id),
    doc.ref.workflow_id,
    doc.ref.version or "",
]
```

#### ObjectStore ([ai_core/adapters/object_store_repository.py:120-182](ai_core/adapters/object_store_repository.py#L120-L182))

```python
def list_by_collection(
    self,
    tenant_id: str,
    collection_id: UUID,
    limit: int = 100,
    cursor: Optional[str] = None,
    latest_only: bool = False,
    *,
    workflow_id: Optional[str] = None,
) -> Tuple[List[DocumentRef], Optional[str]]:
    # FILTER: collection_id via file-system Metadata-JSON
    tenant_segment = object_store.sanitize_identifier(tenant_id)
    base = object_store.BASE_PATH / tenant_segment

    entries: List[Tuple[Tuple, NormalizedDocument]] = []
    for wf_dir in base.iterdir():
        if workflow_id and wf_dir.name != object_store.sanitize_identifier(workflow_id):
            continue
        uploads_dir = wf_dir / "uploads"
        for meta_path in uploads_dir.glob("*.meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_collection_id = meta.get("collection_id")
            if meta_collection_id and str(meta_collection_id) == str(collection_id):
                doc = _rebuild_document_from_meta(tenant_id, doc_id, meta, meta_path)
                entries.append(self._document_entry(doc))

    entries.sort(key=lambda entry: entry[0])
    start = self._cursor_start(entries, cursor)
    sliced = entries[start : start + limit]
    refs = [entry[1].ref.model_copy(deep=True) for entry in sliced]
    next_cursor = self._next_cursor(entries, start, limit)
    return refs, next_cursor
```

**Cursor-Encoding**: Identisch mit InMemory (base64-encoded `created_at|doc_id|workflow_id|version`).

#### DB ([ai_core/adapters/db_documents_repository.py:253-290](ai_core/adapters/db_documents_repository.py#L253-L290))

```python
def list_by_collection(
    self,
    tenant_id: str,
    collection_id: UUID,
    limit: int = 100,
    cursor: Optional[str] = None,
    latest_only: bool = False,
    *,
    workflow_id: Optional[str] = None,
) -> Tuple[List[DocumentRef], Optional[str]]:
    # FILTER: collection_id via DocumentCollectionMembership JOIN
    memberships = _collection_queryset(tenant_id, collection_id, workflow_id)
    memberships = _apply_cursor_filter(memberships, cursor)

    entries: list[tuple[tuple, NormalizedDocument]] = []
    for membership in memberships[: limit + 1]:
        document = membership.document
        normalized = _build_document_from_metadata(document)
        if normalized is None:
            continue
        entries.append(self._document_entry(normalized))

    entries.sort(key=lambda entry: entry[0])
    refs = [doc.ref.model_copy(deep=True) for _, doc in entries[:limit]]
    next_cursor = self._next_cursor(entries, 0, limit)
    return refs, next_cursor

def _collection_queryset(tenant_id, collection_id, workflow_id):
    DocumentCollectionMembership = apps.get_model("documents", "DocumentCollectionMembership")
    lifecycle_model = apps.get_model("documents", "DocumentLifecycleState")
    workflow_key = _workflow_storage_key(workflow_id)

    queryset = DocumentCollectionMembership.objects.filter(collection_id=collection_id)

    if workflow_id is not None:
        lifecycle_exists = models.Exists(
            lifecycle_model.objects.filter(
                tenant_id=tenant_id,
                workflow_id=workflow_key,
                document_id=models.OuterRef("document__id"),
            )
        )
        queryset = queryset.annotate(has_lifecycle=lifecycle_exists).filter(has_lifecycle=True)

    return queryset.order_by("-added_at", "document_id")
```

**Cursor-Filter** ([ai_core/adapters/db_documents_repository.py:558-578](ai_core/adapters/db_documents_repository.py#L558-L578)):
```python
def _apply_cursor_filter(queryset, cursor: Optional[str]):
    if not cursor:
        return queryset
    parts = _decode_cursor(cursor)
    timestamp = datetime.fromisoformat(parts[0])
    document_id = UUID(parts[1])

    return queryset.filter(
        models.Q(document__created_at__lt=timestamp)
        | (
            models.Q(document__created_at=timestamp)
            & models.Q(document__id__gt=document_id)
        )
    )
```

### 2.3 Divergenz-Bewertung: `list_by_collection`

| Aspekt                        | InMemory                              | ObjectStore                           | DB                                    |
|-------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| **Collection-Filter**         | `doc.ref.collection_id` (In-Memory)   | Metadata-JSON `collection_id`         | `DocumentCollectionMembership` JOIN   |
| **Workflow-Filter**           | In-Memory Iteration                   | File-System Prefix-Match              | Django ORM `Exists` Subquery          |
| **Cursor-Format**             | Base64 `created_at\|doc_id\|wf\|ver` | Base64 `created_at\|doc_id\|wf\|ver` | Base64 `created_at\|doc_id\|wf\|ver`  |
| **Cursor-Anwendung**          | In-Memory Sort + Slice                | In-Memory Sort + Slice                | **DB-Filter** (ORM `Q` objects)       |
| **Pagination Konsistenz**     | ✅ Deterministisch                    | ⚠️ Abhängig von FS-Scan-Reihenfolge  | ✅ Deterministisch (ORDER BY)        |

**❌ RISIKO: Collection-ID-Semantik**

- **InMemory**: Verwendet `NormalizedDocument.ref.collection_id` (transient, im Speicher)
- **ObjectStore**: Liest `meta["collection_id"]` aus JSON-Datei (serialisiert bei `upsert`)
- **DB**: Verwendet `DocumentCollectionMembership.collection_id` (relationale FK, evtl. **logische UUID**, nicht DB-PK!)

**⚠️ KRITISCHER BUG** ([ai_core/adapters/db_documents_repository.py:136-163](ai_core/adapters/db_documents_repository.py#L136-L163)):
```python
if collection_id:
    try:
        coll = DocumentCollection.objects.get(
            tenant=tenant,
            collection_id=collection_id  # ✅ Logische UUID
        )
        DocumentCollectionMembership.objects.get_or_create(
            document=document,
            collection=coll,
            defaults={"added_by": "system"},
        )
    except DocumentCollection.DoesNotExist:
        # Fallback zu PK-Lookup (ID-Verwirrung!)
        try:
            coll = DocumentCollection.objects.get(pk=collection_id)
            if coll.tenant_id != tenant.id:
                raise DocumentCollection.DoesNotExist
            DocumentCollectionMembership.objects.get_or_create(...)
        except (DocumentCollection.DoesNotExist, ValueError):
            pass  # Silent Failure!
```

**⚠️ PROBLEM**: Wenn `collection_id` (UUID) weder als logische `collection_id` noch als DB-`pk` existiert, wird Membership **silent** nicht erstellt → Dokument ist in Collection „unsichtbar" für `list_by_collection`.

### 2.4 `list_latest_by_collection` – „Latest"-Logik

#### InMemory ([documents/repository.py:1056-1102](documents/repository.py#L1056-L1102))

```python
def list_latest_by_collection(...):
    with self._lock:
        latest: Dict[UUID, NormalizedDocument] = {}
        for doc in self._iter_documents_locked(tenant_id, collection_id, workflow_id):
            current = latest.get(doc.ref.document_id)
            if current is None or self._newer(doc, current):
                latest[doc.ref.document_id] = doc

        entries = [self._document_entry(doc) for doc in latest.values()]
        entries.sort(key=lambda entry: entry[0])
        # ... Pagination wie list_by_collection

@staticmethod
def _newer(left: NormalizedDocument, right: NormalizedDocument) -> bool:
    if left.created_at > right.created_at:
        return True
    if left.created_at < right.created_at:
        return False
    left_version = left.ref.version or ""
    right_version = right.ref.version or ""
    return left_version > right_version  # Lexicographic!
```

**Logik**: Gruppiert nach `document_id`, wählt neuestes `created_at` (Tie-Break: lexicographic `version`).

#### ObjectStore ([ai_core/adapters/object_store_repository.py:184-245](ai_core/adapters/object_store_repository.py#L184-L245))

```python
def list_latest_by_collection(...):
    latest: Dict[UUID, NormalizedDocument] = {}
    for wf_dir in base.iterdir():
        for meta_path in uploads_dir.glob("*.meta.json"):
            # ... metadata filtering
            doc = _rebuild_document_from_meta(tenant_id, doc_id, meta, meta_path)
            current = latest.get(doc.ref.document_id)
            if current is None or self._newer(doc, current):
                latest[doc.ref.document_id] = doc

    entries = [self._document_entry(doc) for doc in latest.values()]
    entries.sort(key=lambda entry: entry[0])
    # ... Pagination

@staticmethod
def _newer(left, right):
    if left.created_at and right.created_at:
        if left.created_at > right.created_at:
            return True
        if left.created_at < right.created_at:
            return False
    left_version = left.ref.version or ""
    right_version = right.ref.version or ""
    return left_version > right_version
```

**Logik**: Identisch mit InMemory, aber `created_at` kann `None` sein (Fallback zu `st_mtime`).

#### DB ([ai_core/adapters/db_documents_repository.py:292-351](ai_core/adapters/db_documents_repository.py#L292-L351))

```python
def list_latest_by_collection(...):
    memberships = _collection_queryset(tenant_id, collection_id, workflow_id)
    memberships = _apply_cursor_filter(memberships, cursor)

    entries: list[tuple[tuple, NormalizedDocument]] = []
    for membership in memberships[: limit + 1]:
        document = membership.document
        normalized = _build_document_from_metadata(document)
        if normalized is None:
            # Error Placeholder (!)
            error_ref = DocumentRef(...)
            error_doc = NormalizedDocument(
                ref=error_ref,
                meta=DocumentMeta(..., title=f"⚠️ ERROR LOADING: {str(e)[:50]}"),
                ...
            )
            entries.append(self._document_entry(error_doc))
            continue
        doc_ref = normalized.ref
        if workflow_id and doc_ref.workflow_id != workflow_id:
            continue
        entries.append(self._document_entry(normalized))

    entries.sort(key=lambda entry: entry[0])
    refs = [doc.ref.model_copy(deep=True) for _, doc in entries[:limit]]
    next_cursor = self._next_cursor(entries, 0, limit)
    return refs, next_cursor
```

**⚠️ ANOMALIE**: DB-Repo gruppiert **nicht** nach `document_id` → liefert **alle Versionen** aus Membership-Join, nicht nur die neueste Version! Dies widerspricht dem Interface-Vertrag von `list_latest_by_collection`.

**❌ KRITISCH**: DocumentSpaceService ruft `list_latest_by_collection` ([documents/services/document_space_service.py:102-106](documents/services/document_space_service.py#L102-L106)):
```python
list_fn = (
    repository.list_latest_by_collection if params.latest_only
    else repository.list_by_collection
)
```

Falls `latest_only=True` und DB-Repo genutzt wird, können **Duplikate** (mehrere Versionen desselben `document_id`) im Explorer erscheinen!

---

## 3. Blob-Handling & Storage-Integration

### 3.1 InMemoryDocumentsRepository

**Storage-Layer**: `ObjectStoreStorage` (default) oder injiziert ([documents/repository.py:894-896](documents/repository.py#L894-L896)):
```python
def __init__(self, storage: Optional[Storage] = None) -> None:
    self._lock = RLock()
    self._storage = storage or ObjectStoreStorage()
    # ...
```

**Blob-Materialisierung** ([documents/repository.py:1445-1485](documents/repository.py#L1445-L1485)):
```python
def _materialize_document(self, doc: NormalizedDocument) -> NormalizedDocument:
    doc.blob = self._materialize_blob(
        doc.blob,
        owner_checksum=doc.checksum,
        checksum_error="document_checksum_mismatch",
    )
    doc.assets = [self._materialize_asset(asset) for asset in doc.assets]
    return doc

def _materialize_blob(
    self,
    blob: BlobLocator,
    *,
    owner_checksum: Optional[str],
    checksum_error: str,
) -> BlobLocator:
    if isinstance(blob, InlineBlob):
        payload = blob.decoded_payload()
        uri, sha256, size = self._storage.put(payload)  # ✅ Persists to ObjectStore
        if blob.sha256 != sha256:
            raise ValueError("inline_checksum_mismatch")
        if owner_checksum is not None and owner_checksum != sha256:
            raise ValueError(checksum_error)
        return FileBlob(type="file", uri=uri, sha256=sha256, size=size)

    blob_sha = getattr(blob, "sha256", None)
    if owner_checksum is not None and blob_sha is not None and blob_sha != owner_checksum:
        raise ValueError(checksum_error)
    return blob  # Already FileBlob → no-op
```

**Verhalten**:
- `InlineBlob` → persists zu ObjectStore → wird `FileBlob`
- `FileBlob` → no-op (bereits persistiert)
- **Checksum-Validierung**: `InlineBlob.sha256` vs. Storage-SHA256 vs. `doc.checksum` (3-way)

### 3.2 ObjectStoreDocumentsRepository

**Storage-Layer**: Keine! Schreibt direkt mit `ai_core.infra.object_store` ([ai_core/adapters/object_store_repository.py:31-54](ai_core/adapters/object_store_repository.py#L31-L54)):
```python
def upsert(self, doc: NormalizedDocument, workflow_id: Optional[str] = None) -> NormalizedDocument:
    tenant_segment = object_store.sanitize_identifier(doc.ref.tenant_id)
    workflow = workflow_id or doc.ref.workflow_id or DEFAULT_WORKFLOW_PLACEHOLDER
    workflow_segment = object_store.sanitize_identifier(workflow)
    uploads_prefix = f"{tenant_segment}/{workflow_segment}/uploads"

    payload = _extract_payload(doc)  # ✅ Extrahiert bytes aus InlineBlob
    checksum = hashlib.sha256(payload).hexdigest()
    filename_base = str(doc.ref.document_id)

    # Persist raw upload
    object_store.write_bytes(
        f"{uploads_prefix}/{filename_base}_upload.bin", payload
    )

    # Persist minimal metadata used by ingestion
    metadata = _build_metadata_snapshot(doc, checksum)
    object_store.write_json(f"{uploads_prefix}/{filename_base}.meta.json", metadata)

    # Return the original model (no mutation)
    return doc  # ❌ WARNUNG: Blob bleibt InlineBlob!
```

**⚠️ PROBLEM**: `upsert` persistiert Payload, gibt aber **unverändertes** `doc` zurück (Blob bleibt `InlineBlob`, kein `FileBlob`). Falls Caller erwartet, dass Blob materialisiert wird (wie bei InMemory), schlägt Re-Parsing fehl.

**Blob-Rekonstruktion** ([ai_core/adapters/object_store_repository.py:401-469](ai_core/adapters/object_store_repository.py#L401-L469)):
```python
def _rebuild_document_from_meta(tenant_id, document_id, meta, meta_path) -> NormalizedDocument:
    upload_path = meta_path.with_name(f"{document_id}_upload.bin")
    payload = upload_path.read_bytes()
    computed_sha = hashlib.sha256(payload).hexdigest()
    sha = checksum if checksum and checksum == computed_sha else computed_sha
    inline_blob = InlineBlob(
        type="inline",
        media_type=media_type or "application/octet-stream",
        base64=base64.b64encode(payload).decode("ascii"),
        sha256=sha,
        size=len(payload),
    )
    # ...
    return NormalizedDocument(ref=doc_ref, meta=doc_meta, blob=inline_blob, ...)
```

**⚠️ RISIKO**: `get()` liefert **InlineBlob**, nicht `FileBlob` → Inkonsistenz mit InMemory/DB-Erwartung.

### 3.3 DbDocumentsRepository

**Storage-Layer**: `ObjectStoreStorage` (default) oder injiziert ([ai_core/adapters/db_documents_repository.py:32-33](ai_core/adapters/db_documents_repository.py#L32-L33)):
```python
def __init__(self, storage: Optional[Storage] = None) -> None:
    self._storage = storage or ObjectStoreStorage()
```

**Blob-Materialisierung** ([ai_core/adapters/db_documents_repository.py:192-210](ai_core/adapters/db_documents_repository.py#L192-L210)):
```python
def _materialize_document_safe(self, doc: NormalizedDocument) -> NormalizedDocument:
    if not isinstance(doc.blob, InlineBlob):
        return doc  # No change needed

    # Prepare new blob
    data = base64.b64decode(doc.blob.base64)
    uri, _, _ = self._storage.put(data)  # ✅ Persists to ObjectStore
    new_blob = FileBlob(
        type="file",
        uri=uri,
        sha256=doc.blob.sha256,
        size=doc.blob.size,
    )

    # Pydantic v2 copy with update (handles frozen models)
    return doc.model_copy(update={"blob": new_blob}, deep=True)
```

**Verhalten**: Identisch mit InMemory, aber explizit `frozen=True`-sicher via `model_copy(update=...)`.

**Metadaten-Persistenz** ([ai_core/adapters/db_documents_repository.py:48-190](ai_core/adapters/db_documents_repository.py#L48-L190)):
```python
def upsert(self, doc, workflow_id):
    doc_copy = self._materialize_document_safe(doc)
    # ...
    metadata = {"normalized_document": doc_copy.model_dump(mode="json")}
    # ...
    document = Document.objects.create(
        id=doc_copy.ref.document_id,
        tenant=tenant,
        hash=doc_copy.checksum,
        source=doc_copy.source or "",
        metadata=metadata,
        lifecycle_state=doc_copy.lifecycle_state,
        lifecycle_updated_at=doc_copy.created_at,
    )
    # ...
```

**Rekonstruktion** ([ai_core/adapters/db_documents_repository.py:396-516](ai_core/adapters/db_documents_repository.py#L396-L516)):
```python
def _build_document_from_metadata(document) -> Optional[NormalizedDocument]:
    payload = document.metadata or {}
    normalized_payload = payload.get("normalized_document")

    if normalized_payload:
        normalized = NormalizedDocument.model_validate(normalized_payload)
    else:
        # Lazy Normalization: Shim from raw input metadata (Crawler)
        normalized = _shim_normalized_document(document, payload)

    normalized.created_at = document.created_at  # Align with DB timestamp
    return normalized
```

**⚠️ KRITISCH: Shim-Logik** ([ai_core/adapters/db_documents_repository.py:419-516](ai_core/adapters/db_documents_repository.py#L419-L516)):
- Falls `metadata.normalized_document` fehlt (z.B. roher Crawler-Input), wird ein **partieller NormalizedDocument-Shim** konstruiert
- Blob wird aus `payload.blob` rekonstruiert (inline_text → base64, etc.)
- **PROBLEM**: Shim-Logik ist **fragil** und kann bei neuen Crawler-Formaten fehlschlagen

---

## 4. Lifecycle & Assets

### 4.1 Lifecycle-Handling

| Repository   | Lifecycle-Speicherung                          | Abruf                                   | Semantik                              |
|--------------|------------------------------------------------|-----------------------------------------|---------------------------------------|
| **InMemory** | `DocumentLifecycleStore` (In-Memory Dict)      | `get_document_state()`                  | Transient, key: `(tenant, wf, docID)` |
| **ObjectStore** | ❌ KEINE (nur Metadata-JSON)                | ❌ KEINE                                | **Nicht implementiert**               |
| **DB**       | `DocumentLifecycleState` (Django Model)        | JOIN in `_build_document_from_metadata` | Persistent, DB-Tabelle                |

**⚠️ RISIKO: ObjectStore**:
- Lifecycle-State wird **nie** persistiert (nur initial `doc.lifecycle_state`)
- Änderungen via `record_document_state()` gehen verloren
- **IMPACT**: Document Explorer zeigt falschen Lifecycle-Status für ObjectStore-Dokumente

### 4.2 Assets

| Methode                      | InMemory                                 | ObjectStore | DB   |
|------------------------------|------------------------------------------|-------------|------|
| `add_asset(asset, wf)`       | ✅ Persists zu Storage + Index           | ❌          | ❌   |
| `get_asset(tenant, asset_id)`| ✅ Aus Index + Storage                   | ❌          | ❌   |
| `list_assets_by_document(…)` | ✅ Index-basiert, sortiert               | ❌          | ❌   |
| `delete_asset(…, hard)`      | ✅ Soft/Hard Delete + Index-Update       | ❌          | ❌   |

**❌ KRITISCH**:
- **ObjectStore & DB**: Asset-API komplett fehlend
- **IMPACT**: Code, der Assets verwendet (z.B. extrahierte Tabellen, PDFs mit eingebetteten Bildern), schlägt mit `NotImplementedError` fehl bei DB/ObjectStore

**InMemory-Asset-Index** ([documents/repository.py:1300-1342](documents/repository.py#L1300-L1342)):
```python
# _assets: Dict[Tuple[str, str, UUID], _StoredAsset]
# _asset_index: Dict[Tuple[str, str, UUID], set[UUID]]  # (tenant, wf, doc_id) -> asset_ids

def _store_asset_locked(self, asset: Asset) -> None:
    key = (asset.ref.tenant_id, asset.ref.workflow_id, asset.ref.asset_id)
    self._assets[key] = _StoredAsset(value=asset_copy, deleted=False)

    index_key = (asset.ref.tenant_id, asset.ref.workflow_id, asset.ref.document_id)
    bucket = self._asset_index.setdefault(index_key, set())
    bucket.add(asset.ref.asset_id)
```

**Keine DB/ObjectStore-Äquivalenz** → Asset-Features unbrauchbar mit DB/ObjectStore.

---

## 5. Collection-ID-Semantik & Membership

### 5.1 InMemory

**Collection-Zuordnung**: Direkt in `doc.ref.collection_id` (transient, UUID) ([documents/repository.py:996-1010](documents/repository.py#L996-L1010)):
```python
def list_by_collection(...):
    for doc in self._iter_documents_locked(tenant_id, collection_id, workflow_id):
        if collection_id is not None and doc.ref.collection_id != collection_id:
            continue
        yield doc
```

**Keine Persistenz** außer im Dokument-Objekt selbst.

### 5.2 ObjectStore

**Collection-Zuordnung**: Via `meta["collection_id"]` in JSON-Datei ([ai_core/adapters/object_store_repository.py:156-174](ai_core/adapters/object_store_repository.py#L156-L174)):
```python
for meta_path in uploads_dir.glob("*.meta.json"):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_collection_id = meta.get("collection_id")
    if meta_collection_id and str(meta_collection_id) == str(collection_id):
        doc = _rebuild_document_from_meta(...)
        entries.append(self._document_entry(doc))
```

**Schreiben** ([ai_core/adapters/object_store_repository.py:366-398](ai_core/adapters/object_store_repository.py#L366-L398)):
```python
def _build_metadata_snapshot(doc: NormalizedDocument, checksum: str) -> dict:
    payload = {
        "workflow_id": doc.ref.workflow_id,
        "document_id": str(doc.ref.document_id),
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "checksum": checksum,
        "media_type": getattr(doc.blob, "media_type", None),
        "source": doc.source,
    }

    if doc.ref.collection_id is not None:
        payload["collection_id"] = str(doc.ref.collection_id)  # ✅ Serialisiert
    # ...
    return payload
```

**⚠️ PROBLEM**: `collection_id` ist **rein metadatenbasiert** → kein Constraint, keine Referential Integrity.

### 5.3 DB

**Collection-Zuordnung**: Via `DocumentCollectionMembership` (Many-to-Many Join-Table) ([ai_core/adapters/db_documents_repository.py:519-544](ai_core/adapters/db_documents_repository.py#L519-L544)):
```python
def _collection_queryset(tenant_id, collection_id, workflow_id):
    DocumentCollectionMembership = apps.get_model("documents", "DocumentCollectionMembership")
    lifecycle_model = apps.get_model("documents", "DocumentLifecycleState")
    workflow_key = _workflow_storage_key(workflow_id)

    queryset = DocumentCollectionMembership.objects.filter(collection_id=collection_id)

    if workflow_id is not None:
        lifecycle_exists = models.Exists(
            lifecycle_model.objects.filter(
                tenant_id=tenant_id,
                workflow_id=workflow_key,
                document_id=models.OuterRef("document__id"),
            )
        )
        queryset = queryset.annotate(has_lifecycle=lifecycle_exists).filter(has_lifecycle=True)

    return queryset.order_by("-added_at", "document_id")
```

**Schreiben** ([ai_core/adapters/db_documents_repository.py:136-163](ai_core/adapters/db_documents_repository.py#L136-L163)):
```python
if collection_id:
    try:
        coll = DocumentCollection.objects.get(
            tenant=tenant,
            collection_id=collection_id  # ✅ Logische UUID
        )
        DocumentCollectionMembership.objects.get_or_create(
            document=document,
            collection=coll,
            defaults={"added_by": "system"},
        )
    except DocumentCollection.DoesNotExist:
        # Fallback zu PK-Lookup
        try:
            coll = DocumentCollection.objects.get(pk=collection_id)
            if coll.tenant_id != tenant.id:
                raise DocumentCollection.DoesNotExist
            DocumentCollectionMembership.objects.get_or_create(...)
        except (DocumentCollection.DoesNotExist, ValueError):
            pass  # ❌ SILENT FAILURE!
```

**❌ KRITISCHER BUG**:
- Wenn `collection_id` (UUID) **nicht** als logische `collection_id` **oder** DB-PK existiert, wird Membership **silent** nicht erstellt
- **IMPACT**: Dokument ist in Collection „unsichtbar" für `list_by_collection`, obwohl `upsert` erfolgreich war
- **ROOT CAUSE**: ID-Verwirrung zwischen logischer `collection_id` (UUID aus Contracts) und DB-PK (Auto-Increment)

**✅ KORREKTUR** (Pseudo-Code):
```python
if collection_id:
    try:
        coll = DocumentCollection.objects.get(tenant=tenant, collection_id=collection_id)
    except DocumentCollection.DoesNotExist:
        logger.warning(
            "document_upsert.collection_missing",
            extra={"tenant_id": tenant_id, "collection_id": collection_id},
        )
        # OPTION 1: Raise Exception (strict)
        raise ValueError(f"collection_not_found: {collection_id}")
        # OPTION 2: Auto-Create (permissive, WARNUNG!)
        # coll = DocumentCollection.objects.create(tenant=tenant, collection_id=collection_id, ...)

    DocumentCollectionMembership.objects.get_or_create(
        document=document,
        collection=coll,
        defaults={"added_by": "system"},
    )
```

---

## 6. Workflow-ID-Handling

### 6.1 Storage-Key-Normalisierung

**InMemory** ([documents/repository.py:368-376](documents/repository.py#L368-L376)):
```python
def _workflow_storage_key(value: Optional[str]) -> str:
    return (str(value).strip() if value else "").strip()

def _workflow_from_storage(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None
```

**DB** ([ai_core/adapters/db_documents_repository.py:19-22](ai_core/adapters/db_documents_repository.py#L19-L22)):
```python
from documents.repository import _workflow_storage_key  # ✅ Import

workflow_key = _workflow_storage_key(workflow)  # ✅ Verwendet dieselbe Funktion
```

**ObjectStore** ([ai_core/adapters/object_store_repository.py:36-37](ai_core/adapters/object_store_repository.py#L36-L37)):
```python
workflow = workflow_id or doc.ref.workflow_id or DEFAULT_WORKFLOW_PLACEHOLDER
workflow_segment = object_store.sanitize_identifier(workflow)  # ❌ DIVERGENT!
```

**⚠️ DIVERGENZ**:
- **InMemory/DB**: `_workflow_storage_key()` (Strip whitespace)
- **ObjectStore**: `object_store.sanitize_identifier()` (ersetzt `/`, `:`, `.` → `_`, lowercase, etc.)

**IMPACT**: Gleicher `workflow_id` kann zu unterschiedlichen Pfaden/Keys führen → Cross-Repo-Inkompatibilität.

**ObjectStore `sanitize_identifier`** (vermutlich in `ai_core/infra/object_store.py`):
```python
def sanitize_identifier(value: str) -> str:
    # Hypothetical implementation (need to verify)
    return value.lower().replace("/", "_").replace(":", "_").replace(".", "_")
```

**Beispiel**:
- Input: `workflow_id = "Case:123/v2"`
- InMemory/DB: `"Case:123/v2"` (unverändert)
- ObjectStore: `"case_123_v2"` (sanitized)

→ `workflow_id`-Filter schlägt fehl bei Cross-Repo-Queries.

---

## 7. Cursor-Kompatibilität & Pagination

### 7.1 Cursor-Format

**Alle drei Repos** nutzen denselben Encoding ([documents/repository.py:1490-1494](documents/repository.py#L1490-L1494), [ai_core/adapters/object_store_repository.py:250-253](ai_core/adapters/object_store_repository.py#L250-L253), [ai_core/adapters/db_documents_repository.py:381-384](ai_core/adapters/db_documents_repository.py#L381-L384)):
```python
def _encode_cursor(parts: list[str]) -> str:
    payload = "|".join(parts)
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8"))
    return encoded.decode("ascii")

# Format: "created_at|document_id|workflow_id|version"
```

### 7.2 Cursor-Anwendung

| Repo         | Methode                         | Determinismus                          |
|--------------|---------------------------------|----------------------------------------|
| **InMemory** | In-Memory Sort + Slice          | ✅ Deterministisch                     |
| **ObjectStore** | In-Memory Sort + Slice       | ⚠️ Abhängig von FS-Scan-Reihenfolge   |
| **DB**       | Django ORM `Q` Filter           | ✅ Deterministisch (SQL ORDER BY)     |

**⚠️ RISIKO: ObjectStore**:
```python
for wf_dir in base.iterdir():  # ❌ FS-Reihenfolge nicht garantiert!
    for meta_path in uploads_dir.glob("*.meta.json"):  # ❌ Glob-Reihenfolge instabil
        # ...
```

**IMPACT**: Bei großen Collections kann Pagination inkonsistent sein (gleicher Cursor → unterschiedliche Results bei Neustart).

**✅ FIX**: Pre-Sort alle Paths **vor** Filter/Pagination:
```python
meta_paths = sorted(uploads_dir.glob("*.meta.json"), key=lambda p: p.name)
for meta_path in meta_paths:
    # ...
```

---

## 8. Risiko-Bewertung & Priorisierung

### 8.1 Schwere-Kategorien

| Schwere       | Definition                                                                                          | Beispiele                                                                                   |
|---------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **KRITISCH**  | Silent Data Loss, Inkonsistenz, Crash bei Nutzung im Workbench                                     | Collection Membership nicht erstellt, Asset-API fehlt, `list_latest` liefert Duplikate     |
| **HOCH**      | Feature-Diskrepanz, die zu falschen Ergebnissen führt (aber kein Crash)                             | Lifecycle nicht persistiert (ObjectStore), Workflow-ID-Mismatch (ObjectStore)              |
| **MITTEL**    | Performance/Pagination-Probleme, nicht-deterministische Reihenfolge                                 | ObjectStore Cursor instabil, Shim-Logik fragil                                              |
| **NIEDRIG**   | Fehlende Implementierung von selten genutzten Features                                              | `delete()` nicht implementiert (ObjectStore/DB), HTTP-Fetch-Header-Unterschiede (Storage)  |

### 8.2 Findings-Tabelle

| # | Schwere      | Fundort                                                  | Beschreibung                                                                                                                                                                 | IMPACT auf Workbench                                                                                              |
|---|--------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 1 | KRITISCH     | `db_documents_repository.py:136-163`                     | **Collection Membership Silent Failure**: Falls `collection_id` nicht als logische UUID oder PK existiert, wird Membership nicht erstellt → Dokument unsichtbar in Explorer | ❌ Dokumente fehlen in Collection-Listing (obwohl `upsert` erfolgreich)                                          |
| 2 | KRITISCH     | `db_documents_repository.py:292-351`                     | **`list_latest_by_collection` liefert Duplikate**: Gruppiert nicht nach `document_id` → alle Versionen werden zurückgegeben, nicht nur neueste                              | ❌ Document Explorer zeigt mehrfach dieselben `document_id` (unterschiedliche Versionen) bei `latest_only=True`  |
| 3 | KRITISCH     | `object_store_repository.py` (global)                    | **Asset-API fehlt komplett**: `add_asset`, `get_asset`, `list_assets_by_document`, `delete_asset` nicht implementiert                                                      | ❌ Code mit Assets schlägt mit `NotImplementedError` fehl                                                         |
| 4 | HOCH         | `object_store_repository.py:120-182`                     | **Lifecycle nicht persistiert**: `upsert` persistiert nur initial `doc.lifecycle_state`, Updates via `record_document_state()` gehen verloren                               | ⚠️ Document Explorer zeigt falschen Lifecycle-Status (z.B. bleibt „active" statt „retired")                       |
| 5 | HOCH         | `object_store_repository.py:36-37`                       | **Workflow-ID Divergenz**: `sanitize_identifier()` vs. `_workflow_storage_key()` → gleicher `workflow_id` führt zu unterschiedlichen Pfaden/Keys                            | ⚠️ Workflow-Filter schlägt fehl bei Cross-Repo-Queries (ObjectStore vs. DB/InMemory)                             |
| 6 | HOCH         | `object_store_repository.py:31-54`                       | **Blob bleibt InlineBlob**: `upsert` gibt unverändertes `doc` zurück (Blob nicht zu `FileBlob` konvertiert)                                                                | ⚠️ Caller erwartet materialisiertes Blob → Re-Parsing schlägt fehl                                               |
| 7 | MITTEL       | `object_store_repository.py:143-174`                     | **Non-deterministische Pagination**: `base.iterdir()` und `glob()` haben instabile Reihenfolge → gleicher Cursor kann unterschiedliche Results liefern                      | ⚠️ Pagination inkonsistent bei großen Collections (Explorer zeigt unterschiedliche Seiten bei Reload)            |
| 8 | MITTEL       | `db_documents_repository.py:419-516`                     | **Shim-Logik fragil**: Fallback für rohe Crawler-Inputs konstruiert partiellen `NormalizedDocument` → kann bei neuen Formaten fehlschlagen                                 | ⚠️ Explorer crasht oder zeigt „⚠️ ERROR LOADING" für Crawler-Dokumente ohne `normalized_document` Metadaten     |
| 9 | MITTEL       | `db_documents_repository.py:318-346`                     | **Error-Placeholder-Dokumente**: Bei `_build_document_from_metadata` Exception wird „Error-Doc" mit Titel `⚠️ ERROR LOADING` eingefügt                                      | ℹ️ Explorer zeigt Error-Placeholder statt Silent Fail (gut für Debugging, aber kein echter Fix)                  |
| 10| NIEDRIG      | `object_store_repository.py`, `db_documents_repository` | **`delete()` nicht implementiert**: `NotImplementedError` bei Aufruf                                                                                                        | ℹ️ Workbench nutzt kein `delete()` (nur `list_*` + `get`)                                                        |
| 11| NIEDRIG      | `storage.py:110-122`                                     | **HTTP-Fetch-Header-Unterschiede**: `ObjectStoreStorage` nutzt `CRAWLER_HTTP_USER_AGENT`, `InMemoryStorage` nicht                                                          | ℹ️ Nur relevant bei HTTP-URI-Blobs (selten in Workbench)                                                         |

---

## 9. Workbench Dataflow-Analyse

### 9.1 Upload → Ingestion → DB-Store

**Flow** ([ai_core/services/__init__.py:1464-1885](ai_core/services/__init__.py#L1464-L1885)):
```
1. HTMX POST /rag-tools/upload
   ↓
2. theme/views.py:ingestion_submit (vermutlich)
   ↓
3. ai_core.services.handle_document_upload(upload, metadata_raw, meta, idempotency_key)
   ↓
4. _get_documents_repository().upsert(normalized_document)  # Zeile 1717
   ↓
5. DB-Store: _materialize_document_safe → ObjectStoreStorage.put → FileBlob
   ↓
6. Document.objects.create(metadata={"normalized_document": doc_copy.model_dump()})
   ↓
7. DocumentCollectionMembership.objects.get_or_create(...)  # ✅ Zeile 143-147
   ↓
8. DocumentLifecycleState.objects.update_or_create(...)  # ✅ Zeile 165-175
   ↓
9. Ingestion queued: run_ingestion.delay(...)
```

**✅ BESTÄTIGT**: Upload nutzt DB-Store → Dokument wird in DB persistiert (inkl. Membership + Lifecycle).

**⚠️ RISIKO**: Falls `collection_id` nicht existiert, **scheitert Membership-Erstellung silent** (Finding #1).

### 9.2 Document Explorer → DB-Store

**Flow** ([theme/views.py:589-670](theme/views.py#L589-L670)):
```
1. GET /rag-tools/document-explorer?collection=XXX&latest=1&limit=25
   ↓
2. theme/views.py:document_explorer(request)
   ↓
3. repository = _get_documents_repository()  # ✅ Zeile 615 → DB-Store
   ↓
4. params = DocumentSpaceRequest(...)
   ↓
5. DOCUMENT_SPACE_SERVICE.build_context(repository=repository, ...)
   ↓
6. repository.list_latest_by_collection(...)  # ✅ Zeile 103 (falls latest_only=True)
   ↓
7. _collection_queryset → DocumentCollectionMembership JOIN
   ↓
8. for membership in memberships[:limit+1]:
       doc = repository.get(tenant_id, ref.document_id, ...)  # ✅ Zeile 204
       _build_document_from_metadata(document)
   ↓
9. Lifecycle JOIN: _load_lifecycle_states → DocumentLifecycleState.objects.filter(...)
   ↓
10. Render: theme/partials/tool_documents.html
```

**✅ BESTÄTIGT**: Explorer nutzt DB-Store → Collection-Membership via JOIN, Lifecycle via separatem Query.

**❌ PROBLEM**: `list_latest_by_collection` gruppiert **nicht** nach `document_id` → liefert **Duplikate** (Finding #2).

### 9.3 ENV-Override-Risiko

**Hypothetischer Fallback** (`.env` / Settings):
```bash
# Falls folgende ENV gesetzt ist:
DOCUMENTS_REPOSITORY_CLASS=documents.repository.InMemoryDocumentsRepository
# ODER
DOCUMENTS_REPOSITORY_CLASS=ai_core.adapters.object_store_repository.ObjectStoreDocumentsRepository
```

**IMPACT**:
- ✅ Upload funktioniert (InMemory/ObjectStore haben `upsert`)
- ⚠️ Explorer mit ObjectStore: Non-deterministische Pagination (Finding #7), Lifecycle fehlt (Finding #4)
- ❌ Explorer mit ObjectStore `latest_only=True`: **Crash** oder falsche Results (keine Duplikat-Filter)
- ❌ InMemory: Daten verloren bei Neustart (transient)

**✅ BESTÄTIGT**: Produktions-Config (`base.py:196-199`) zeigt `DbDocumentsRepository` als Default → **kein ENV-Override in aktueller .env** (muss validiert werden).

### 9.4 Test-Monkey-Patch-Risiko

**Test-Fixture** ([ai_core/tests/conftest.py:115-124](ai_core/tests/conftest.py#L115-L124)):
```python
@pytest.fixture
def documents_repository(monkeypatch):
    repository = InMemoryDocumentsRepository()
    monkeypatch.setattr(
        "DOCUMENTS_REPOSITORY_CLASS",
        "documents.repository.InMemoryDocumentsRepository",
    )
    monkeypatch.setattr(views, "DOCUMENTS_REPOSITORY", repository, raising=False)
    monkeypatch.setattr(services, "_DOCUMENTS_REPOSITORY", None, raising=False)
    return repository
```

**IMPACT**: Tests nutzen InMemory → **Asset-API funktioniert in Tests**, aber **schlägt in Prod fehl** (DB-Repo hat keine Assets).

**⚠️ RISIKO**: Test-Coverage täuscht Feature-Support vor, der in Prod nicht existiert.

---

## 10. Fix-Empfehlungen

### 10.1 KRITISCH – Sofort

#### Fix #1: Collection Membership Silent Failure

**Datei**: [ai_core/adapters/db_documents_repository.py:136-163](ai_core/adapters/db_documents_repository.py#L136-L163)

**Aktueller Code**:
```python
if collection_id:
    try:
        coll = DocumentCollection.objects.get(tenant=tenant, collection_id=collection_id)
        DocumentCollectionMembership.objects.get_or_create(...)
    except DocumentCollection.DoesNotExist:
        try:
            coll = DocumentCollection.objects.get(pk=collection_id)
            # ...
        except (DocumentCollection.DoesNotExist, ValueError):
            pass  # ❌ SILENT FAILURE!
```

**Korrektur**:
```python
if collection_id:
    try:
        coll = DocumentCollection.objects.get(tenant=tenant, collection_id=collection_id)
    except DocumentCollection.DoesNotExist:
        logger.error(
            "document_upsert.collection_not_found",
            extra={"tenant_id": tenant.schema_name, "collection_id": collection_id},
        )
        raise ValueError(f"collection_not_found: {collection_id}")

    DocumentCollectionMembership.objects.get_or_create(
        document=document,
        collection=coll,
        defaults={"added_by": "system"},
    )
```

**Alternative (permissive, mit Warnung)**:
```python
if collection_id:
    coll, created = DocumentCollection.objects.get_or_create(
        tenant=tenant,
        collection_id=collection_id,
        defaults={
            "name": f"Auto-Collection {collection_id}",
            "key": str(collection_id),
            "type": "auto",
            "visibility": "tenant",
        },
    )
    if created:
        logger.warning(
            "document_upsert.collection_auto_created",
            extra={"tenant_id": tenant.schema_name, "collection_id": collection_id},
        )

    DocumentCollectionMembership.objects.get_or_create(
        document=document,
        collection=coll,
        defaults={"added_by": "system"},
    )
```

#### Fix #2: `list_latest_by_collection` Duplikate

**Datei**: [ai_core/adapters/db_documents_repository.py:292-351](ai_core/adapters/db_documents_repository.py#L292-L351)

**Aktueller Code**:
```python
def list_latest_by_collection(...):
    memberships = _collection_queryset(tenant_id, collection_id, workflow_id)
    # ... direkt zu entries ohne Gruppierung
    entries: list[tuple[tuple, NormalizedDocument]] = []
    for membership in memberships[: limit + 1]:
        document = membership.document
        normalized = _build_document_from_metadata(document)
        entries.append(self._document_entry(normalized))
    # ...
```

**Korrektur**:
```python
def list_latest_by_collection(...):
    memberships = _collection_queryset(tenant_id, collection_id, workflow_id)
    memberships = _apply_cursor_filter(memberships, cursor)

    # Gruppiere nach document_id, wähle neueste Version
    latest: dict[UUID, NormalizedDocument] = {}
    for membership in memberships:
        document = membership.document
        normalized = _build_document_from_metadata(document)
        if normalized is None:
            continue

        doc_id = normalized.ref.document_id
        if workflow_id and normalized.ref.workflow_id != workflow_id:
            continue

        current = latest.get(doc_id)
        if current is None or self._newer(normalized, current):
            latest[doc_id] = normalized

    entries = [self._document_entry(doc) for doc in latest.values()]
    entries.sort(key=lambda entry: entry[0])
    refs = [doc.ref.model_copy(deep=True) for _, doc in entries[:limit]]
    next_cursor = self._next_cursor(entries, 0, limit)
    return refs, next_cursor

@staticmethod
def _newer(left: NormalizedDocument, right: NormalizedDocument) -> bool:
    if left.created_at > right.created_at:
        return True
    if left.created_at < right.created_at:
        return False
    left_version = left.ref.version or ""
    right_version = right.ref.version or ""
    return left_version > right_version
```

### 10.2 HOCH – Kurz- bis Mittelfristig

#### Fix #3: ObjectStore Lifecycle-Persistenz

**Option A (Quick Fix)**: Disable ObjectStore in Prod (nur DB nutzen)
**Option B (Feature-Parity)**: Erweitere ObjectStore um Lifecycle-JSON:
```python
# In upsert():
lifecycle_json = {
    "state": doc.lifecycle_state,
    "changed_at": doc.created_at.isoformat(),
    "workflow_id": workflow,
}
object_store.write_json(f"{uploads_prefix}/{filename_base}.lifecycle.json", lifecycle_json)

# In get():
lifecycle_path = meta_path.with_name(f"{document_id}.lifecycle.json")
if lifecycle_path.exists():
    lifecycle_data = json.loads(lifecycle_path.read_text())
    doc.lifecycle_state = lifecycle_data["state"]
```

#### Fix #4: Workflow-ID Normalisierung in ObjectStore

**Datei**: [ai_core/adapters/object_store_repository.py:36-37](ai_core/adapters/object_store_repository.py#L36-L37)

**Aktueller Code**:
```python
workflow = workflow_id or doc.ref.workflow_id or DEFAULT_WORKFLOW_PLACEHOLDER
workflow_segment = object_store.sanitize_identifier(workflow)
```

**Korrektur**:
```python
from documents.repository import _workflow_storage_key

workflow = workflow_id or doc.ref.workflow_id or DEFAULT_WORKFLOW_PLACEHOLDER
workflow_key = _workflow_storage_key(workflow)  # ✅ Konsistent mit InMemory/DB
workflow_segment = object_store.sanitize_identifier(workflow_key)  # Falls FS-Sicherheit nötig
```

#### Fix #5: ObjectStore Blob-Materialisierung

**Datei**: [ai_core/adapters/object_store_repository.py:31-54](ai_core/adapters/object_store_repository.py#L31-L54)

**Aktueller Code**:
```python
def upsert(self, doc, workflow_id):
    # ... write payload, metadata
    return doc  # ❌ Blob bleibt InlineBlob
```

**Korrektur**:
```python
def upsert(self, doc, workflow_id):
    # ... write payload, metadata

    # Materialisiere Blob zu FileBlob
    upload_path = f"{uploads_prefix}/{filename_base}_upload.bin"
    file_blob = FileBlob(
        type="file",
        uri=upload_path,
        sha256=checksum,
        size=len(payload),
    )

    # Return copy mit materialisiertem Blob
    return doc.model_copy(update={"blob": file_blob}, deep=True)
```

### 10.3 MITTEL – Mittelfristig

#### Fix #6: ObjectStore deterministische Pagination

**Datei**: [ai_core/adapters/object_store_repository.py:143-174](ai_core/adapters/object_store_repository.py#L143-L174)

**Aktueller Code**:
```python
for wf_dir in base.iterdir():
    for meta_path in uploads_dir.glob("*.meta.json"):
        # ...
```

**Korrektur**:
```python
# Pre-sort directories and files
wf_dirs = sorted(base.iterdir(), key=lambda p: p.name)
for wf_dir in wf_dirs:
    if not wf_dir.is_dir():
        continue
    # ...
    uploads_dir = wf_dir / "uploads"
    if not uploads_dir.exists():
        continue

    meta_paths = sorted(uploads_dir.glob("*.meta.json"), key=lambda p: p.name)
    for meta_path in meta_paths:
        # ...
```

#### Fix #7: DB Shim-Logik robuster machen

**Datei**: [ai_core/adapters/db_documents_repository.py:419-516](ai_core/adapters/db_documents_repository.py#L419-L516)

**Empfehlung**:
- Erweitere Shim um Validierung & Logging für fehlende/malformierte Felder
- Fallback zu Error-Placeholder mit detaillierter Nachricht (statt silent None)
- Erwäge Migration: Crawler soll **immer** `normalized_document` schreiben (keine Shim-Abhängigkeit)

```python
def _shim_normalized_document(document_model, payload: dict) -> NormalizedDocument:
    try:
        # ... existing shim logic
        return NormalizedDocument(...)
    except Exception as e:
        logger.error(
            "documents.shim_normalization_failed",
            extra={
                "document_id": document_model.id,
                "tenant": document_model.tenant.schema_name,
                "error": str(e),
                "payload_keys": list(payload.keys()),
            },
        )
        raise  # Oder Error-Placeholder zurückgeben
```

### 10.4 NIEDRIG – Optional

#### Fix #8: Implementiere `delete()` für ObjectStore/DB

**Datei**: ObjectStore & DB

**Empfehlung**: Falls `delete()` benötigt wird (z.B. für GDPR), implementiere:
- **ObjectStore**: Lösche `_upload.bin` + `.meta.json` + `.lifecycle.json`
- **DB**: Soft-Delete via `Document.lifecycle_state = "deleted"` **oder** Hard-Delete via `document.delete()`

**Alternativ**: Dokumentiere explizit, dass nur InMemory `delete()` unterstützt (für Tests).

---

## 11. Validierungsplan (Manuelle Steps)

### 11.1 Lokales Dev-Setup validieren

**Ziel**: Bestätigen, dass `.env` auf `DbDocumentsRepository` zeigt.

```bash
# 1. Prüfe .env
grep "DOCUMENTS_REPOSITORY_CLASS" .env
# Erwartung: Entweder NICHT gesetzt (Default = DB) oder explizit:
# DOCUMENTS_REPOSITORY_CLASS=ai_core.adapters.db_documents_repository.DbDocumentsRepository

# 2. Prüfe aktive Repo-Klasse (Django Shell)
npm run dev:manage shell
```

```python
from ai_core.services import _get_documents_repository
repo = _get_documents_repository()
print(type(repo))
# Erwartung: <class 'ai_core.adapters.db_documents_repository.DbDocumentsRepository'>
```

### 11.2 Workbench Upload → Explorer Round-Trip

**Setup**: Dev-Stack läuft (`npm run dev:stack`).

**Steps**:
1. Öffne `http://dev.localhost:8000/rag-tools`
2. **Upload**:
   - Wähle Datei (z.B. PDF)
   - Metadata: `{"collection_id": "<existing-collection-uuid>"}`
     *(Hinweis: `<existing-collection-uuid>` muss existieren! Nutze `manual-dev-docs` oder erstelle via Admin)*
   - Submit
   - **Erwartung**: HTTP 202, `document_id` & `ingestion_run_id` in Response
3. **Explorer**:
   - Navigiere zu Document Space (`/rag-tools/document-space`)
   - Wähle Collection (sollte das hochgeladene Dokument enthalten)
   - **Erwartung**: Dokument erscheint in Liste
   - Prüfe:
     - `document_id` stimmt mit Upload überein
     - `lifecycle_state` = "active"
     - `collection_id` = `<existing-collection-uuid>`
     - **KEIN Duplikat** (gleicher `document_id` mehrfach)
4. **Pagination**:
   - Falls >25 Dokumente, prüfe "Next"-Link
   - **Erwartung**: Cursor-Navigation funktioniert, keine Duplikate über Seiten hinweg
5. **Workflow-Filter**:
   - Setze `workflow=<workflow_id>` (z.B. aus Upload)
   - **Erwartung**: Nur Dokumente mit diesem Workflow erscheinen

### 11.3 Collection Membership Validierung (DB)

**Setup**: Django Shell.

```python
from documents.models import DocumentCollection, DocumentCollectionMembership, Document
from customers.models import Tenant

tenant = Tenant.objects.get(schema_name="dev")
collection_id = "<existing-collection-uuid>"

# 1. Prüfe Collection existiert
coll = DocumentCollection.objects.get(tenant=tenant, collection_id=collection_id)
print(f"Collection: {coll.name} (PK={coll.id})")

# 2. Prüfe Memberships
memberships = DocumentCollectionMembership.objects.filter(collection=coll)
print(f"Memberships: {memberships.count()}")

# 3. Prüfe, ob hochgeladenes Dokument Membership hat
doc_id = "<uploaded-document-id>"
doc = Document.objects.get(tenant=tenant, id=doc_id)
membership = DocumentCollectionMembership.objects.filter(
    document=doc,
    collection=coll,
).first()
print(f"Membership exists: {membership is not None}")
# Erwartung: True
```

### 11.4 Lifecycle-Persistenz Validierung

**Setup**: Django Shell.

```python
from documents.models import DocumentLifecycleState

tenant_id = "dev"
doc_id = "<uploaded-document-id>"
workflow_id = "<workflow_id_from_upload>"

# Prüfe Lifecycle-Record
lifecycle = DocumentLifecycleState.objects.filter(
    tenant_id=tenant_id,
    document_id=doc_id,
    workflow_id=workflow_id or "",
).first()

print(f"Lifecycle: state={lifecycle.state}, changed_at={lifecycle.changed_at}")
# Erwartung: state="active", changed_at=Upload-Timestamp
```

### 11.5 Negative Tests

#### Test 1: Collection nicht existent

**Setup**: Upload mit non-existentem `collection_id`.

```bash
curl -X POST http://dev.localhost:8000/rag-tools/upload \
  -H "X-Tenant-ID: dev" \
  -F "file=@test.pdf" \
  -F "metadata={\"collection_id\": \"00000000-0000-0000-0000-000000000000\"}"
```

**Erwartung (aktuell, BUG)**:
- HTTP 202 (Upload erfolgreich)
- Dokument erscheint **NICHT** in Explorer (Membership fehlt)

**Erwartung (nach Fix #1)**:
- HTTP 400 (Bad Request: `collection_not_found`)

#### Test 2: `latest_only=True` mit mehreren Versionen

**Setup**:
1. Upload desselben Dokuments zweimal (mit unterschiedlichem Content → neue Version)
2. Explorer mit `latest=1`

**Erwartung (aktuell, BUG)**:
- **Beide Versionen** erscheinen in Liste (Duplikat)

**Erwartung (nach Fix #2)**:
- **Nur neueste Version** erscheint

---

## 12. Zusammenfassung & Nächste Schritte

### 12.1 Kritische Findings

| # | Schwere      | Beschreibung                                                  | Status        |
|---|--------------|---------------------------------------------------------------|---------------|
| 1 | KRITISCH     | Collection Membership Silent Failure                          | ❌ Offen      |
| 2 | KRITISCH     | `list_latest_by_collection` liefert Duplikate (DB)            | ❌ Offen      |
| 3 | KRITISCH     | Asset-API fehlt in ObjectStore/DB                             | ℹ️ Kein Bedarf (noch) |
| 4 | HOCH         | ObjectStore: Lifecycle nicht persistiert                      | ℹ️ Nicht in Prod (DB aktiv) |
| 5 | HOCH         | ObjectStore: Workflow-ID Divergenz                            | ℹ️ Nicht in Prod |

### 12.2 Production Safety Assessment

**✅ SICHER (aktuell)**:
- Default-Config nutzt `DbDocumentsRepository` → Workbench läuft auf DB
- Upload → Ingestion → Explorer nutzt konsistent DB-Repo
- ObjectStore/InMemory nur in Tests/Dev-Override (muss validiert werden)

**❌ RISIKEN (sofort beheben)**:
- **Finding #1**: Collection Membership kann silent fehlschlagen → Dokumente unsichtbar
- **Finding #2**: `list_latest` liefert Duplikate → Explorer UI verwirrend

**⚠️ EMPFEHLUNGEN**:
1. **Sofort**: Fix #1 & #2 umsetzen (siehe 10.1)
2. **Kurzfristig**: `.env` validieren (kein ObjectStore-Override), Test-Plan durchführen (siehe 11.2)
3. **Mittelfristig**: Deprecate ObjectStore (oder Feature-Parity herstellen), Asset-API für DB implementieren (falls benötigt)
4. **Langfristig**: InMemory nur für Unit-Tests nutzen (Integration-Tests gegen DB)

### 12.3 Dokumentation & Architektur-Entscheidung

**Frage für Team**: Sollen **drei** Repos weiterhin unterstützt werden?

**Option A (Empfehlung)**: **Deprecate ObjectStore + InMemory für Prod**
- ✅ Fokus auf DB-Repo (produktionsreif)
- ✅ InMemory nur für Unit-Tests
- ✅ ObjectStore entfernen oder „Experimental"-Label

**Option B**: **Feature-Parity herstellen**
- ⚠️ Asset-API in DB/ObjectStore implementieren
- ⚠️ ObjectStore: Lifecycle, deterministische Pagination
- ⚠️ Workflow-ID-Normalisierung vereinheitlichen
- **AUFWAND**: Hoch (6-8 Findings pro Repo)

**Option C**: **Explizite Interface-Segregation**
- Define `CoreDocumentsRepository` (nur `upsert`, `get`, `list_*`)
- Define `ExtendedDocumentsRepository` (+ Assets, Lifecycle)
- InMemory/DB implementieren Extended, ObjectStore nur Core
- **VORTEIL**: Klare Erwartungen, keine NotImplementedError-Überraschungen

---

## Anhang: ENV-Variablen Checkliste

### Kritische Settings (für Prod-Deployment)

```bash
# Repository-Klasse (DEFAULT = DB, explizit setzen nur bei Dev-Override)
DOCUMENTS_REPOSITORY_CLASS=ai_core.adapters.db_documents_repository.DbDocumentsRepository

# Storage-Backend (für Blobs)
# (Standard: ObjectStoreStorage → lokales FS, für Prod evtl. S3)
# Keine ENV-Variable nötig (hard-coded in __init__)

# Lifecycle-Store (DEFAULT = PersistentDocumentLifecycleStore)
DOCUMENT_LIFECYCLE_STORE_CLASS=documents.repository.PersistentDocumentLifecycleStore
```

### Validierungs-Script (Shell)

```bash
#!/bin/bash
set -e

echo "=== Repository Config Validation ==="

# 1. Check ENV
REPO_CLASS="${DOCUMENTS_REPOSITORY_CLASS:-ai_core.adapters.db_documents_repository.DbDocumentsRepository}"
echo "DOCUMENTS_REPOSITORY_CLASS: $REPO_CLASS"

if [[ "$REPO_CLASS" == *"InMemoryDocumentsRepository"* ]]; then
  echo "❌ WARNING: InMemory repository in ENV (data will be lost on restart!)"
  exit 1
fi

if [[ "$REPO_CLASS" == *"ObjectStoreDocumentsRepository"* ]]; then
  echo "⚠️ WARNING: ObjectStore repository in ENV (missing Asset/Lifecycle features!)"
  exit 1
fi

# 2. Check active repo in Django
python manage.py shell -c "
from ai_core.services import _get_documents_repository
repo = _get_documents_repository()
expected = 'DbDocumentsRepository'
actual = type(repo).__name__
if actual != expected:
    print(f'❌ FAIL: Expected {expected}, got {actual}')
    exit(1)
else:
    print(f'✅ OK: {actual}')
"

echo "=== Validation Complete ==="
```

---

**Ende des Berichts**

**Nächste Schritte**:
1. Review mit Team (Priorität auf Finding #1 & #2)
2. `.env` validieren (kein ObjectStore-Override)
3. Fixes #1 & #2 implementieren
4. Test-Plan durchführen (siehe Kapitel 11)
5. Architektur-Entscheidung: Option A/B/C für langfristige Repo-Strategie
