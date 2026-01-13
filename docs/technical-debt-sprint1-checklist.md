# Sprint 1: Quick Wins - Detaillierte Checkliste

**Ziel**: Entfernung von sicher zu löschendem Legacy-Code
**Geschätzter Aufwand**: ~3 Stunden
**Risiko**: NIEDRIG
**Pre-Requisites**: Feature Branch `tech-debt-cleanup` erstellt

---

## ✅ Pre-Flight Checks

```bash
# 1. Backup & Branch
git checkout main
git pull origin main
git checkout -b tech-debt-cleanup

# 2. Baseline Tests
npm run test:py:parallel
npm run lint

# 3. Status dokumentieren
git status > /tmp/pre-cleanup-status.txt
```

**Erwartung**: Alle Tests grün, keine Lint-Fehler

---

## 🎯 Task 1: RAG Demo Endpoint entfernen (30 Min)

### 1.1 View Class entfernen

**Datei**: `ai_core/views.py`

```bash
# Zeilen finden
sed -n '2200,2243p' ai_core/views.py
```

**Zu löschen**: Lines 2200-2243
- Gesamte `RagDemoViewV1` Klasse inkl. Docstring
- Methode `post()` mit `@default_extend_schema`

### 1.2 Response Serializer entfernen

**Datei**: `ai_core/views.py`

```bash
# Zeilen finden
sed -n '1391,1409p' ai_core/views.py
```

**Zu löschen**: Lines 1391-1409
- `RAG_DEMO_DEPRECATED_RESPONSE` Serializer
- Nested Serializer: `RagDemoDeprecatedErrorDetail`
- Nested Serializer: `RagDemoDeprecatedErrorMeta`

### 1.3 View Aliases entfernen

**Datei**: `ai_core/views.py`

```bash
# Zeilen finden
sed -n '2254,2255p' ai_core/views.py
```

**Zu löschen**: Lines 2254-2255
```python
rag_demo_v1 = RagDemoViewV1.as_view()
rag_demo = rag_demo_v1
```

### 1.4 URL Route entfernen

**Datei**: `ai_core/urls.py`

```bash
# Zeile finden
sed -n '26p' ai_core/urls.py
```

**Zu löschen**: Line 26
```python
path("v1/rag-demo/", views.rag_demo, name="rag_demo"),
```

### 1.5 Test-Datei entfernen

```bash
# Prüfen ob existiert
ls -la tests/demo/test_rag_demo_removed.py

# Entfernen
git rm tests/demo/test_rag_demo_removed.py
```

### 1.6 Verifikation

```bash
# 1. Imports prüfen
rg "RagDemoViewV1|RAG_DEMO_DEPRECATED|rag_demo" --type py

# 2. Tests laufen
npm run test:py:fast

# 3. OpenAPI Schema generieren
npm run api:schema

# 4. Commit
git add -A
git commit -m "refactor: Remove deprecated RAG demo endpoint (HTTP 410)

- Remove RagDemoViewV1 class and response serializers
- Remove /v1/rag-demo/ URL route
- Remove associated tests
- Update OpenAPI schema

BREAKING: Endpoint /v1/rag-demo/ no longer exists (was returning HTTP 410)

Refs: #tech-debt-cleanup"
```

**Erwartung**: Keine Treffer mehr, Tests grün

---

## 🎯 Task 2: Legacy Ping/Intake Endpoints entfernen (1 Std)

### 2.1 Verwendung prüfen

```bash
# Sicherstellen, dass keine Tests abhängen
rg "LegacyPingView|LegacyIntakeView|ping_legacy|intake_legacy" --type py

# Erwartung: Nur Treffer in ai_core/views.py
```

### 2.2 LegacyPingView entfernen

**Datei**: `ai_core/views.py`

**Zu löschen**: Lines 1435-1443
```python
class LegacyPingView(_PingBase):
    """Legacy heartbeat endpoint served under the unversioned prefix."""

    api_deprecated = True
    api_deprecation_id = LEGACY_DEPRECATION_ID

    @default_extend_schema(**_legacy_schema_kwargs(PING_SCHEMA))
    def get(self, request: Request) -> Response:
        return super().get(request)
```

### 2.3 LegacyIntakeView entfernen

**Datei**: `ai_core/views.py`

**Zu löschen**: Lines 1488-1497
```python
class LegacyIntakeView(_GraphView):
    """Deprecated intake endpoint retained for backwards compatibility."""

    api_deprecated = True
    api_deprecation_id = LEGACY_DEPRECATION_ID
    graph_name = "info_intake"

    @default_extend_schema(**_legacy_schema_kwargs(INTAKE_SCHEMA))
    def post(self, request: Request) -> Response:
        return super().post(request)
```

### 2.4 Aliase entfernen

**Datei**: `ai_core/views.py`

**Zu löschen**: Lines 2247-2252
```python
ping_legacy = LegacyPingView.as_view()
ping = ping_legacy

intake_legacy = LegacyIntakeView.as_view()
intake = intake_legacy
```

### 2.5 Helper-Funktion entfernen

**Datei**: `ai_core/views.py`

**Zu löschen**: Lines 786-789
```python
def _legacy_schema_kwargs(base_kwargs: dict[str, object]) -> dict[str, object]:
    legacy_kwargs = dict(base_kwargs)
    legacy_kwargs["deprecated"] = True
    return legacy_kwargs
```

### 2.6 Konstante entfernen

**Datei**: `ai_core/views.py`

**Zu löschen**: Line 783
```python
LEGACY_DEPRECATION_ID = "ai-core-legacy"
```

### 2.7 Verifikation

```bash
# 1. Keine Verwendung mehr
rg "LegacyPingView|LegacyIntakeView|ping_legacy|intake_legacy|LEGACY_DEPRECATION_ID|_legacy_schema_kwargs" --type py

# 2. Tests
npm run test:py:fast

# 3. API Schema
npm run api:schema

# 4. Commit
git add -A
git commit -m "refactor: Remove legacy unversioned ping/intake endpoints

- Remove LegacyPingView and LegacyIntakeView classes
- Remove _legacy_schema_kwargs helper
- Remove LEGACY_DEPRECATION_ID constant
- Remove view aliases (ping_legacy, intake_legacy)
- Update OpenAPI schema

BREAKING: Unversioned endpoints /ping and /intake no longer available
Migration: Use /v1/ping and /v1/intake instead

Refs: #tech-debt-cleanup"
```

**Erwartung**: Keine Treffer, Tests grün

---

## 🎯 Task 3: SemanticChunker entfernen (1 Std)

### 3.1 Verwendung analysieren

```bash
# Alle Verwendungen finden
rg "SemanticChunker" --type py -C 3

# Erwartete Treffer:
# - ai_core/rag/semantic_chunker.py (Implementierung)
# - ai_core/tasks/ingestion_tasks.py (mit DeprecationWarning)
# - ai_core/rag/chunking/__init__.py (Export)
```

### 3.2 Implementierung entfernen

```bash
# Datei löschen
git rm ai_core/rag/semantic_chunker.py
```

### 3.3 Warning in ingestion_tasks.py entfernen

**Datei**: `ai_core/tasks/ingestion_tasks.py`

**Zu ändern**: Lines 581-595

**Vorher**:
```python
if structured_blocks:
    warnings.warn(
        "SemanticChunker is deprecated and will be removed in a future version. "
        "Use HybridChunker from ai_core.rag.chunking instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    semantic_chunker = SemanticChunker(
        embedding_model=embedding_model,
        target_token_size=target_token_size,
        # ... weitere Parameter
    )
    # ... Chunking-Logik
```

**Nachher**:
```python
if structured_blocks:
    # SemanticChunker wurde entfernt - verwende HybridChunker
    # Falls benötigt, hier HybridChunker Integration hinzufügen
    # Siehe: ai_core/rag/chunking/hybrid_chunker.py
    pass  # TODO: Implementierung wenn benötigt
```

**ODER** (falls SemanticChunker komplett durch HybridChunker ersetzt wurde):
```python
# Gesamten if structured_blocks Block entfernen, wenn nicht mehr benötigt
```

### 3.4 Export aus chunking/__init__.py entfernen

**Datei**: `ai_core/rag/chunking/__init__.py`

**Zu entfernen**:
```python
from ai_core.rag.semantic_chunker import SemanticChunker  # Falls vorhanden

# Oder aus __all__ Liste:
__all__ = [
    # ... andere Exports
    # "SemanticChunker",  # Diese Zeile entfernen
]
```

### 3.5 Verifikation

```bash
# 1. Keine Verwendung mehr
rg "SemanticChunker" --type py

# 2. Keine Imports
rg "from ai_core.rag.semantic_chunker import|from ai_core.rag import.*SemanticChunker" --type py

# 3. Tests (sollten weiterhin grün sein)
npm run test:py:fast

# 4. Unit-Tests für Chunking
npm run test:py:single -- ai_core/tests/rag/chunking/

# 5. Commit
git add -A
git commit -m "refactor: Remove deprecated SemanticChunker

- Remove ai_core/rag/semantic_chunker.py implementation
- Remove DeprecationWarning from ingestion_tasks.py
- Remove export from chunking/__init__.py
- HybridChunker is the recommended replacement

BREAKING: SemanticChunker no longer available
Migration: Use HybridChunker from ai_core.rag.chunking instead

Refs: #tech-debt-cleanup"
```

**Erwartung**: Keine Treffer mehr (außer in Docs/Changelog), Tests grün

---

## 🎯 Task 4: Collection Bootstrap Test cleanup (15 Min)

### 4.1 Test-Datei editieren

**Datei**: `tests/documents/test_collection_bootstrap.py`

**Zu ändern**: Line 96

**Vorher**:
```python
{
    "tenant_id": str(tenant.id),
    "collection_key": "manual-search",  # NEW: Use key
    "collection_id": collection_id,     # OLD: Also pass ID for compat
    "hash": "test-hash-123",
    "content_hash": "test-hash-123",
    # ...
}
```

**Nachher**:
```python
{
    "tenant_id": str(tenant.id),
    "collection_key": "manual-search",
    "hash": "test-hash-123",
    "content_hash": "test-hash-123",
    # ...
}
```

### 4.2 Verifikation

```bash
# 1. Test läuft
npm run test:py:single -- tests/documents/test_collection_bootstrap.py -v

# 2. Keine anderen Verwendungen von collection_id für compat
rg "collection_id.*# OLD|# OLD.*collection_id" --type py

# 3. Commit
git add tests/documents/test_collection_bootstrap.py
git commit -m "test: Remove legacy collection_id compatibility field

- Use collection_key exclusively
- Remove redundant collection_id parameter
- Test passes with collection_key only

Refs: #tech-debt-cleanup"
```

**Erwartung**: Test grün, keine anderen Treffer

---

## 🎯 Task 5: build_graph() Shim analysieren (30 Min)

### 5.1 Verwendung finden

```bash
# Alle Importe finden
rg "from ai_core.tasks import.*build_graph" --type py

# Direkte Verwendung finden
rg "ai_core\.tasks\.build_graph" --type py

# Erwartung: Möglicherweise in Tests verwendet
```

### 5.2 Entscheidung treffen

**Option A: Keine Verwendung gefunden**
→ Shim kann entfernt werden

**Option B: Verwendung gefunden in Tests**
→ Tests migrieren auf direkten Import

### 5.3a Wenn keine Verwendung: Entfernen

**Datei**: `ai_core/tasks/graph_tasks.py`

**Zu löschen**: Lines 62-65
```python
def build_graph(*, event_emitter: Optional[Any] = None):
    """Legacy shim so older tests can import ai_core.tasks.build_graph."""
    return _build_ingestion_graph(event_emitter)
```

```bash
# Commit
git add ai_core/tasks/graph_tasks.py
git commit -m "refactor: Remove build_graph() legacy shim

- No usages found in codebase
- Tests already migrated to direct imports

Refs: #tech-debt-cleanup"
```

### 5.3b Wenn Verwendung gefunden: Migrieren

Für jeden Test der `build_graph` importiert:

**Vorher**:
```python
from ai_core.tasks import build_graph
```

**Nachher**:
```python
from ai_core.graphs import _build_ingestion_graph as build_graph
# ODER direkt:
from ai_core.graphs import _build_ingestion_graph
```

**Dann**: Shim entfernen wie in 5.3a

### 5.4 Verifikation

```bash
# 1. Keine Verwendung mehr
rg "from ai_core.tasks import.*build_graph|ai_core\.tasks\.build_graph" --type py

# 2. Tests
npm run test:py:fast

# 3. Commit (falls Migration nötig war)
git add -A
git commit -m "refactor: Migrate tests from build_graph shim and remove it

- Migrate test imports to direct _build_ingestion_graph import
- Remove legacy build_graph() shim from graph_tasks.py
- All tests pass with direct imports

BREAKING: ai_core.tasks.build_graph no longer available
Migration: Import from ai_core.graphs instead

Refs: #tech-debt-cleanup"
```

**Erwartung**: Keine Treffer, Tests grün

---

## 🎯 Final Checks & Push

### 6.1 Gesamte Test-Suite

```bash
# Full test suite (inkl. slow tests)
npm run test:py:parallel

# Linting
npm run lint

# Coverage check
npm run test:py:cov
```

**Erwartung**: Alles grün, Coverage unverändert oder besser

### 6.2 OpenAPI Schema aktualisieren

```bash
# Schema neu generieren
npm run api:schema

# Prüfen ob deprecated endpoints entfernt wurden
cat api/openapi-schema.yml | grep -i "rag-demo\|legacy"
```

**Erwartung**: Keine deprecated Endpoints mehr im Schema

### 6.3 Git Status

```bash
# Übersicht aller Änderungen
git status

# Diff prüfen
git diff main...tech-debt-cleanup

# Alle Commits anzeigen
git log main..tech-debt-cleanup --oneline
```

**Erwartete Commits**:
1. `refactor: Remove deprecated RAG demo endpoint (HTTP 410)`
2. `refactor: Remove legacy unversioned ping/intake endpoints`
3. `refactor: Remove deprecated SemanticChunker`
4. `test: Remove legacy collection_id compatibility field`
5. `refactor: Remove build_graph() legacy shim` (falls entfernt)

### 6.4 Push & PR

```bash
# Push to remote
git push origin tech-debt-cleanup

# PR erstellen (via gh CLI)
gh pr create \
  --title "refactor: Sprint 1 - Remove deprecated legacy code" \
  --body "$(cat <<EOF
## Sprint 1: Technical Debt Cleanup - Quick Wins

Removes safely deletable legacy code from pre-MVP development phase.

### Changes

- ✅ Remove deprecated RAG demo endpoint (HTTP 410)
- ✅ Remove legacy unversioned ping/intake endpoints
- ✅ Remove deprecated SemanticChunker (use HybridChunker)
- ✅ Remove legacy collection_id compat field from tests
- ✅ Remove/migrate build_graph() shim

### Breaking Changes

- **Endpoints**: /v1/rag-demo/, /ping, /intake no longer available
- **API**: SemanticChunker removed (use HybridChunker)
- **Tests**: build_graph import from ai_core.tasks removed

### Testing

- ✅ All tests pass: \`npm run test:py:parallel\`
- ✅ Linting clean: \`npm run lint\`
- ✅ Coverage maintained
- ✅ OpenAPI schema updated

### References

- [Technical Debt Cleanup Plan](../docs/technical-debt-cleanup.md)
- [Sprint 1 Checklist](../docs/technical-debt-sprint1-checklist.md)

Refs: #tech-debt-cleanup
EOF
)" \
  --base main \
  --head tech-debt-cleanup
```

---

## 📊 Success Metrics

- [ ] Alle 5 Tasks abgeschlossen
- [ ] Alle Tests grün: `npm run test:py:parallel`
- [ ] Linting clean: `npm run lint`
- [ ] Coverage >= Baseline
- [ ] OpenAPI Schema aktualisiert
- [ ] 5 Clean Commits erstellt
- [ ] PR erstellt und ready for review
- [ ] Keine neuen Warnings/Errors

---

## 🚨 Troubleshooting

### Problem: Tests schlagen fehl nach Endpoint-Entfernung

**Diagnose**:
```bash
# Welche Tests scheitern?
npm run test:py:parallel 2>&1 | grep -A 5 "FAILED"
```

**Lösung**:
- Prüfe ob Tests die entfernten Endpoints aufrufen
- Migriere Tests auf neue Endpoints
- ODER: Entferne veraltete Tests

### Problem: Import-Fehler nach SemanticChunker Entfernung

**Diagnose**:
```bash
# Wo wird noch importiert?
rg "from.*SemanticChunker|import.*SemanticChunker" --type py
```

**Lösung**:
- Alle Importe auf `HybridChunker` umstellen
- Oder Import-Zeilen entfernen falls ungenutzt

### Problem: API Schema enthält noch deprecated Endpoints

**Diagnose**:
```bash
# Schema prüfen
cat api/openapi-schema.yml | grep -i "deprecated"
```

**Lösung**:
- `npm run api:schema` erneut ausführen
- Server neu starten falls nötig
- Schema-Cache leeren

---

## 📝 Post-Sprint Review

Nach Abschluss dokumentieren:

```markdown
## Sprint 1 Results

**Completed**: [Datum]
**Duration**: [X Stunden]
**Commits**: [Anzahl]

### Removed

- [ ] RAG Demo Endpoint (Lines: ~100)
- [ ] Legacy Ping/Intake (Lines: ~40)
- [ ] SemanticChunker (Lines: ~200)
- [ ] Test legacy fields (Lines: ~5)
- [ ] build_graph shim (Lines: ~5)

**Total lines removed**: ~350 Lines

### Metrics

- Test Coverage: Before [X%] → After [Y%]
- Deprecated Markers: Before [N] → After [M]
- API Endpoints: Before [X] → After [Y]

### Lessons Learned

[Was lief gut? Was kann verbessert werden?]

### Next Steps

→ Sprint 2: Breaking Changes (siehe technical-debt-cleanup.md)
```

---

**Version**: 1.0
**Created**: 2026-01-13
**Estimated Duration**: 3 hours
**Risk Level**: LOW
