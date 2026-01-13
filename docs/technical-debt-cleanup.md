# Technical Debt Cleanup - Pre-MVP

**Status**: Pre-MVP Phase - Breaking Changes erlaubt
**Erstellt**: 2026-01-13
**Ziel**: Technische Schulden aus Entwicklungsphase entfernen

## Executive Summary

Während der Codebase-Analyse wurden **5 Kategorien** technischer Schulden identifiziert:
- 🔴 **Deprecated API Endpoints** (4 Endpoints)
- 🟡 **Legacy Compatibility Fields** (3 Bereiche)
- 🟠 **Deprecated Code mit Warnings** (1 Komponente)
- 🔵 **Legacy Shims** (2 Funktionen)
- 🟣 **Test-spezifische Legacy** (2 Tests)

**Breaking Change Potenzial**: MITTEL
**Geschätzter Aufwand**: 2-3 Tage
**Risiko**: NIEDRIG (pre-MVP, keine Produktion)

---

## 🔴 Phase 1: API Endpoint Cleanup (HOHE PRIORITÄT)

### 1.1 RAG Demo Endpoint - SOFORT ENTFERNEN ✅

**Status**: Gibt bereits HTTP 410 (Gone) zurück
**Risiko**: KEINS
**Aufwand**: 30 Minuten

#### Zu entfernen:

1. **View Class**: `ai_core/views.py:2200-2243`
   ```python
   class RagDemoViewV1(_BaseAgentView):
       """Deprecated demo endpoint..."""
   ```

2. **Response Serializer**: `ai_core/views.py:1391-1409`
   ```python
   RAG_DEMO_DEPRECATED_RESPONSE = inline_serializer(...)
   ```

3. **View Aliases**: `ai_core/views.py:2254-2255`
   ```python
   rag_demo_v1 = RagDemoViewV1.as_view()
   rag_demo = rag_demo_v1
   ```

4. **URL Route**: `ai_core/urls.py:26`
   ```python
   path("v1/rag-demo/", views.rag_demo, name="rag_demo"),
   ```

5. **Tests**: `tests/demo/test_rag_demo_removed.py` (gesamte Datei)

#### Migrations:
- Keine DB-Migration nötig
- OpenAPI Schema wird automatisch aktualisiert

---

### 1.2 Legacy Ping/Intake Endpoints - ENTFERNEN ✅

**Status**: Keine Test-Abhängigkeiten gefunden
**Risiko**: NIEDRIG
**Aufwand**: 1 Stunde

#### Zu entfernen:

1. **LegacyPingView**: `ai_core/views.py:1435-1443`
2. **LegacyIntakeView**: `ai_core/views.py:1488-1497`
3. **Aliase**: `ai_core/views.py:2247-2252`
   ```python
   ping_legacy = LegacyPingView.as_view()
   ping = ping_legacy
   intake_legacy = LegacyIntakeView.as_view()
   intake = intake_legacy
   ```
4. **Helper**: `ai_core/views.py:786-789` - `_legacy_schema_kwargs()`
5. **Constant**: `ai_core/views.py:783` - `LEGACY_DEPRECATION_ID`

#### Auswirkungen:
- Unversionierte Endpoints `/ping` und `/intake` nicht mehr verfügbar
- Clients MÜSSEN auf `/v1/ping` und `/v1/intake` migrieren
- Pre-MVP: Keine Produktion-Clients vorhanden

---

### 1.3 Info Intake Endpoint - PRÜFEN ⚠️

**Status**: Deprecated, aber möglicherweise in Verwendung
**Risiko**: MITTEL
**Aufwand**: TBD nach Analyse

#### Zu prüfen:

```bash
# Suche nach Verwendung in Tests
npm run test:py:single -- -k "info_intake" -v

# Suche nach Verwendung in Code
rg "IntakeViewV1|graph_name.*info_intake" --type py
```

#### Entscheidung:
- ✅ **ENTFERNEN** wenn keine aktive Verwendung
- ⏸️ **BEHALTEN** wenn noch benötigt → Migration planen

---

## 🟡 Phase 2: Legacy Compatibility Fields (MITTLERE PRIORITÄT)

### 2.1 formatted_status in UniversalIngestionGraph

**Datei**: `ai_core/graphs/technical/universal_ingestion_graph.py:66-67, 428`
**Risiko**: MITTEL (Breaking Change für Konsumenten)
**Aufwand**: 1 Stunde

#### Aktueller Code:

```python
class UniversalIngestionOutput(TypedDict):
    decision: Literal["processed", "skipped", "failed"]
    # ... andere Felder ...
    formatted_status: str | None  # Legacy compatibility fields (can be deprecated later)
```

#### Analyse:

**Konsumenten**:
- `ai_core/tasks/graph_tasks.py` - Celery Task
- `tests/chaos/test_graph_io_contracts.py` - Contract Tests

**Nutzung**:
```python
formatted_status=decision.upper()  # Redundant zu decision
```

#### Empfehlung: ENTFERNEN ✅

**Begründung**:
1. Redundant: `formatted_status` = `decision.upper()`
2. Konsumenten können `.upper()` selbst machen
3. Explizit als "can be deprecated" markiert

**Migrationsschritte**:
1. Feld aus `UniversalIngestionOutput` entfernen
2. Zeile 428 entfernen: `formatted_status=decision.upper()`
3. Test `test_graph_io_contracts.py` aktualisieren
4. Celery Task auf `decision` umstellen

---

### 2.2 Legacy Scope Flattening in Theme Views

**Datei**: `theme/views_rag_tools.py:246-253`
**Risiko**: HOCH (Breaking Change für Worker)
**Aufwand**: 4-6 Stunden

#### Aktueller Code:

```python
# Flatten context for legacy worker (Scope + Business)
legacy_scope = tool_context.scope.model_dump(mode="json", exclude_none=True)
legacy_scope.update(tool_context.business.model_dump(mode="json", exclude_none=True))

result_payload, completed = views.submit_worker_task(
    task_payload=task_payload,
    scope=legacy_scope,  # Flattened dict
    ...
)
```

#### Analyse:

**Problem**: Worker erwarten flattened dict statt strukturiertes `ToolContext`

**Betroffene Komponenten**:
- `theme/views_rag_tools.py` (RAG-Tools View)
- `theme/views_web_search.py` (Web Search View)
- `ai_core/tasks/graph_tasks.py` (Worker Tasks)

#### Empfehlung: MIGRATION ⚠️

**Option A: Worker auf ToolContext migrieren** (EMPFOHLEN)
1. Worker akzeptieren strukturiertes `ToolContext`
2. `scope` Parameter wird zu `tool_context`
3. Views senden strukturiertes Objekt statt flat dict

**Option B: Explicit Legacy Flag**
1. `scope` → `legacy_scope` umbenennen
2. Neue `tool_context` Parameter hinzufügen
3. Schrittweise Migration über beide Interfaces

**Geschätzter Aufwand**:
- Option A: 4-6 Stunden (Clean Break)
- Option B: 8-10 Stunden (Dual Interface)

**Empfehlung**: Option A (Pre-MVP erlaubt Breaking Changes)

---

### 2.3 Legacy UI Structure in Web Search

**Datei**: `theme/views_web_search.py:324`
**Risiko**: NIEDRIG
**Aufwand**: 30 Minuten

#### Aktueller Code:

```python
# Legacy UI expects "search.results" structure
search_results = output.get("search_results") or []
```

#### Zu prüfen:

```bash
# Suche Frontend-Code nach "search.results"
rg "search\.results" theme/static_src/
```

#### Empfehlung: DOKUMENTATION 🔵

1. Prüfen ob Frontend bereits migriert ist
2. Wenn ja: Kommentar entfernen
3. Wenn nein: Frontend migrieren, dann Kommentar entfernen

---

## 🟠 Phase 3: Deprecated Code (MITTLERE PRIORITÄT)

### 3.1 SemanticChunker - ENTFERNEN ✅

**Status**: Deprecated mit Warning, Nachfolger vorhanden
**Risiko**: NIEDRIG
**Aufwand**: 1 Stunde

#### Zu entfernen:

1. **Implementierung**: `ai_core/rag/semantic_chunker.py` (gesamte Datei)
2. **Warning**: `ai_core/tasks/ingestion_tasks.py:583-587`
3. **Import**: `ai_core/rag/chunking/__init__.py` - SemanticChunker Export

#### Nachfolger:

```python
from ai_core.rag.chunking import HybridChunker  # ✅ Verwenden
```

#### Migrationsschritte:

1. Suche nach allen Verwendungen:
   ```bash
   rg "SemanticChunker" --type py
   ```
2. Ersetze durch `HybridChunker`
3. Entferne Dateien
4. Tests aktualisieren

---

## 🔵 Phase 4: Legacy Shims (NIEDRIGE PRIORITÄT)

### 4.1 build_graph() Shim

**Datei**: `ai_core/tasks/graph_tasks.py:62-65`
**Risiko**: MITTEL (Test-Abhängigkeiten)
**Aufwand**: 30 Minuten

#### Aktueller Code:

```python
def build_graph(*, event_emitter: Optional[Any] = None):
    """Legacy shim so older tests can import ai_core.tasks.build_graph."""
    return _build_ingestion_graph(event_emitter)
```

#### Zu prüfen:

```bash
# Suche nach Verwendung
rg "from ai_core.tasks import build_graph" --type py
rg "ai_core\.tasks\.build_graph" --type py
```

#### Empfehlung: PRÜFEN → ENTFERNEN ⚠️

1. Finde alle Importierenden
2. Migriere auf direkten Import: `from ai_core.graphs import _build_ingestion_graph`
3. Entferne Shim

---

### 4.2 Backward Compatibility Export

**Datei**: `ai_core/tasks/__init__.py:4`
**Risiko**: KEINS
**Aufwand**: 0

#### Empfehlung: BEHALTEN 🔵

**Begründung**: Saubere Public API, kein Cleanup nötig.

---

## 🟣 Phase 5: Test-spezifische Legacy (NIEDRIGE PRIORITÄT)

### 5.1 Collection Bootstrap Test

**Datei**: `tests/documents/test_collection_bootstrap.py:96`
**Risiko**: KEINS
**Aufwand**: 15 Minuten

#### Aktueller Code:

```python
{
    "collection_key": "manual-search",  # NEW: Use key
    "collection_id": collection_id,     # OLD: Also pass ID for compat
    ...
}
```

#### Empfehlung: CLEANUP ✅

1. Entferne `collection_id` Zeile
2. Behalte nur `collection_key`
3. Verifiziere Test läuft

---

### 5.2 RAG Tools Support Frontend Tests

**Datei**: `theme/static_src/scripts/rag-tools-support.test.ts`
**Risiko**: NIEDRIG
**Aufwand**: 30 Minuten

#### Tests für:

```typescript
resolveEffectiveCollectionId({
  collectionInput: ' support-faq ',
  legacyDocClass: 'ignored',    // Legacy Fallback
  allowLegacy: true,
})
```

#### Zu prüfen:

1. Wird `legacyDocClass` noch verwendet?
2. Kann `allowLegacy` Flag entfernt werden?

#### Empfehlung: PRÜFEN → ENTFERNEN ⚠️

---

## 📊 Priorisierung & Roadmap

### Sprint 1: Quick Wins (1 Tag)

- ✅ **RAG Demo Endpoint entfernen** (30 Min)
- ✅ **Legacy Ping/Intake Endpoints entfernen** (1 Std)
- ✅ **SemanticChunker entfernen** (1 Std)
- ✅ **Collection Bootstrap Test cleanup** (15 Min)
- ⚠️ **build_graph() Shim prüfen** (30 Min)

**Gesamt**: ~3 Stunden
**Risiko**: NIEDRIG

---

### Sprint 2: Breaking Changes (1-2 Tage)

- ⚠️ **formatted_status entfernen** (1 Std)
- ⚠️ **Legacy Scope Flattening migrieren** (4-6 Std)
- ⚠️ **Info Intake Endpoint prüfen** (TBD)
- ⚠️ **Frontend Legacy Tests prüfen** (30 Min)

**Gesamt**: 6-8 Stunden
**Risiko**: MITTEL

---

### Sprint 3: Dokumentation (0.5 Tag)

- 🔵 **Legacy UI Structure dokumentieren** (30 Min)
- 🔵 **Migration-Guide schreiben** (1 Std)
- 🔵 **Changelog aktualisieren** (30 Min)

**Gesamt**: 2 Stunden
**Risiko**: KEINS

---

## ✅ Checkliste vor Start

- [ ] Backup von `main` Branch erstellen
- [ ] Feature Branch `tech-debt-cleanup` erstellen
- [ ] Alle Tests laufen erfolgreich: `npm run test:py:parallel`
- [ ] Team informiert über Breaking Changes
- [ ] Migration-Guide vorbereitet

---

## 🔍 Detaillierte Verwendungsanalyse

### Search Checks (rg)

- RAG Demo Endpoint: `ai_core/views.py:1391`, `ai_core/views.py:2200`, `ai_core/views.py:2254`
- Legacy Ping/Intake + helpers: `ai_core/views.py:783`, `ai_core/views.py:786`, `ai_core/views.py:1435`, `ai_core/views.py:1488`, `ai_core/views.py:2247`
- Info Intake Endpoint: `ai_core/views.py:1476`, `ai_core/views.py:1481`, `ai_core/views.py:1493`, `ai_core/views.py:2250`
- formatted_status legacy field: `ai_core/graphs/technical/universal_ingestion_graph.py:67`, `ai_core/graphs/technical/universal_ingestion_graph.py:109`, `ai_core/graphs/technical/universal_ingestion_graph.py:428`
- Legacy scope flattening: `theme/views_rag_tools.py:246`
- Legacy UI structure note: `theme/views_web_search.py:324`
- SemanticChunker usage: `ai_core/tasks/ingestion_tasks.py:51`, `ai_core/tasks/ingestion_tasks.py:583`, `ai_core/rag/chunking/__init__.py:38`, `ai_core/rag/semantic_chunker.py:34`
- build_graph shim: `ai_core/tasks/graph_tasks.py:63`
- Collection bootstrap legacy field: `tests/documents/test_collection_bootstrap.py:96`
- Frontend legacy tests: `theme/static_src/scripts/rag-tools-support.test.ts:22`

### Expanded Search Checks (consumers)

- formatted_status consumers: `tests/chaos/test_graph_io_contracts.py:95`, `tests/chaos/test_graph_io_contracts.py:146`, `tests/chaos/test_graph_io_contracts.py:159`, `tests/chaos/test_graph_io_contracts.py:183`, `tests/chaos/test_graph_io_contracts.py:196`
- rag_demo references: `ai_core/urls_v1.py:14`, `ai_core/urls.py:26`, `tests/demo/test_rag_demo_removed.py:1`, `noesis2/settings/base.py:67`, `ai_core/management/commands/apply_rag_schema.py:18`
- info_intake references: `ai_core/graphs/technical/info_intake.py:34`, `ai_core/services/graph_support.py:172`, `ai_core/graph/bootstrap.py:21`, `ai_core/tests/test_graphs.py:16`, `ai_core/tests/test_meta_normalization.py:43`, `ai_core/tests/test_graph_registry.py:16`
- build_graph surface area: `ai_core/tasks/graph_tasks.py:62`, `ai_core/tasks/__init__.py:21`, `llm_worker/graphs/hybrid_search_and_score.py:1539`, `ai_core/graphs/technical/retrieval_augmented_generation.py:1106`

### Befehle für Due Diligence:

```bash
# 1. Suche nach deprecated Endpoints
rg "LegacyPingView|LegacyIntakeView|RagDemoViewV1" --type py

# 2. Suche nach formatted_status
rg "formatted_status" --type py

# 3. Suche nach SemanticChunker
rg "SemanticChunker" --type py

# 4. Suche nach build_graph Shim
rg "from ai_core.tasks import build_graph|ai_core\.tasks\.build_graph" --type py

# 5. Suche nach legacy_scope
rg "legacy_scope|Flatten context for legacy" --type py

# 6. Frontend Legacy Checks
rg "legacyDocClass|allowLegacy" theme/static_src/

# 7. Test für RAG Demo
npm run test:py:single -- tests/demo/test_rag_demo_removed.py
```

---

## 📝 Migration-Guide Template

Für jeden Breaking Change:

```markdown
### [Component Name] Migration

**Breaking**: [Was ändert sich]
**Reason**: [Warum entfernen wir es]
**Migration**:

Before:
```python
# Alter Code
```

After:
```python
# Neuer Code
```

**Testing**: [Wie verifizieren]
```

---

## 🚨 Rollback-Plan

Falls Probleme auftreten:

1. **Schneller Rollback**: `git revert <commit-hash>`
2. **Tests prüfen**: `npm run test:py:parallel`
3. **Selektiver Rollback**: Einzelne Änderungen zurücknehmen
4. **Hotfix Branch**: Bei Produktions-Issues

---

## 📌 Entscheidungs-Matrix

| Item | Priorität | Risiko | Aufwand | Empfehlung | Status |
|------|-----------|--------|---------|------------|--------|
| RAG Demo Endpoint | 🔴 Hoch | Keins | 30 Min | ✅ ENTFERNEN | Pending |
| Legacy Ping/Intake | 🔴 Hoch | Niedrig | 1 Std | ✅ ENTFERNEN | Pending |
| Info Intake | 🔴 Hoch | Mittel | TBD | ⚠️ PRÜFEN | Pending |
| formatted_status | 🟡 Mittel | Mittel | 1 Std | ✅ ENTFERNEN | Pending |
| Legacy Scope Flatten | 🟡 Mittel | Hoch | 4-6 Std | ⚠️ MIGRIEREN | Pending |
| Legacy UI Structure | 🟡 Mittel | Niedrig | 30 Min | 🔵 DOKUMENTIEREN | Pending |
| SemanticChunker | 🟠 Mittel | Niedrig | 1 Std | ✅ ENTFERNEN | Pending |
| build_graph Shim | 🔵 Niedrig | Mittel | 30 Min | ⚠️ PRÜFEN | Pending |
| Collection Test | 🟣 Niedrig | Keins | 15 Min | ✅ CLEANUP | Pending |
| Frontend Legacy Tests | 🟣 Niedrig | Niedrig | 30 Min | ⚠️ PRÜFEN | Pending |

**Legende**:
- ✅ **ENTFERNEN**: Sicher zu entfernen
- ⚠️ **PRÜFEN**: Analyse nötig vor Entscheidung
- 🔵 **BEHALTEN/DOC**: Dokumentieren oder behalten
- ⏸️ **VERSCHIEBEN**: Nach MVP

---

## 🎯 Success Criteria

- [ ] Alle deprecated API Endpoints entfernt
- [ ] Keine `@deprecated` Marker in neuem Code
- [ ] Alle Tests laufen: `npm run test:py:parallel`
- [ ] OpenAPI Schema aktualisiert
- [ ] Migration-Guide dokumentiert
- [ ] Code Coverage unverändert oder besser
- [ ] Keine Compiler/Linter Warnings

---

## 📚 Referenzen

- [AGENTS.md](../AGENTS.md) - Tool Contracts
- [CLAUDE.md](../CLAUDE.md) - Entwicklungs-Workflows
- [docs/architecture/overview.md](architecture/overview.md) - System-Architektur
- [docs/rag/overview.md](rag/overview.md) - RAG-System

---

**Version**: 1.0
**Author**: Claude (Code Analysis)
**Date**: 2026-01-13
**Review Status**: Pending Team Review
