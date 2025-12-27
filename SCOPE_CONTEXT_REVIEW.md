# ScopeContext Architecture Review

**Datum**: 2025-12-27
**Scope**: Redundanzen zwischen `ScopeContext`, `ToolContext` und Tool-Input-Modellen
**Ziel**: Bewertung der Praxistauglichkeit für "Vibe Coding"

---

## Executive Summary

Die aktuelle Architektur hat **3 Ebenen** von Context-Modellen mit **signifikanten Redundanzen**. Während die konzeptionelle Trennung sinnvoll ist, führen Duplikate zu:
- ❌ **Verwirrung** über die "Source of Truth"
- ❌ **Maintenance-Overhead** bei Änderungen
- ❌ **Inkonsistenzen** zwischen Modellen (bereits dokumentiert)
- ✅ **Aber**: Klare Separation of Concerns und Type Safety

**Empfehlung**: **Option B - Pragmatische Konsolidierung** (siehe unten)

---

## 1. Aktuelle Architektur

### 1.1 Die drei Context-Ebenen

| Modell | Datei | Zweck | Hauptfelder |
|--------|-------|-------|-------------|
| **ScopeContext** | `ai_core/contracts/scope.py` | Kanonischer HTTP/Graph-Kontext | `tenant_id`, `trace_id`, `invocation_id`, `case_id`, `workflow_id`, `run_id`, `ingestion_run_id`, `collection_id`, `user_id`, `service_id` |
| **ToolContext** | `ai_core/tool_contracts/base.py` | Tool-Invocation-Kontext | Alle ScopeContext-Felder + `now_iso`, `locale`, `timeouts_ms`, `budget_tokens`, `auth`, `metadata`, `document_id`, `document_version_id` |
| **Tool Inputs** | `ai_core/tools/*.py`, `ai_core/nodes/*.py` | Fachliche Tool-Parameter | Fachliche Felder + **teilweise** Context-IDs wie `collection_id`, `workflow_id` |

### 1.2 Conversion Pattern

```python
# ScopeContext → ToolContext
scope = ScopeContext(tenant_id="...", trace_id="...", ...)
tool_ctx = scope.to_tool_context(metadata={...}, budget_tokens=512)

# In Tools:
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    # Beide enthalten collection_id, workflow_id!
    collection_id = params.collection_id  # Aus Input?
    # ODER
    collection_id = context.collection_id  # Aus Context?
```

---

## 2. Redundanz-Matrix

### 2.1 Vollständige Feld-Übersicht

| Feld | ScopeContext | ToolContext | RetrieveInput | FrameworkAnalysisInput | WebSearchContext |
|------|--------------|-------------|---------------|------------------------|------------------|
| `tenant_id` | ✅ | ✅ | ❌ | ❌ | ✅ |
| `trace_id` | ✅ | ✅ | ❌ | ❌ | ✅ |
| `invocation_id` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `case_id` | ✅ | ✅ | ❌ | ❌ | ✅ |
| `user_id` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `service_id` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `run_id` | ✅ | ✅ | ❌ | ❌ | ✅ |
| `ingestion_run_id` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `workflow_id` | ✅ | ✅ | ✅ | ❌ | ✅ |
| `collection_id` | ✅ | ✅ | ✅ | ✅ (document_collection_id) | ❌ |
| `tenant_schema` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `idempotency_key` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `timestamp/now_iso` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `document_id` | ❌ | ✅ | ❌ | ✅ | ❌ |
| `document_version_id` | ❌ | ✅ | ❌ | ❌ | ❌ |
| `locale` | ❌ | ✅ | ❌ | ❌ | ❌ |
| `timeouts_ms` | ❌ | ✅ | ❌ | ❌ | ❌ |
| `budget_tokens` | ❌ | ✅ | ❌ | ❌ | ❌ |
| `visibility` | ❌ | ❌ | ✅ | ❌ | ❌ |
| `visibility_override_allowed` | ❌ | ✅ | ✅ | ❌ | ❌ |

### 2.2 Identifizierte Redundanzen

**Kritische Duplikate** (in ≥2 Modellen):
1. ✅ `collection_id`: ScopeContext, ToolContext, RetrieveInput, FrameworkAnalysisInput
2. ✅ `workflow_id`: ScopeContext, ToolContext, RetrieveInput, WebSearchContext
3. ✅ `visibility_override_allowed`: ToolContext, RetrieveInput

**Bekannte Inkonsistenzen** (aus `ID_SCOPE_TRACING_INVENTUR.md`):
- `collection_id`: `str | None` in ScopeContext vs. ToolContext (sollte UUID sein?)
- `document_id`, `document_version_id`: nur in ToolContext, fehlen in ScopeContext
- **Doku-Bug (BEHOBEN)**: AGENTS.md behauptete, `ToolContext` enforced `case_id`, aber Code-Realität: nur `normalize_meta` enforced es für Graph-Executions

---

## 3. Praxisbeispiel: retrieve.py

### 3.1 Aktuelle Verwendung

```python
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    # Line 574-583: Aus ToolContext
    tenant_id = str(context.tenant_id)
    tenant_schema = context.tenant_schema
    case_id = context.case_id

    # Line 586-590: Aus RetrieveInput (Duplikat!)
    collection_id = params.collection_id  # ⚠️ AUCH in context!
    workflow_id = params.workflow_id      # ⚠️ AUCH in context!

    # Line 604-606: Fallback-Logik (welche Source?)
    override_flag = params.visibility_override_allowed
    if override_flag is None:
        override_flag = context.visibility_override_allowed
```

### 3.2 Probleme in der Praxis

1. **Unklare Priorität**: Welches `collection_id` gilt - aus `params` oder `context`?
2. **Inkonsistente Handhabung**: Manchmal `params` zuerst, manchmal `context`
3. **Test-Overhead**: Müssen beide Quellen getestet werden
4. **Refactoring-Risk**: Änderungen an einer Stelle brechen andere

---

## 4. Bewertung für "Vibe Coding"

### 4.1 Was funktioniert gut? ✅

1. **Type Safety**: Pydantic validiert alle Ebenen
2. **Separation**: Fachliche Parameter (query, filters) klar getrennt von Runtime-Context
3. **Flexibilität**: `scope.to_tool_context()` ist einfach zu nutzen
4. **Dokumentiert**: AGENTS.md beschreibt Contracts klar

### 4.2 Was nervt beim Coding? ❌

1. **"Wo kommt das her?"**: Bei jedem Feld überlegen: Aus context oder params?
2. **Boilerplate**: Immer wieder dieselben Felder in mehreren Modellen
3. **Wartung**: Neue ID hinzufügen = 3 Dateien ändern + Tests
4. **Fehleranfällig**: Vergessene Propagierung → stille Bugs
5. **Inkonsistenzen**: Typ-Unterschiede zwischen Ebenen (UUID vs. str)

### 4.3 Vibe-Faktor: **6/10** 🤔

- Konzept ist solide, aber praktische Reibung
- Zu viel mentale Last beim Coding
- Nicht intuitiv, welche IDs wohin gehören

---

## 5. Lösungsoptionen

### Option A: Strikte Trennung (Clean, aber Breaking)

**Prinzip**: Jede Ebene hat **exklusive** Verantwortlichkeiten, keine Duplikate.

```python
# ScopeContext: Nur Correlation & Runtime IDs
class ScopeContext(BaseModel):
    tenant_id: str
    trace_id: str
    invocation_id: str
    run_id: str | None
    ingestion_run_id: str | None
    user_id: str | None
    service_id: str | None
    timestamp: datetime

# ToolContext: Runtime-Metadata (kein Business Context!)
class ToolContext(BaseModel):
    scope: ScopeContext  # Komposition statt Vererbung
    locale: str | None
    timeouts_ms: int | None
    budget_tokens: int | None
    auth: dict | None
    metadata: dict

# Tool-Inputs: Nur fachliche Parameter
class RetrieveInput(BaseModel):
    query: str
    filters: dict | None
    process: str | None
    doc_class: str | None
    # KEINE Context-IDs!

# Business-Context wird über SCOPE geführt:
scope = ScopeContext(
    tenant_id="...",
    case_id="...",       # ← Jetzt in ScopeContext!
    collection_id="...", # ← Jetzt in ScopeContext!
    workflow_id="...",   # ← Jetzt in ScopeContext!
)
tool_ctx = ToolContext(scope=scope, locale="de-DE")
```

**Vorteile**:
- ✅ Keine Redundanzen
- ✅ Klare Source of Truth
- ✅ Einfachere Tests
- ✅ Bessere Wartbarkeit

**Nachteile**:
- ❌ **BREAKING CHANGE**: Alle Tool-Signaturen ändern
- ❌ **Migration**: Alle bestehenden Tools anpassen
- ❌ **Aufwand**: ~50-100 Dateien betroffen

---

### Option B: Pragmatische Konsolidierung (Empfohlen!)

**Prinzip**: Tool-Inputs verlieren Context-IDs, alles über ToolContext.

```python
# ScopeContext: Bleibt wie es ist (kanonische IDs)
class ScopeContext(BaseModel):
    tenant_id: str
    trace_id: str
    invocation_id: str
    case_id: str | None
    workflow_id: str | None
    collection_id: str | None  # ← Business-Context bleibt hier
    document_id: str | None    # ← NEU (aus ToolContext hochziehen)
    document_version_id: str | None  # ← NEU
    run_id: str | None
    ingestion_run_id: str | None
    user_id: str | None
    service_id: str | None
    # ...

# ToolContext: Erweitert ScopeContext (wie aktuell)
class ToolContext(BaseModel):
    # Alle ScopeContext-Felder direkt (keine Komposition)
    tenant_id: str
    case_id: str | None
    collection_id: str | None
    # ... + Tool-spezifische Felder
    locale: str | None
    timeouts_ms: int | None
    # ...

# Tool-Inputs: NUR fachliche Parameter (ÄNDERUNG!)
class RetrieveInput(BaseModel):
    query: str
    filters: dict | None
    process: str | None
    doc_class: str | None
    visibility: str | None
    hybrid: dict | None
    top_k: int | None
    # ❌ RAUS: collection_id, workflow_id, visibility_override_allowed

# Tool-Funktion:
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    # Kein Duplikat mehr - alles aus context!
    collection_id = context.collection_id
    workflow_id = context.workflow_id
    visibility_override = context.visibility_override_allowed
```

**Vorteile**:
- ✅ **Kleinere Migration**: Nur Tool-Inputs anpassen
- ✅ **Klare Regel**: Context-IDs → ToolContext, Fachliches → Tool-Input
- ✅ **Weniger Verwirrung**: Keine Duplikate mehr
- ✅ **Rückwärtskompatibel**: ScopeContext bleibt stabil

**Nachteile**:
- ⚠️ Immer noch etwas Redundanz zwischen ScopeContext und ToolContext (aber akzeptabel)
- ⚠️ Tool-Inputs können keine Context-IDs mehr überschreiben (ist das gewollt?)

**Migration-Steps**:
1. `document_id`, `document_version_id` zu ScopeContext hinzufügen
2. `tool_context_from_scope()` updaten
3. Tool-Inputs refactoren:
   - RetrieveInput: `collection_id`, `workflow_id`, `visibility_override_allowed` entfernen
   - FrameworkAnalysisInput: `document_collection_id` umbenennen zu separatem Feld oder raus
4. Tool-Funktionen anpassen (aus `context` statt `params` lesen)
5. Tests updaten
6. Dokumentation in AGENTS.md updaten

---

### Option C: Status Quo mit Leitlinien

**Prinzip**: Nichts ändern, aber klare Regeln für neue Tools.

**Leitlinien**:
1. **Priorität**: ToolContext > Tool-Input bei Duplikaten
2. **Neue Tools**: Keine Context-IDs in Tool-Inputs
3. **Bestehende Tools**: Schrittweise migrieren
4. **Dokumentation**: AGENTS.md erweitern mit Prioritätsregeln

**Vorteile**:
- ✅ Kein Breaking Change
- ✅ Sofort umsetzbar

**Nachteile**:
- ❌ Problem bleibt bestehen
- ❌ Inkonsistenz zwischen alten und neuen Tools
- ❌ Vibe-Faktor bleibt bei 6/10

---

## 6. Empfehlung

### 🎯 **Option B: Pragmatische Konsolidierung**

**Warum?**
1. **Balance**: Guter Kompromiss zwischen Clean Architecture und Pragmatismus
2. **Migration**: Überschaubar (~10-15 Tool-Dateien)
3. **Klarheit**: Klare Regel für Entwickler: "Context-IDs nur über ToolContext"
4. **Vibe**: Vibe-Faktor steigt auf **8/10** 🎉

**Rollout-Plan**:
```
Phase 1 (Tag 1): ScopeContext erweitern
  - document_id, document_version_id hinzufügen
  - tool_context_from_scope() updaten
  - Tests schreiben

Phase 2 (Tag 2-3): Tool-Inputs refactoren
  - RetrieveInput, FrameworkAnalysisInput anpassen
  - Andere Tool-Inputs durchgehen
  - Tool-Funktionen anpassen

Phase 3 (Tag 4): Dokumentation & Tests
  - AGENTS.md updaten
  - Integration-Tests
  - E2E-Smoke-Tests

Phase 4 (Tag 5): Review & Merge
  - Code-Review
  - Migration-Guide für andere Entwickler
```

---

## 7. Next Steps

**Sofort**:
1. ✅ Diesen Review mit Team besprechen
2. ⬜ Entscheidung für Option A, B oder C
3. ⬜ Bei Option B: Migration-Branch erstellen

**Kurzfristig**:
1. ⬜ ScopeContext/ToolContext harmonisieren (Typ-Inkonsistenzen)
2. ⬜ Tool-Inputs refactoren
3. ⬜ Dokumentation updaten

**Langfristig**:
1. ⬜ Linting-Rule: "Keine Context-IDs in Tool-Inputs"
2. ⬜ Generator für neue Tools (mit korrektem Pattern)
3. ⬜ Architektur-ADR schreiben

---

## 8. Offene Fragen

1. **Überschreibbarkeit**: Sollen Tools jemals Context-IDs überschreiben dürfen? (Batch-Processing-Use-Case?)
2. **Rückwärtskompatibilität**: Müssen alte API-Clients weiter funktionieren?
3. **Performance**: Hat die Konsolidierung Auswirkungen auf Serialisierung?
4. **WebSearchContext**: Ist das ein separates Pattern oder sollte es auch ToolContext verwenden?

---

**Review erstellt von**: Claude Sonnet 4.5
**Nächster Review**: Nach Migration (Option B) oder in 3 Monaten (Option C)

---

## 9. Updates & Änderungen

### 2025-12-27: AGENTS.md Korrektur
- ✅ **Behoben**: Fehlerhafte Dokumentation in AGENTS.md:68
  - **Vorher**: "`case_id` ... Required for tool invocations (enforced in `ToolContext`)"
  - **Nachher**: "`case_id` ... Required for graph executions (enforced in `normalize_meta`). `ToolContext` itself does not enforce `case_id`"
- ✅ Ergänzt in AGENTS.md:114: Explizit klargestellt, dass `ToolContext` keine `case_id`-Validierung hat
- **Code-Realität**: Nur `normalize_meta` (ai_core/graph/schemas.py:166-167) enforced `case_id` für Graph-Executions
- **Impact**: Dokumentation jetzt konsistent mit tatsächlicher Implementierung
