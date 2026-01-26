# Technisches Audit: Tracing-Inkonsistenzen & Kwargs-Reduktion

**Datum:** 2026-01-23
**Scope:** Pre-MVP Refactoring-Vorbereitung
**Status:** Audit abgeschlossen (keine Code-Änderungen)

---

## Executive Summary

Die Codebase zeigt **konsistente Tracing-Key-Verwendung** (`trace_id` mit Unterstrich) ohne die befürchteten `trace.id`-Inkonsistenzen. Der Hauptbedarf besteht bei der **Kwargs-Reduktion**, wo 5 kritische Hotspots identifiziert wurden, die durch explizite Pydantic-Modelle ersetzt werden sollten.

---

## 1. Audit-Bereich: Tracing (trace.id vs. trace_id)

### 1.1 Status Quo

| Aspekt | Ergebnis |
|--------|----------|
| `trace.id` (mit Punkt) | **0 Vorkommen** |
| `trace_id` (mit Unterstrich) | **267 Dateien** |
| `TraceId` (Type Alias) | 1 Definition in `ai_core/contracts/scope.py:42` |
| `X-Trace-ID` (HTTP Header) | Konsistent via `common/constants.py:7` |

### 1.2 Tracing-Flow-Analyse

#### Generierung
| Datei | Zeile | Kontext |
|-------|-------|---------|
| [ai_core/middleware/context.py](../middleware/context.py#L177) | 177 | `coerce_trace_id(headers)` - primäre Generierung |
| [ai_core/middleware/context.py](../middleware/context.py#L208) | 208 | `uuid.uuid4().hex` - Fallback-Generierung |
| [ai_core/graph/schemas.py](../graph/schemas.py#L120-121) | 120-121 | `_coalesce()` + `get_contextvars()` |

#### Logger-Bindung (structlog.contextvars)
| Datei | Zeile | Bindungs-Key |
|-------|-------|--------------|
| [ai_core/middleware/context.py](../middleware/context.py#L84) | 84 | `bind_contextvars(**meta["log_context"])` → `trace_id` |
| [noesis2/celery.py](../../noesis2/celery.py#L108) | 108 | `bind_contextvars(trace_id=trace_id)` |
| [common/celery.py](../../common/celery.py#L184) | 184 | `bind_contextvars(**context)` → `trace_id` |
| [ai_core/graphs/technical/universal_ingestion_graph.py](../graphs/technical/universal_ingestion_graph.py#L183) | 183 | `bind_contextvars(**payload)` → `trace_id` |

#### Header-Injektion
| Datei | Zeile | Header-Name |
|-------|-------|-------------|
| [ai_core/infra/resp.py](../infra/resp.py#L91) | 91 | `X_TRACE_ID_HEADER: context.scope.trace_id` |
| [ai_core/llm/client.py](../llm/client.py#L357) | 357 | `X_TRACE_ID_HEADER: metadata.get("trace_id")` |
| [common/celery.py](../../common/celery.py#L992) | 992 | `request_headers[X_TRACE_ID_HEADER] = str(trace_id)` |

### 1.3 Risikobewertung Tracing

| Risiko | Bewertung | Begründung |
|--------|-----------|------------|
| Key-Inkonsistenz | **NIEDRIG** | Durchgängig `trace_id` verwendet |
| Tracing-Abbruch | **NIEDRIG** | Fallback-Generierung vorhanden |
| Header-Propagation | **MITTEL** | Celery-Tasks verlieren manchmal Context bei komplexen Chains |

**Bekanntes Issue:** In `noesis2/celery.py:89-90` werden alternative Header-Namen geprüft (`x-trace-id` lowercase), was Redundanz schafft.

---

## 2. Audit-Bereich: Kwargs-Eliminierung

### 2.1 Quantitative Analyse

| Kategorie | Anzahl Dateien | Beschreibung |
|-----------|----------------|--------------|
| `**kwargs` gesamt | 74 | Alle Dateien mit kwargs-Pattern |
| `def ...(**kwargs):` nur | 8 | Funktionen mit NUR kwargs als Parameter |
| `kwargs.get/pop/update` | 46 | Aktive kwargs-Manipulation |

### 2.2 Klassifizierung nach Pattern

#### A) Pass-through kwargs (Niedrige Priorität)
Diese werden nur weitergereicht ohne interne Verarbeitung:

| Datei | Zeile | Funktion | Verwendung |
|-------|-------|----------|------------|
| [conftest.py](../../conftest.py#L734) | 734 | `patch(self, target, new=None, **kwargs)` | Test-Mocking |
| [profiles/signals.py](../../profiles/signals.py#L10) | 10 | `create_user_profile(sender, instance, created, **kwargs)` | Django Signal |
| [ai_core/infra/observability.py](../infra/observability.py#L238-282) | 238-282 | `_run_async/*wrapped*` | Decorator-Delegation |

#### B) Dynamic Metadata (Hohe Priorität - Pydantic-Kandidaten)
Diese verwenden kwargs als untypisierte Dictionaries:

| Datei | Zeile | Funktion | Kritikalität |
|-------|-------|----------|--------------|
| [common/celery.py](../../common/celery.py#L267-289) | 267-289 | `_gather_context()` | **KRITISCH** |
| [common/celery.py](../../common/celery.py#L471-559) | 471-559 | `PIIScopedTask.__call__()` | **KRITISCH** |
| [noesis2/api/schema.py](../../noesis2/api/schema.py#L189-242) | 189-242 | `default_extend_schema()` | **HOCH** |
| [ai_core/rag/vector_store.py](../rag/vector_store.py#L1100-1109) | 1100-1109 | `hybrid_kwargs` Filterung | **HOCH** |
| [documents/logging_utils.py](../../documents/logging_utils.py#L106-125) | 106-125 | Log-Context-Wrapper | MITTEL |

### 2.3 Top 5 Kwargs-Hotspots (Pydantic-Refactoring-Kandidaten)

#### #1: `common/celery.py` - PIIScopedTask (Komplexität: 10/10)
```
Zeilen: 267-289, 307-359, 471-559
Pattern: kwargs.get("meta"), kwargs.get("tool_context"), scope_kwargs.pop()
Problem: Tiefe Verschachtelung, dynamische Field-Extraktion
Empfehlung: TaskScopeInput Pydantic-Modell
```

| Methode | LOC | kwargs-Operationen |
|---------|-----|-------------------|
| `_gather_context()` | 23 | 5x `.get()`, 2x `.update()` |
| `_from_meta()` | 53 | Field-by-field Extraktion |
| `__call__()` | 89 | 4x `.pop()`, 6x `.get()` |

#### #2: `noesis2/api/schema.py` - default_extend_schema (Komplexität: 7/10)
```
Zeilen: 189-242
Pattern: kwargs.pop() für Custom-Flags
Problem: 8 verschiedene kwargs.pop() Aufrufe
Empfehlung: ExtendSchemaOptions Pydantic-Modell
```

| Pop-Aufruf | Zeile | Default |
|------------|-------|---------|
| `include_tenant_headers` | 213 | `True` |
| `include_trace_header` | 214 | `False` |
| `include_error_responses` | 215 | `True` |
| `error_statuses` | 216-218 | `DEFAULT_ERROR_STATUSES` |
| `parameters` | 220 | `[]` |
| `responses` | 227 | `None` |
| `extensions` | 235 | `{}` |

#### #3: `ai_core/rag/vector_store.py` - hybrid_kwargs (Komplexität: 6/10)
```
Zeilen: 1090-1109
Pattern: Dynamische Keyword-Filterung via allowed_keywords Set
Problem: Implizite API-Kontrakte, fragile Keyword-Listen
Empfehlung: HybridSearchParams Pydantic-Modell
```

#### #4: `common/celery.py` - _resolve_priority_from_kwargs (Komplexität: 5/10)
```
Zeilen: 648-659
Pattern: Multi-level kwargs.get() mit meta-Fallback
Problem: Doppelte Lookup-Logik (kwargs + meta)
Empfehlung: TaskPriorityInput Pydantic-Modell
```

#### #5: `documents/logging_utils.py` - Log-Context-Wrapper (Komplexität: 4/10)
```
Zeilen: 106-125
Pattern: signature.bind_partial(*args, **kwargs)
Problem: Dynamische Parameter-Inspektion
Empfehlung: LogContextParams Pydantic-Modell
```

---

## 3. Dateiliste: Betroffene Dateien

### 3.1 Tracing-relevante Dateien (26 Dateien)

| Datei | Zeilen | Kategorie |
|-------|--------|-----------|
| ai_core/middleware/context.py | 35, 84, 90, 100, 104, 129 | Generation & Binding |
| ai_core/graph/schemas.py | 19, 31, 120-121 | Fallback-Resolution |
| ai_core/infra/resp.py | 17, 91 | Header-Injection |
| ai_core/llm/client.py | 19, 357, 764 | LLM-Header-Propagation |
| common/celery.py | 19, 176, 184, 265, 992 | Celery-Propagation |
| common/constants.py | 7, 25, 43-44, 80 | Konstanten-Definition |
| common/logging.py | 47, 306, 325, 341, 555-569 | Logger-Integration |
| noesis2/celery.py | 6, 9, 89, 108, 124 | Task-Context |
| ai_core/contracts/scope.py | 42, 86, 253 | Type-Definition |
| ai_core/ids/headers.py | 12, 143 | Header-Parsing |

### 3.2 Kwargs-kritische Dateien (15 Dateien)

| Datei | kwargs-Ops | Priorität |
|-------|------------|-----------|
| common/celery.py | 18 | **KRITISCH** |
| noesis2/api/schema.py | 8 | **HOCH** |
| ai_core/rag/vector_store.py | 3 | **HOCH** |
| ai_core/infra/observability.py | 14 | MITTEL |
| documents/logging_utils.py | 4 | MITTEL |
| ai_core/authz/visibility.py | 1 | NIEDRIG |
| users/tests/factories.py | 5 | NIEDRIG (Test) |
| customers/tests/factories.py | 1 | NIEDRIG (Test) |

---

## 4. Aufwandsschätzung

### 4.1 S-Track: Nur Tracing-Keys vereinheitlichen
**Aufwand: 0.5-1 PT / 2 Story Points**

| Task | Aufwand |
|------|---------|
| Redundante Header-Checks in noesis2/celery.py entfernen | 0.25 PT |
| Dokumentation aktualisieren | 0.25 PT |
| Tests validieren | 0.5 PT |

**Begründung:** Die Tracing-Keys sind bereits konsistent (`trace_id`). Nur minimale Cleanup-Arbeit nötig.

### 4.2 M-Track: Tracing + Kritischste Kwargs-Passagen
**Aufwand: 5-7 PT / 13 Story Points**

| Task | Aufwand |
|------|---------|
| S-Track (Tracing) | 1 PT |
| Pydantic-Modell für PIIScopedTask | 2 PT |
| Pydantic-Modell für default_extend_schema | 1 PT |
| Pydantic-Modell für HybridSearchParams | 1 PT |
| Migration + Tests | 2 PT |

**Scope:**
- common/celery.py: `TaskScopeInput`, `TaskPriorityInput`
- noesis2/api/schema.py: `ExtendSchemaOptions`
- ai_core/rag/vector_store.py: `HybridSearchParams`

### 4.3 L-Track: Vollständige Typsicherheit (Ingestion-Graphen)
**Aufwand: 15-20 PT / 34 Story Points**

| Task | Aufwand |
|------|---------|
| M-Track | 7 PT |
| Graph Input/Output Contracts überarbeiten | 5 PT |
| Observability-Wrapper typisieren | 3 PT |
| Document-Pipeline-Contracts | 3 PT |
| E2E-Validierung + Chaos-Tests | 2 PT |

**Scope:**
- Alle 5 Top-Hotspots
- ai_core/graphs/technical/*.py
- documents/logging_utils.py
- ai_core/infra/observability.py

---

## 5. Empfehlungen

### 5.1 Sofortmaßnahmen (S-Track)
1. **Keine Code-Änderungen nötig** für Tracing-Konsistenz
2. Dokumentation in AGENTS.md erweitern um explizite Tracing-Key-Policy

### 5.2 Kurzfristig (M-Track)
1. `TaskScopeInput` Pydantic-Modell für common/celery.py erstellen
2. `ExtendSchemaOptions` für API-Schema einführen
3. `HybridSearchParams` für Vector-Store typisieren

### 5.3 Mittelfristig (L-Track)
1. Vollständige Graph I/O Contracts mit `schema_id`/`schema_version`
2. Observability-Decorator mit typisierten Annotations
3. Document-Pipeline-Contracts konsolidieren

---

## 6. Appendix

### A. Grep-Patterns verwendet
```bash
# Tracing-Keys
grep -rn "trace\.id"        # 0 Treffer
grep -rn "trace_id"         # 267 Dateien
grep -rn "traceId|TRACE_ID" # Header-Konstanten

# Kwargs
grep -rn "def.*\*\*kwargs"  # 8 Funktionen
grep -rn "kwargs\.(get|pop)" # 46 Dateien
```

### B. Risiko-Matrix

| Komponente | Tracing-Risiko | Kwargs-Risiko | Gesamt |
|------------|----------------|---------------|--------|
| common/celery.py | Niedrig | **Hoch** | **Hoch** |
| ai_core/middleware/context.py | Niedrig | Niedrig | Niedrig |
| noesis2/api/schema.py | Niedrig | **Hoch** | Mittel |
| ai_core/rag/vector_store.py | Niedrig | Mittel | Mittel |
| ai_core/infra/observability.py | Niedrig | Mittel | Mittel |

---

**Autor:** Claude Code Audit
**Review:** Pending
**Nächste Schritte:** Entscheidung S/M/L-Track durch Team
