# RAG Scope Flexibilisierung (Case/Collection optional, Global möglich)

**Status:** 🔄 MERGED into [consolidated-architecture-cleanup-plan.md](consolidated-architecture-cleanup-plan.md)
**Created:** 2026-01-XX
**Updated:** 2026-01-15 (Consolidated)

---

## ⚠️ IMPORTANT: This plan has been consolidated

**Dieser Plan wurde mit [rag-tools-refactoring.md](rag-tools-refactoring.md) zusammengeführt.**

**➡️ Siehe vollständigen konsolidierten Plan:** [consolidated-architecture-cleanup-plan.md](consolidated-architecture-cleanup-plan.md)

**Entscheidungen getroffen (2026-01-15):**
- ✅ **Global Scope:** Workbench-weite Auswahl (Collection/Case) gilt für gesamte Workbench, Chat folgt dieser Auswahl
- ✅ **Kein `dev-case-local` Fallback:** Wird entfernt (P0-SCOPE-1)
- ✅ **WebSocket Async:** Bereits implementiert
- ✅ **Context Helper:** Bereits implementiert, wird in P0-SCOPE-2 für WebSocket unified

---

## Original Plan (Reference)

### Zielbild (bereits in konsolidiertem Plan)
- RAG-Queries können wahlweise tenant-global, fallgebunden (case_id), sammlungsgebunden (collection_id) oder kombiniert laufen.
- Dev-UI liefert wählbare Scopes (Global / Case / Collection), ohne erzwungenes `dev-case-local` Fallback.
- Cases/Collections sind in der UI filterbar (insb. nach Status) und leicht auswählbar (Labels + IDs).

### Offene Entscheidungen → RESOLVED
- ~~Default-Handling: Bei fehlender Auswahl → tenant-global (keine case/collection Filter) statt `dev-case-local`?~~ → ✅ **JA, Global Scope** (Workbench-weite Auswahl)
- ~~Fallback-Collection: Soll "manual collection" automatisch genutzt werden?~~ → ❌ **NEIN**, keine automatische Fallback-Collection

## Tasks
1) **Context-/Scope-Refactor (Backend)**
   - `theme/helpers/context.py:prepare_workbench_context`: keine erzwungene `case_id`; aktiven Scope (global/case/collection) berücksichtigen; optionalen `collection_id`/`case_id` durchreichen.
   - `theme/views_chat.py` / `views_rag_tools.py`: Scope-Option validieren, Session setzen, keinen Dev-Fallback injizieren; `manual_collection` nur, wenn explizit gewünscht.
   - `ai_core/graph/schemas.py:normalize_meta` / `retrieval_augmented_generation.py`: sicherstellen, dass fehlende `case_id`/`collection_id` zu **keinem** Filter führen (kein stilles Default-Case). Dokumentiere Filterlogik.

2) **UI/UX-Vorbereitung (Dev-UI, spätere Prod-UI anschlussfähig)**
   - `theme/templates/theme/workbench.html` + `partials/tool_chat.html`: Scope-Schalter klar (Global/Case/Collection), aktive Auswahl in Hidden-Fields / Form-Posts persistieren.
   - `theme/views_rag_tools.py:tool_chat/tool_search`: Kontext an Templates liefern (case/collection + scope) ohne hartes Default; im Chat nur den Global Scope zulassen („Scoping UI“ bleibt sicht­bar für Konsistenz, sendet aber immer `case_id=None`/`collection_id=None`, damit nur globale RAG-Zugriffe laufen).

3) **Cases & Collections Listing mit Status**
   - Cases: Endpoint/Helper, der Cases mit Status liefert (z.B. `cases` API oder dedicated helper) inkl. Sortierung/Filter (Status, Titel). Dev-UI nutzt das für Dropdown/Autocomplete.
   - Collections: Endpoint/Helper (mit Authz-Hooks später) für Browsing/Filter.

4) **Tests & Logging**
   - Adjust theme tests (`test_tool_search_context.py`, `test_workbench_context.py`) auf neues Default (kein dev-case Zwang).
   - Add regression test: Global scope → no case/collection filter; Case scope → filter by case; Collection scope → filter by collection.
   - Add debug logging around resolved scope/case/collection in chat/search views.

## Acceptance Criteria
- RAG Chat/Search laufen tenant-global, wenn kein Case/Collection gewählt (kein stilles `dev-case-local`).
- Bei Scope=Case wird `case_id` gesetzt, `collection_id` optional; bei Scope=Collection wird `collection_id` gesetzt, `case_id` optional; Global setzt keines von beiden (optional manual-collection nur per Opt-in).
- Dev-UI zeigt Scope-Auswahl + Case/Collection Auswahl; Cases können mit Status angezeigt/gefunden werden.
- Tests für alle Scopes grün; Logs zeigen gelösten Scope + IDs zur Diagnose.
