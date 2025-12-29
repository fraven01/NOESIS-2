# Option A: Contract Impact Analysis

**Datum**: 2025-12-27
**Scope**: Strikte Trennung - Auswirkungen auf Pydantic-Verträge
**Status**: ⚠️ BREAKING CHANGES (Pre-MVP approved)

---

## Executive Summary

Option A führt **3 neue Verträge** ein und **ändert fundamental** die Struktur von:
- ✅ `ScopeContext` (reduziert auf Correlation IDs)
- ✅ `BusinessContext` (NEU - extrahiert aus ScopeContext)
- ✅ `ToolContext` (Komposition statt flache Struktur)
- ✅ Alle Tool-Input-Modelle (Context-IDs raus)

**AGENTS.md Stop Condition erfüllt**: Ja (Zeile 12: "anything affecting `ScopeContext`, `ToolContext`")
**Pre-MVP Breaking Change Erlaubnis**: ✅ Ja, vom User bestätigt

---

## 1. Neue Verträge (zu erstellen)

### 1.1 BusinessContext (komplett neu)

**Datei**: `ai_core/contracts/business.py` (NEU)

```python
"""Business domain context for tool and graph invocations.

BusinessContext captures the domain-specific identifiers that describe
WHAT is being processed, independent of WHO is processing it (ScopeContext)
or HOW it's being processed (ToolContext runtime metadata).

Separation rationale:
- Business IDs (case, document, collection) are domain concepts
- Scope IDs (tenant, trace, invocation) are infrastructure concerns
- Clear separation enables independent evolution and testing
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

CaseId = str | None
CollectionId = str | None
WorkflowId = str | None
DocumentId = str | None
DocumentVersionId = str | None


class BusinessContext(BaseModel):
    """Business domain identifiers for case/document/collection scoping.

    All fields are optional because not every operation requires all business context.
    Individual tools/graphs validate required fields based on their needs.

    Examples:
    - Document ingestion: requires document_id, collection_id
    - Case retrieval: requires case_id, collection_id
    - Global search: may omit case_id for cross-case search
    """

    case_id: CaseId = Field(
        default=None,
        description="Case identifier. Required for case-scoped operations.",
    )

    collection_id: CollectionId = Field(
        default=None,
        description="Collection identifier ('Aktenschrank'). Required for scoped retrieval.",
    )

    workflow_id: WorkflowId = Field(
        default=None,
        description="Workflow identifier within a case. May span multiple runs.",
    )

    document_id: DocumentId = Field(
        default=None,
        description="Document identifier. Required for document operations.",
    )

    document_version_id: DocumentVersionId = Field(
        default=None,
        description="Document version identifier for versioned operations.",
    )

    model_config = ConfigDict(frozen=True)


__all__ = [
    "BusinessContext",
    "CaseId",
    "CollectionId",
    "WorkflowId",
    "DocumentId",
    "DocumentVersionId",
]
```

**Vertraglich fixiert**:
- ✅ Alle Felder optional (Tools validieren selbst, was sie brauchen)
- ✅ Immutable (`frozen=True`)
- ✅ JSON-serialisierbar (nur str | None Typen)
- ✅ Klar dokumentiert: WHAT, nicht WHO oder HOW

---

### 1.2 ScopeContext (reduziert)

**Datei**: `ai_core/contracts/scope.py` (BREAKING CHANGE)

**Raus damit**:
- ❌ `case_id` → BusinessContext
- ❌ `collection_id` → BusinessContext
- ❌ `workflow_id` → BusinessContext

**Was bleibt**:
```python
class ScopeContext(BaseModel):
    """Request correlation scope - WHO and WHEN.

    ScopeContext captures infrastructure-level identifiers for request correlation,
    tracing, and tenant isolation. It does NOT contain business domain identifiers
    (case, document, collection) - those live in BusinessContext.

    BREAKING CHANGE from v1:
    - Removed: case_id, collection_id, workflow_id (moved to BusinessContext)
    - This is a stricter separation of concerns for cleaner architecture
    """

    # Mandatory correlation IDs
    tenant_id: TenantId
    trace_id: TraceId
    invocation_id: InvocationId

    # Identity IDs (mutually exclusive per hop type)
    user_id: UserId = Field(default=None, description="...")
    service_id: ServiceId = Field(default=None, description="...")

    # Runtime IDs (may co-exist when workflow triggers ingestion)
    run_id: RunId | None = None
    ingestion_run_id: IngestionRunId | None = None

    # Optional technical context
    tenant_schema: TenantSchema | None = None
    idempotency_key: IdempotencyKey | None = None
    timestamp: Timestamp = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    # Validatoren bleiben gleich
    @model_validator(mode="after")
    def validate_run_scope(self) -> "ScopeContext": ...

    @model_validator(mode="after")
    def validate_identity(self) -> "ScopeContext": ...
```

**Vertraglich fixiert**:
- ✅ NUR Correlation & Runtime IDs
- ✅ KEINE Business-Domain-IDs mehr
- ✅ Validatoren für run_id/ingestion_run_id und user_id/service_id bleiben

---

### 1.3 ToolContext (Komposition)

**Datei**: `ai_core/tool_contracts/base.py` (BREAKING CHANGE)

**Neu: Komposition statt flache Struktur**
```python
class ToolContext(BaseModel):
    """Complete tool invocation context with separated concerns.

    BREAKING CHANGE from v1:
    - Structure changed from flat to compositional
    - ScopeContext and BusinessContext are now nested objects
    - Backward compatibility via @property accessors (deprecated)

    New tools should access:
    - context.scope.tenant_id (not context.tenant_id)
    - context.business.case_id (not context.case_id)

    Legacy property accessors will be removed in future version.
    """

    model_config = ConfigDict(frozen=True)

    # === Compositional Structure (NEW) ===
    scope: ScopeContext = Field(
        description="Request correlation scope (tenant, trace, runtime IDs)"
    )

    business: BusinessContext = Field(
        description="Business domain context (case, document, collection IDs)"
    )

    # === Tool Runtime Metadata ===
    locale: str | None = None
    timeouts_ms: PositiveInt | None = None
    budget_tokens: int | None = None
    safety_mode: str | None = None
    auth: dict[str, Any] | None = None
    visibility_override_allowed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    # === Backward Compatibility Properties (DEPRECATED) ===
    # These will be removed in a future version
    # New code should use context.scope.X or context.business.X

    @property
    def tenant_id(self) -> str:
        """Deprecated: Use context.scope.tenant_id instead."""
        return self.scope.tenant_id

    @property
    def trace_id(self) -> str:
        """Deprecated: Use context.scope.trace_id instead."""
        return self.scope.trace_id

    @property
    def invocation_id(self) -> str:
        """Deprecated: Use context.scope.invocation_id instead."""
        return self.scope.invocation_id

    @property
    def user_id(self) -> str | None:
        """Deprecated: Use context.scope.user_id instead."""
        return self.scope.user_id

    @property
    def service_id(self) -> str | None:
        """Deprecated: Use context.scope.service_id instead."""
        return self.scope.service_id

    @property
    def run_id(self) -> str | None:
        """Deprecated: Use context.scope.run_id instead."""
        return self.scope.run_id

    @property
    def ingestion_run_id(self) -> str | None:
        """Deprecated: Use context.scope.ingestion_run_id instead."""
        return self.scope.ingestion_run_id

    @property
    def case_id(self) -> str | None:
        """Deprecated: Use context.business.case_id instead."""
        return self.business.case_id

    @property
    def collection_id(self) -> str | None:
        """Deprecated: Use context.business.collection_id instead."""
        return self.business.collection_id

    @property
    def workflow_id(self) -> str | None:
        """Deprecated: Use context.business.workflow_id instead."""
        return self.business.workflow_id

    @property
    def document_id(self) -> str | None:
        """Deprecated: Use context.business.document_id instead."""
        return self.business.document_id

    @property
    def document_version_id(self) -> str | None:
        """Deprecated: Use context.business.document_version_id instead."""
        return self.business.document_version_id

    @property
    def tenant_schema(self) -> str | None:
        """Deprecated: Use context.scope.tenant_schema instead."""
        return self.scope.tenant_schema

    @property
    def idempotency_key(self) -> str | None:
        """Deprecated: Use context.scope.idempotency_key instead."""
        return self.scope.idempotency_key

    @property
    def now_iso(self) -> datetime:
        """Deprecated: Use context.scope.timestamp instead."""
        return self.scope.timestamp
```

**Vertraglich fixiert**:
- ✅ Komposition: `scope: ScopeContext` + `business: BusinessContext`
- ✅ Properties für Backward Compatibility (deprecated)
- ✅ Neue Tools nutzen `context.scope.X` / `context.business.X`
- ✅ Alte Tools funktionieren via Properties (können schrittweise migriert werden)

---

### 1.4 tool_context_from_scope (erweitert)

**Datei**: `ai_core/tool_contracts/base.py`

**Neue Signatur**:
```python
def tool_context_from_scope(
    scope: ScopeContext,
    business: BusinessContext | None = None,
    *,
    now: datetime | None = None,
    **overrides: Any,
) -> ToolContext:
    """Build a ToolContext from ScopeContext and BusinessContext.

    Args:
        scope: Request correlation scope
        business: Business domain context (optional, can be empty)
        now: Override timestamp (defaults to scope.timestamp)
        **overrides: Additional ToolContext fields (locale, budgets, etc.)

    Returns:
        Complete ToolContext with compositional structure

    Example:
        scope = ScopeContext(tenant_id="...", trace_id="...", ...)
        business = BusinessContext(case_id="...", collection_id="...")
        context = tool_context_from_scope(scope, business, locale="de-DE")
    """

    if business is None:
        business = BusinessContext()  # Empty business context

    payload: dict[str, Any] = {
        "scope": scope,
        "business": business,
    }

    payload.update(overrides)

    return ToolContext(**payload)
```

**ScopeContext.to_tool_context erweitert**:
```python
# In ScopeContext:
def to_tool_context(
    self,
    business: BusinessContext | None = None,
    **overrides: object
) -> ToolContext:
    """Project this scope into a ToolContext with optional business context."""
    from ai_core.tool_contracts.base import tool_context_from_scope
    return tool_context_from_scope(self, business, **overrides)
```

---

## 2. Geänderte Verträge (bestehende Modelle)

### 2.1 Tool-Input-Modelle

**Alle Tool-Inputs verlieren Context-IDs**. Beispiele:

#### RetrieveInput (BREAKING)
```python
# Vorher (v1):
class RetrieveInput(BaseModel):
    query: str
    collection_id: str | None = None      # ❌ RAUS
    workflow_id: str | None = None        # ❌ RAUS
    visibility_override_allowed: bool | None = None  # ❌ RAUS
    filters: dict | None = None
    # ...

# Nachher (v2):
class RetrieveInput(BaseModel):
    """Pure functional parameters for retrieval.

    BREAKING CHANGE: Removed context IDs (collection_id, workflow_id).
    These are now passed via ToolContext.business instead.
    """
    query: str
    filters: dict | None = None
    process: str | None = None
    doc_class: str | None = None
    visibility: str | None = None
    hybrid: dict | None = None
    top_k: int | None = None
```

**Migration für Tool-Funktion**:
```python
# Vorher:
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    collection_id = params.collection_id or context.collection_id  # 🤯

# Nachher:
def run(context: ToolContext, params: RetrieveInput) -> RetrieveOutput:
    collection_id = context.business.collection_id  # ✅ Klar!
```

#### FrameworkAnalysisInput (BREAKING)
```python
# Vorher (v1):
class FrameworkAnalysisInput(BaseModel):
    document_collection_id: UUID  # ❌ RAUS
    document_id: UUID | None      # ❌ RAUS
    force_reanalysis: bool = False
    confidence_threshold: float = 0.70

# Nachher (v2):
class FrameworkAnalysisInput(BaseModel):
    """Framework analysis parameters.

    BREAKING CHANGE: Removed document_collection_id, document_id.
    These are now in ToolContext.business.
    """
    force_reanalysis: bool = False
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
```

**Migration**:
```python
# Vorher:
def analyze(context: ToolContext, params: FrameworkAnalysisInput):
    collection_id = params.document_collection_id  # ❌
    document_id = params.document_id

# Nachher:
def analyze(context: ToolContext, params: FrameworkAnalysisInput):
    collection_id = context.business.collection_id  # ✅
    document_id = context.business.document_id
```

### 2.2 Tool-Output-Modelle

**Keine Änderung!** ✅

`ToolResult`, `ToolError`, `ToolOutput` bleiben unverändert.

### 2.3 Graph State Schemas

**Minimale Änderung**: Meta-Dictionary enthält neue Struktur

```python
# ai_core/graph/schemas.py:normalize_meta

# Vorher:
meta = {
    "scope_context": scope.model_dump(),
    "tool_context": tool_ctx.model_dump(),
}

# Nachher:
business = BusinessContext(
    case_id=scope.case_id,  # Aus altem Scope extrahieren
    collection_id=request.headers.get("X-Collection-ID"),
    # ...
)

meta = {
    "scope_context": scope.model_dump(),      # Neue reduzierte Struktur
    "business_context": business.model_dump(),  # NEU
    "tool_context": tool_ctx.model_dump(),    # Neue Kompositionsstruktur
}
```

---

## 3. Vertragsregeln (neu zu dokumentieren)

### 3.1 Goldene Regel
> **"Context-IDs gehören in ScopeContext/BusinessContext, NIE in Tool-Inputs"**

### 3.2 Verantwortlichkeiten

| Contract | Verantwortung | Beispiele |
|----------|---------------|-----------|
| `ScopeContext` | Request Correlation (WHO, WHEN) | `tenant_id`, `trace_id`, `user_id`, `run_id` |
| `BusinessContext` | Domain Context (WHAT) | `case_id`, `document_id`, `collection_id` |
| `ToolContext` | Runtime Metadata (HOW) | `locale`, `timeouts_ms`, `budget_tokens` |
| Tool-Inputs | Functional Parameters | `query`, `filters`, `confidence_threshold` |

### 3.3 Validierungsregeln

```python
# Linting-Rule (zukünftig):
# Tool-Input-Modelle dürfen NICHT enthalten:
FORBIDDEN_FIELDS_IN_TOOL_INPUTS = {
    "tenant_id",
    "trace_id",
    "invocation_id",
    "case_id",
    "collection_id",
    "workflow_id",
    "document_id",
    "document_version_id",
    "run_id",
    "ingestion_run_id",
    "user_id",
    "service_id",
}
```

### 3.4 Migration-Path

**Phase 1 (sofort)**:
- Neue Contracts erstellen
- Properties für Backward Compatibility

**Phase 2 (schrittweise)**:
- Tools migrieren zu `context.business.X`
- Deprecation Warnings für Properties

**Phase 3 (später)**:
- Properties entfernen
- Clean Architecture etabliert

---

## 4. AGENTS.md Updates (erforderlich)

### 4.1 Neue Section: Business Context Contract

```markdown
### Business context contract

Business domain identifiers live in `ai_core/contracts/business.py`.

- `BusinessContext` is immutable (`ConfigDict(frozen=True)`).
- All fields are optional; individual tools validate required business context.
- Contains: `case_id`, `collection_id`, `workflow_id`, `document_id`, `document_version_id`.
- Separation rationale: Business IDs (WHAT) are independent of request scope (WHO/WHEN).
```

### 4.2 Update: Scope Context Contract

```markdown
### Scope context contract (BREAKING CHANGE in v2)

**REMOVED from ScopeContext**:
- `case_id` → moved to `BusinessContext`
- `collection_id` → moved to `BusinessContext`
- `workflow_id` → moved to `BusinessContext`

ScopeContext now contains ONLY request correlation identifiers:
- Tenant & Identity: `tenant_id`, `user_id`, `service_id`
- Tracing: `trace_id`, `invocation_id`
- Runtime: `run_id`, `ingestion_run_id`, `timestamp`
```

### 4.3 Update: Tool Context Contract

```markdown
### Tool context contract (BREAKING CHANGE in v2)

**New compositional structure**:
- `scope: ScopeContext` - request correlation
- `business: BusinessContext` - domain context
- Plus: `locale`, `timeouts_ms`, `budget_tokens`, etc.

**Backward compatibility**:
- Properties `context.tenant_id`, `context.case_id` etc. still work (deprecated)
- New code should use `context.scope.tenant_id`, `context.business.case_id`

**Build from contexts**:
```python
business = BusinessContext(case_id="...", collection_id="...")
context = scope.to_tool_context(business=business)
# or
context = tool_context_from_scope(scope, business)
```
```

### 4.4 New: Tool Input Rules

```markdown
### Tool input contracts (ENFORCED)

**Golden Rule**: Tool-Input models MUST NOT contain context identifiers.

Forbidden fields:
- Any ScopeContext field (`tenant_id`, `trace_id`, `user_id`, etc.)
- Any BusinessContext field (`case_id`, `collection_id`, etc.)

Tool-Inputs contain ONLY functional parameters specific to the tool's operation.

Example:
```python
# ✅ GOOD
class RetrieveInput(BaseModel):
    query: str
    filters: dict | None
    top_k: int | None

# ❌ BAD
class RetrieveInput(BaseModel):
    query: str
    collection_id: str | None  # ← FORBIDDEN! Use ToolContext.business instead
```
```

---

## 5. Rückwärtskompatibilität

### 5.1 Was bricht sofort?

1. **Tool-Input-Konstruktion**: Code, der Context-IDs in Tool-Inputs setzt
   ```python
   # Bricht:
   params = RetrieveInput(query="...", collection_id="...")  # ❌ Field removed
   ```

2. **ToolContext-Konstruktion**: Direkter Constructor-Call
   ```python
   # Bricht:
   context = ToolContext(tenant_id="...", case_id="...")  # ❌ Signature changed
   ```

3. **Direkte Feld-Zugriffe** (falls keine Properties):
   ```python
   # Bricht (ohne Properties):
   context.case_id  # ❌ Field doesn't exist

   # Funktioniert (mit Properties):
   context.case_id  # ✅ Property returns context.business.case_id
   ```

### 5.2 Was funktioniert weiter?

1. **Tool-Funktionen mit Properties**:
   ```python
   # Funktioniert weiter:
   def run(context: ToolContext, params: Input):
       case_id = context.case_id  # ✅ Property
   ```

2. **Helper-Funktionen**:
   ```python
   # Funktioniert weiter:
   context = scope.to_tool_context(business=business)  # ✅ Angepasst
   ```

3. **JSON-Serialisierung**:
   ```python
   # Funktioniert (aber Struktur ändert sich):
   context.model_dump()  # ✅ Neue nested Struktur
   ```

---

## 6. Migration Checklist

### Phase 1: Contracts erstellen
- [ ] `ai_core/contracts/business.py` erstellen
- [ ] `ScopeContext` reduzieren (Business-IDs raus)
- [ ] `ToolContext` mit Komposition umbauen
- [ ] `tool_context_from_scope()` erweitern
- [ ] Unit-Tests für neue Contracts

### Phase 2: Tool-Inputs säubern
- [ ] `RetrieveInput`: `collection_id`, `workflow_id` raus
- [ ] `FrameworkAnalysisInput`: `document_collection_id`, `document_id` raus
- [ ] Alle anderen Tool-Inputs durchgehen
- [ ] Integration-Tests anpassen

### Phase 3: Tool-Funktionen migrieren
- [ ] `ai_core/nodes/retrieve.py:run()` anpassen
- [ ] Framework Analysis Graph anpassen
- [ ] Alle Tool-Run-Funktionen durchgehen
- [ ] E2E-Tests anpassen

### Phase 4: Normalizer anpassen
- [ ] `normalize_meta()` für BusinessContext erweitern
- [ ] `normalize_request()` für neue Struktur
- [ ] `normalize_task_context()` für Celery

### Phase 5: Dokumentation
- [ ] AGENTS.md updaten (alle Sections)
- [ ] CLAUDE.md updaten
- [ ] Migration-Guide schreiben
- [ ] ADR oder Backlog-Item erstellen

---

## 7. Risiken & Mitigations

| Risiko | Impact | Mitigation |
|--------|--------|------------|
| Viele Tests brechen | 🔴 High | Properties halten Tests erstmal am Laufen |
| Entwickler verwirrt | 🟡 Medium | Klare Doku + Migration-Guide |
| Performance (Properties) | 🟢 Low | Properties sind trivial, kein Impact |
| Vergessene Migrations | 🟡 Medium | Linting-Rule für Tool-Inputs (später) |

---

## 8. Entscheidung erforderlich

**Der User hat bereits grünes Licht gegeben für "hart breaken".**

Nächste Schritte:
1. ✅ Bestätigung: Ist dieser Contract-Impact akzeptabel?
2. 🚀 Start: Soll ich mit Phase 1 beginnen (Contracts erstellen)?

---

**Erstellt von**: Claude Sonnet 4.5
**Status**: Waiting for confirmation to proceed
**Nächster Schritt**: Phase 1 - BusinessContext & reduzierter ScopeContext
