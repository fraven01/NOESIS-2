# Architecture Anti-Patterns Analysis

**Date**: 2025-12-31
**Scope**: NOESIS-2 Codebase (~674 Python files)
**Status**: Pre-MVP, Breaking Changes Allowed
**Related Backlog**: [roadmap/backlog.md](../../roadmap/backlog.md#code-quality--architecture-cleanup-pre-mvp-refactoring)

---

## Executive Summary

Comprehensive architectural analysis identified **critical "vibe coding" patterns** across the codebase. While the foundation is solid (Pydantic, Type Hints, LangGraph), the connective tissue is bloated with unnecessary abstraction layers.

**Impact**: Estimated **20-30% code reduction possible** without functionality loss.

### Key Metrics

| Metric | Count | Critical Files |
|--------|-------|----------------|
| Private helper functions (views.py) | ~17 | [theme/views.py](../../theme/views.py) |
| Normalize/convert functions | ~54 | 36 files |
| Error raise sites | ~395 | 81 files |
| Adapter/Wrapper classes | ~8 | 8 files |
| Lines in largest service | 2,034 | [ai_core/services/__init__.py](../../ai_core/services/__init__.py) |
| Lines in largest view | 2,055 | [theme/views.py](../../theme/views.py) |

_Counts are approximate snapshots; re-run counts before executing refactors._

---

## 1. Glue Code Inflation
### Severity: HIGH

**Definition**: Pass-through functions that move data without adding business logic.

### Evidence: theme/views.py (17 Private Helpers)

**File**: [theme/views.py](../../theme/views.py)

```python
Resolved: pass-through helpers removed from `theme/views.py`; validation now lives in
`theme/validators.py` with Pydantic field validators.

```python
from theme.validators import DocumentSpaceQueryParams

params = DocumentSpaceQueryParams.model_validate(request.GET)
```

### Complete List of Pass-Through Functions

| Function | Status | Purpose | Alternative |
|----------|-------|---------|-------------|
| `_parse_bool()` | removed | Boolean parsing | Pydantic `@field_validator` |
| `_parse_limit()` | removed | Int clamping | Pydantic `@field_validator` with `ge=5, le=200` |
| `_extract_user_id()` | removed | Attribute access | Inline extraction in views |
| `_human_readable_bytes()` | removed | Formatting | Template filter or frontend |
| `_normalise_quality_mode()` | removed | String validation | Pydantic `Literal` type |
| `_normalise_max_candidates()` | removed | Int clamping | Pydantic field constraint |

**Note**: Helpers that include IO or domain logic (tenant resolution, collection lookup, cache/LLM routing) are not pass-through. These should be refactored or relocated, not deleted (e.g. `_resolve_tenant_context`, `_resolve_manual_collection`, `_normalize_collection_id`, `_resolve_rerank_model_preset`).

### Impact

- **Maintainability**: Logic scattered across helpers instead of models
- **Testability**: Each helper needs separate unit tests
- **Type Safety**: Dict-based validation instead of Pydantic
- **DX**: Developers must hunt through helpers to understand validation

### Recommendation

**P1 Task**: [Eliminate Pass-Through Glue Functions](../../roadmap/backlog.md#p1---high-value-cleanups-low-medium-effort)

**Implementation**:
```python
# BEFORE: theme/views.py helpers
def _parse_bool(value, default=False): ...
def _parse_limit(value, default=25): ...

# AFTER: theme/validators.py
from pydantic import BaseModel, Field, field_validator

class SearchQueryParams(BaseModel):
    enabled: bool = False
    limit: int = Field(default=25, ge=5, le=200)

    @field_validator('enabled', mode='before')
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, bool): return v
        return str(v).lower() in ('true', '1', 'yes')
```

---

## 2. Normalizer Syndrome
### Severity: HIGH

**Definition**: Excessive conversion functions that are nearly identical, indicating copy-paste over abstraction.

### Evidence: 54 Normalize/Convert Functions

**Primary Offender (resolved)**: [ai_core/services/__init__.py](../../ai_core/services/__init__.py)

```python
_JSON_ADAPTER = TypeAdapter(Any)

def _dump_jsonable(value: Any) -> Any:
    """Return a structure that json.dumps can serialise."""
    return _JSON_ADAPTER.dump_python(value, mode="json")
```

**Previous anti-pattern**: a 43-line `_make_json_safe` function that hand-rolled type conversions (removed).

### Pattern Duplication: theme/views.py

```python
# Line 264
def _normalise_quality_mode(value: object, default: str = "standard") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in {"standard", "premium", "fast"}:
        return candidate
    return default

# Line 271
def _normalise_max_candidates(value: object, *, default: int = 20) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = default
    return max(5, min(40, numeric))

```

**Problem**: Many functions do similar normalization with small variations. Consolidate only truly identical patterns (e.g., strip/lower, int clamping) to avoid behavior drift.

### Distribution Across Codebase

| Pattern | Count | Locations |
|---------|-------|-----------|
| `normalize_*` / `normalise_*` | 18 | theme/views.py, ai_core/services/ |
| `convert_*` / `to_*` / `from_*` | 22 | ai_core/tools/, documents/ |
| `map_*` | 6 | theme/views.py, api/ |
| `_make_*_safe` | 8 | ai_core/services/, common/ |

### Adapter Classes (Ceremonial Wrappers)

**File**: [theme/views.py:439-504](../../theme/views.py#L439-L504)

```python
class _ViewCrawlerIngestionAdapter:
    """Adapter that triggers crawler ingestion by calling run_crawler_runner.

    55 lines that just... reformat args and call another function.
    No transformation logic, no validation, just dict reshaping.
    """

    def trigger(self, *, url: str, collection_id: str, context: Mapping[str, str]):
        tenant_id = context.get("tenant_id", "dev")
        # ... build payload dict ...
        result = run_crawler_runner(meta=meta, request_model=request_model, ...)
        return {"decision": "ingested", ...}
```

**Problem**: This adapter exists solely to reshape dictionaries. No value-add transformation.

### Impact

- **Code Volume**: 54 functions doing similar work = ~800 lines of boilerplate
- **Maintenance Burden**: Changes to normalization require updating multiple functions
- **Bug Risk**: Inconsistent normalization across different code paths
- **Onboarding**: New developers confused by function proliferation

### Recommendation

**P0 Task**: [Kill JSON Normalization Boilerplate](../../roadmap/backlog.md#p0---critical-quick-wins-high-impact-medium-effort)
**P1 Task**: [Normalize the Normalizers](../../roadmap/backlog.md#p1---high-value-cleanups-low-medium-effort)

**Implementation**:
```python
# BEFORE (removed): ai_core/services/__init__.py
def _make_json_safe(value): ...  # 43 lines of manual conversions

# AFTER: Use Pydantic JSON adapters for mixed payloads
from typing import Any
from pydantic import TypeAdapter

result = TypeAdapter(Any).dump_python(payload, mode="json")

# BEFORE: Multiple normalize_* functions
def _strip_lower(v): return str(v).strip().lower()
def _clamp_int(v): return max(5, min(40, int(v)))

# AFTER: Shared validator in common/validators.py
from pydantic import field_validator

class StripLowerStr(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return str(v).strip().lower()

class QualityParams(BaseModel):
    mode: StripLowerStr
    max_candidates: StripLowerStr
```

---

## 3. Anemic Domain Model
### Severity: CRITICAL

**Definition**: Domain models are pure data containers with no behavior, while "service" classes contain all procedural logic.

### Evidence: framework_contracts.py

**File**: [ai_core/tools/framework_contracts.py](../../ai_core/tools/framework_contracts.py)

All classes are frozen Pydantic models with **zero methods**:

```python
# Lines 16-23
class TypeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: str
    field_type: str
    occurrence_count: int
    # NO METHODS, NO BUSINESS LOGIC!

# Lines 26-33
class AlternativeType(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type_name: str
    confidence: float
    reasoning: str
    # NO METHODS!

# Lines 74-92
class ComponentLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 12 fields, ZERO methods
```

### Fat Service Layer

**File**: [ai_core/services/__init__.py](../../ai_core/services/__init__.py) - **2,034 lines**

```python
# Lines 776-1231: 455 lines of procedural logic
def execute_graph(meta, graph_type, state):
    """God function doing everything."""
    # Validation
    # State mutation
    # Graph execution
    # Result processing
    # Error handling
    # Logging
    # Metrics recording
    # 455 lines of sequential operations!

# Lines 1234-1367: 133 lines
def start_ingestion_run(meta, collection_id, ...):
    """Another god function."""
    # 133 lines of orchestration

# Lines 1577-2033: 456 lines
def handle_document_upload(meta, file, ...):
    """Massive procedural document processing."""
    # 456 lines of sequential steps
```

### Anti-Pattern Diagram

```
Anemic Models (data only)
        |
        v
Fat Services (procedural logic)

```

**Classic OOP Anti-Pattern**: Should be reversed.

### Impact

- **Testability**: God functions hard to test, require full integration setup
- **Reusability**: Logic tied to service layer, can't reuse model behavior
- **Cohesion**: Data and behavior separated, violating encapsulation
- **Complexity**: Services become procedural scripts, not objects

### Recommendation

**P2 Task**: [Targeted Domain Enrichment](../../roadmap/backlog.md#p2---long-term-improvements-high-effort)

**Implementation**:
```python
# BEFORE: Anemic model
class TypeEvidence(BaseModel):
    field_name: str
    occurrence_count: int

# Service has the logic
def is_strong_evidence(evidence: TypeEvidence) -> bool:
    return evidence.occurrence_count > 5

# AFTER: Rich domain model
class TypeEvidence(BaseModel):
    field_name: str
    occurrence_count: int

    @property
    def is_strong(self) -> bool:
        """Business logic belongs in the model."""
        return self.occurrence_count > 5

    def merge_with(self, other: TypeEvidence) -> TypeEvidence:
        """Domain operations on domain objects."""
        if self.field_name != other.field_name:
            raise ValueError("Cannot merge different fields")
        return TypeEvidence(
            field_name=self.field_name,
            occurrence_count=self.occurrence_count + other.occurrence_count
        )
```

---

## 4. Boilerplate Hallucination
### Severity: MEDIUM

**Definition**: Design patterns implemented without clear value, adding complexity for "professional appearance."

### Evidence: Fake Builder Pattern

**File**: [ai_core/services/__init__.py:307-326](../../ai_core/services/__init__.py#L307-L326)

**Status**: RESOLVED (DocumentComponents removed; call sites now import storage/captioner directly.)

```python
class DocumentComponents:
    """Container for document processing pipeline components."""

    def __init__(self, storage, captioner):
        self.storage = storage
        self.captioner = captioner

def get_document_components() -> DocumentComponents:
    """Return default document processing components."""
    from documents.storage import ObjectStoreStorage
    from documents.captioning import DeterministicCaptioner

    return DocumentComponents(
        storage=ObjectStoreStorage,  # Just passing class references!
        captioner=DeterministicCaptioner,
    )
```

**Problem**: This is "Builder Pattern Cosplay." It's just a named tuple of class references. Could be:
```python
# Direct imports (KISS principle)
from documents.storage import ObjectStoreStorage
from documents.captioning import DeterministicCaptioner

# Use directly, no wrapper needed
```

### Evidence: Fake Strategy Pattern

**File**: [ai_core/services/__init__.py:128-138](../../ai_core/services/__init__.py#L128-L138)

**Status**: RESOLVED (services now uses the real `ai_core.infra.ledger` module.)

```python
class _LedgerShim:
    """Shim for recording ledger entries (replaced in tests)."""

    def record(self, meta):
        # No-op by default; tests replace this with a spy.
        try:
            _ = dict(meta)  # "force materialization for safety"
        except Exception:
            pass
        return None

ledger = _LedgerShim()
```

**Problem**: This pretends to be a Strategy pattern for testability, but is actually:
- A glorified no-op in production
- Comment says "tests replace this" -> monkey-patching instead of DI
- The `try/except` does nothing (silent failure)

**Better approach**: Proper dependency injection or remove if unused.

### Pattern Inventory

| Pattern | Location | Purpose | Assessment |
|---------|----------|---------|------------|
| DocumentComponents | services/__init__.py:307 | "Container" | Fake builder, just a tuple |
| _LedgerShim | services/__init__.py:128 | "Strategy" | No-op masquerading as abstraction |
| _ViewCrawlerIngestionAdapter | theme/views.py:439 | "Adapter" | Dictionary reshaper, no logic |

### Impact

- **Cognitive Load**: Developers must understand unnecessary abstractions
- **Indirection**: More classes to navigate for simple operations
- **False Complexity**: Gives impression of sophisticated design without benefits
- **Maintenance**: More code to maintain for zero value

### Recommendation

**P1 Task**: [Remove Fake Abstractions](../../roadmap/backlog.md#p1---high-value-cleanups-low-medium-effort)

**Implementation**:
```python
# BEFORE: Fake abstraction
class DocumentComponents:
    def __init__(self, storage, captioner):
        self.storage = storage
        self.captioner = captioner

# AFTER: Direct usage (KISS)
from documents.storage import ObjectStoreStorage
from documents.captioning import DeterministicCaptioner

storage = ObjectStoreStorage()
captioner = DeterministicCaptioner()
```

---

## 5. Fragmented Logic
### Severity: HIGH

**Definition**: Inconsistent error handling and logging strategies across the codebase.

### Evidence: 4 Different Error Patterns

**Pattern 1: Typed Exceptions** ([ai_core/tool_contracts/__init__.py](../../ai_core/tool_contracts/__init__.py))
```python
from ai_core.tool_contracts import (
    InputError,
    NotFoundError,
    RateLimitedError,
    TimeoutError,
    UpstreamServiceError,
)
```

**Pattern 2: Pydantic ValidationError** (pervasive)
```python
try:
    validated = Model.model_validate(data)
except ValidationError as exc:
    return _error_response(str(exc), "validation_error", 400)
```

**Pattern 3: Custom Graph Exceptions** ([ai_core/graphs/technical/universal_ingestion_graph.py](../../ai_core/graphs/technical/universal_ingestion_graph.py))
```python
class UniversalIngestionError(Exception):
    """Error raised during universal ingestion graph execution."""
```

**Pattern 4: String-based Error Codes** ([theme/views.py](../../theme/views.py))
```python
return _error_response("Upload failed", "upload_failed", 500)
```

### Error Handling Metrics

| Metric | Count |
|--------|-------|
| Total error raise sites | 395 |
| Files with error handling | 81 |
| Different error patterns | 4 |
| Custom exception classes | 12 |

### Logging Chaos

**File**: [theme/views.py](../../theme/views.py)

```python
# Line 860, 869, 1132
logger.info("Processing request")

# Line 500
logger.exception("Failed to process")

# Line 1055, 1075
logger.warning("Invalid parameter")

# Line 1214 - PRODUCTION CODE! (removed)
print("Debug: user_id =", user_id)
```

**Problem (resolved)**: `print()` statement in production code (removed; structured logging enforced).

**File**: [ai_core/services/crawler_runner.py](../../ai_core/services/crawler_runner.py)

```python
# Lines 186, 258, 273 - Structured logging
logger.info("Crawler started", extra={"tenant_id": tenant_id, "url": url})

# Line 319 - Unstructured warning
logger.warning("Rate limit approaching")

# Mixed conventions: `logger` vs `LOGGER`
```

### Impact

- **Inconsistency**: No standard way to handle errors
- **Debugging**: Hard to trace errors across layers
- **Monitoring**: Can't reliably alert on error types
- **Client Experience**: Inconsistent error responses

### Recommendation

**P0 Task**: [Standardize Error Handling](../../roadmap/backlog.md#p0---critical-quick-wins-high-impact-medium-effort)
**Observability Task**: [Fix Logging Chaos](../../roadmap/backlog.md#observability-cleanup) (done)

**Implementation**:
```python
# ai_core/errors.py (boundary mapping helper)
from pydantic import ValidationError
from ai_core.tool_contracts import ToolError, ToolErrorDetail, ToolErrorMeta
from ai_core.tools.errors import ToolErrorType

def tool_error_response(input_payload, *, message, error_type, code=None, details=None, took_ms=0):
    return ToolError(
        input=input_payload,
        error=ToolErrorDetail(type=error_type, message=message, code=code, details=details),
        meta=ToolErrorMeta(took_ms=took_ms),
    )

# Boundary handler example
except ValidationError as exc:
    return tool_error_response(
        input_payload,
        message=str(exc),
        error_type=ToolErrorType.VALIDATION,
        code="validation_error",
    )
```

**Logging Standards**:
```python
# ALWAYS include context
logger.info(
    "Graph executed successfully",
    extra={
        "tenant_id": str(ctx.scope.tenant_id),
        "trace_id": ctx.scope.trace_id,
        "graph_type": graph_type,
        "duration_ms": duration,
    }
)

# NEVER use print()
# ALWAYS use structured logging (extra={}) - see `docs/observability/logging-standards.md`
```

---

## 6. Context-Propagation Spaghetti
### Severity: HIGH

**Definition**: Manual dictionary unpacking repeated across 50+ locations instead of typed context objects.

### Evidence: Repeated Pattern

**Files**:
- [ai_core/services/crawler_runner.py:63-67, 80-83, 122-123](../../ai_core/services/crawler_runner.py)
- [ai_core/services/__init__.py:808-816, 1251-1254, 1626-1627](../../ai_core/services/__init__.py)
- [theme/views.py:934-947, 1164-1176, 1443-1452](../../theme/views.py)

```python
# REPEATED 50+ TIMES across codebase
scope_context = meta["scope_context"]
business_context = meta.get("business_context", {})
tenant_id = scope_context["tenant_id"]
case_id = business_context.get("case_id")
trace_id = scope_context["trace_id"]
invocation_id = scope_context["invocation_id"]
run_id = scope_context.get("run_id")
```

### Problems

1. **No Type Safety**: Typos in dict keys caught at runtime
2. **Duplication**: Same unpacking logic copy-pasted
3. **Fragility**: Changes to structure require updates in 50+ locations
4. **IDE Support**: No autocomplete, no refactoring support

### Impact

- **Error Prone**: KeyError risk on every access
- **Maintenance**: Structural changes require mass updates
- **Onboarding**: New developers copy-paste without understanding
- **Testing**: Hard to mock/stub context

### Recommendation

**P0 Task**: [ToolContext-First Context Access](../../roadmap/backlog.md#p0---critical-quick-wins-high-impact-medium-effort)

**Implementation**:
```python
# ai_core/tool_contracts/context_helpers.py (NEW)
from ai_core.tool_contracts import ToolContext

def tool_context_from_meta(meta: dict) -> ToolContext:
    """Parse the existing ToolContext contract from meta."""
    return ToolContext.model_validate(meta["tool_context"])

# USAGE: Replace 50+ dict unpacking sites
ctx = tool_context_from_meta(meta)
tenant_id = ctx.scope.tenant_id
case_id = ctx.business.case_id
trace_id = ctx.scope.trace_id
```

---

## 7. State Dict Manipulation
### Severity: MEDIUM

**Definition**: Building nested dictionaries manually instead of using typed dataclasses.

### Evidence

**File**: [ai_core/services/crawler_runner.py:298-314](../../ai_core/services/crawler_runner.py#L298-L314)

```python
synthesized_state = {
    "artifacts": result.get("artifacts", {}),
    "transitions": output.get("transitions", []),
    "control": build.state.get("control", {}),
}

entry = {
    "build": build,
    "result": {
        "decision": output.get("decision"),
        "reason": output.get("reason"),
    },
    "state": synthesized_state,
}
```

### Problems

- **Type Safety**: No validation, runtime KeyErrors
- **IDE Support**: No autocomplete for keys
- **Maintenance**: Structural changes break silently
- **Documentation**: Dict shape implicit, not documented

### Recommendation

**Hygiene Task**: [State Dict -> Dataclasses](../../roadmap/backlog.md#hygiene)

**Implementation**:
```python
# BEFORE: Manual dict building
synthesized_state = {
    "artifacts": result.get("artifacts", {}),
    "transitions": output.get("transitions", []),
}

# AFTER: Typed dataclass
from dataclasses import dataclass
from typing import Any

@dataclass
class SynthesizedState:
    artifacts: dict[str, Any]
    transitions: list[dict]
    control: dict[str, Any]

    @classmethod
    def from_result(cls, result: dict, build_state: dict):
        return cls(
            artifacts=result.get("artifacts", {}),
            transitions=result.get("transitions", []),
            control=build_state.get("control", {}),
        )

# Type-safe, validated, documented
state = SynthesizedState.from_result(result, build.state)
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2) - Breaking Changes Allowed

**P0 Tasks** - Must complete before other refactoring:

1. **ToolContext-First Context Access** BREAKING
   - Use `ToolContext.model_validate(meta["tool_context"])` (or helper)
   - Migrate 50+ dict unpacking sites to `context.scope.*` / `context.business.*`
   - Update all tests
   - **Blocker for**: All other refactorings depend on this

2. **Kill JSON Normalization** BREAKING
   - Delete `_make_json_safe()` (43 lines) (done: replaced with `_dump_jsonable`)
   - Replace with `TypeAdapter(Any).dump_python(..., mode="json")` or `pydantic_core.to_jsonable_python`
   - **Quick win**: Immediate -43 lines

3. **Standardize Error Handling** BREAKING (done)
   - Keep `ToolErrorType` as the single response contract
   - Migrate error sites to boundary mapping
   - Add unified error-to-response mapping (see `ai_core/infra/resp.py:build_tool_error_payload`)
   - **Blocker for**: Consistent observability

### Phase 2: Cleanup (Week 3-4)

**P1 Tasks** - High value, low-medium effort:

4. **Eliminate Pass-Through Functions** (done)
   - Create `theme/validators.py` with Pydantic models (done)
   - Remove helper functions from views.py (done)
   - **Benefit**: -150+ lines, better validation

5. **Normalize the Normalizers**
   - Create `common/validators.py` shared validators
   - Consolidate 54 -> <10 distinct normalization patterns (no behavior drift)
   - **Benefit**: -600+ lines, consistent validation

6. **Remove Fake Abstractions**
   - Delete DocumentComponents, _LedgerShim, adapters
   - Replace with direct imports or proper DI
   - **Benefit**: -200+ lines, reduced complexity

7. **Fix Logging Chaos**
   - Remove all `print()` statements
   - Enforce structured logging standards
   - **Benefit**: Better observability

### Phase 3: Architecture (Week 5-8)

**P2 Tasks** - Long-term improvements:

8. **Break Up God Files** BREAKING
   - Split `ai_core/services/__init__.py` (2,034 lines)
   - Split `theme/views.py` (2,045 lines)
   - **Benefit**: -4,000+ lines -> modular structure

9. **Targeted Domain Enrichment** BREAKING
   - Add small, local methods where duplication exists
   - Move repeated logic from services into models
   - **Benefit**: Better encapsulation, testability

10. **Service Layer Refactoring**
    - Replace god-functions with Command Pattern
    - Extract graph execution logic
    - **Benefit**: Cleaner separation of concerns

---

## Success Metrics

### Code Quality

- [ ] No file >500 lines
- [ ] <10 normalization functions total
- [ ] Single error contract in responses (ToolErrorType)
- [ ] Zero `print()` statements in production code
- [ ] All logging structured with context

### Type Safety

- [ ] Context accessed via typed objects (not dicts)
- [ ] State management via dataclasses (not manual dicts)
- [ ] JSON serialization via Pydantic (not manual functions)

### Maintainability

- [ ] 20-30% code reduction achieved
- [ ] No pass-through functions without clear value
- [ ] No fake design patterns (builders/adapters with no logic)
- [ ] Domain models have behavior (not anemic DTOs)

---

## Risk Assessment

### Low Risk (P0, P1 Tasks)

- **ToolContext-First Context Access**: Mechanical refactoring, type-safe
- **Kill JSON Normalization**: Pydantic JSON adapters, predictable output
- **Eliminate Pass-Through**: Inline or move to validators
- **Fix Logging**: Purely operational, no behavior change

### Medium Risk (P2 Tasks)

- **Break Up God Files**: Large refactoring, requires careful module design
- **Standardize Error Handling**: Touches many call sites, needs comprehensive testing

### High Risk (Future)

- **Targeted Domain Enrichment**: Architectural shift, requires deep domain understanding
- **Service Layer Refactoring**: Major redesign, high test coverage required

### Mitigation

- OK: Pre-MVP status allows breaking changes
- OK: DB reset planned post-MVP
- OK: Comprehensive test suite exists
- OK: Gradual rollout: P0 -> P1 -> P2
- OK: Feature freeze during Phase 1

---

## Appendix: Quantitative Analysis

### File Size Distribution

| File | Lines | Category |
|------|-------|----------|
| ai_core/services/__init__.py | 92 | OK (re-exports) |
| ai_core/services/graph_executor.py | 510 | Large |
| ai_core/services/document_upload.py | 427 | Large |
| theme/views.py | 478 | Large |
| theme/views_web_search.py | 492 | Large |
| ai_core/graphs/technical/retrieval_augmented_generation.py | 254 | Medium |
| ai_core/tools/framework_contracts.py | 255 | Anemic Models |

### Function Complexity

| Function | Lines | Cyclomatic Complexity | Category |
|----------|-------|----------------------|----------|
| execute_graph() | 455 | ~25 | God Function |
| handle_document_upload() | 456 | ~30 | God Function |
| start_ingestion_run() | 133 | ~15 | Large Function |

### Anti-Pattern Distribution

```
Glue Code Inflation:       ###########--------- 55%
Normalizer Syndrome:       ################---- 80%
Anemic Domain Model:       ###################- 95%
Boilerplate Hallucination: #####--------------- 25%
Fragmented Logic:          ############-------- 60%
```

---

## Conclusion

The NOESIS-2 codebase exhibits classic "vibe coding" patterns:
- **Layers of abstraction** that feel professional but add ceremony without value
- **Copy-paste normalization** instead of shared abstractions
- **Anemic models + fat services** violating OOP principles
- **Inconsistent error handling** across layers
- **Manual context threading** instead of typed objects

**Root Cause**: Likely developed prompt-by-prompt without holistic architectural vision.

**Prognosis**: **Excellent foundation** (Pydantic, Type Hints, LangGraph) makes refactoring straightforward. Estimated **20-30% code reduction** without functionality loss.

**Recommendation**: Execute P0 tasks immediately (pre-MVP window), then iterate through P1/P2 as capacity allows.

---

**Report Author**: Claude Sonnet 4.5
**Analysis Date**: 2025-12-31
**Backlog Integration**: [roadmap/backlog.md](../../roadmap/backlog.md#code-quality--architecture-cleanup-pre-mvp-refactoring)
