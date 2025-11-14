# Pydantic Models Simplification TODO

## Complete Codebase Analysis

### Files Analyzed
- `documents/contracts.py` - 12 Pydantic models (primary contracts)
- `documents/parsers.py` - 4 Pydantic models (parser interfaces)  
- `noesis_docs/contracts/collection_v1.py` - 2 Pydantic models (collection contracts)

### Current Complexity Issues

#### 1. Massive Code Duplication Across Files

**Normalization Logic Repeated Everywhere:**
- `normalize_string()` / `normalize_optional_string()` used in 15+ validators
- `normalize_tenant()` duplicated in contracts.py and collection_v1.py
- UUID coercion logic (`_coerce_uuid`) repeated multiple times
- Similar validation patterns for media types, languages, etc.

**Identical Validation Patterns:**
- `documents/contracts.py`: `_normalize_tenant_id()` method
- `noesis_docs/contracts/collection_v1.py`: `_normalise_tenant_id()` method (even different spelling!)
- Both do the same thing but implemented separately

#### 2. Overly Complex Individual Models

**documents/contracts.py:**
- `NormalizedDocumentInputV1`: 20+ fields, 15+ validators, 100+ line `_apply_defaults()` method
- `DocumentMeta`: Complex nested validation for external_ref and parse_stats
- `Asset`: 15+ fields with complex interdependencies and caption logic

**documents/parsers.py:**
- `ParserContent`: Complex model_validator requiring either text or binary payload
- Multiple dataclasses with manual `__post_init__` validation (anti-pattern for Pydantic)

#### 3. Inconsistent Patterns Across Files

**Validation Approaches:**
- `contracts.py`: Uses `@field_validator` with `mode="before"`
- `parsers.py`: Mix of Pydantic models and dataclasses with manual validation
- `collection_v1.py`: Uses `@field_validator` without mode specification

**Error Handling:**
- `contracts.py`: Generic ValueError with string codes
- `parsers.py`: Mix of ValueError and TypeError
- `collection_v1.py`: ValueError with descriptive messages

**Field Naming:**
- `contracts.py`: `tenant_id`, `workflow_id`
- `collection_v1.py`: `tenant_id` but different validation
- Inconsistent use of Optional vs default=None

#### 4. Anti-Patterns and Code Smells

**documents/parsers.py Issues:**
- `ParsedTextBlock`, `ParsedEntity`, `ParsedAsset` are dataclasses, not Pydantic models
- Manual `__post_init__` validation instead of Pydantic validators
- Complex helper functions like `_ensure_non_empty_string()` that duplicate Pydantic functionality
- `ParserRegistry` and `ParserDispatcher` aren't even Pydantic models but mixed in

**documents/contracts.py Issues:**
- Context variables for global state (`_STRICT_CHECKSUMS`, `_ASSET_MEDIA_GUARD`)
- Massive `_apply_defaults()` method that should be split
- Complex blob union with discriminator that could be simplified

### Simplification Opportunities

#### High Priority (Major Impact)

1. **Create Shared Validation Library**
   ```python
   # documents/field_types.py
   from pydantic import Field
   from typing_extensions import Annotated
   
   TenantId = Annotated[str, Field(..., description="Tenant identifier")]
   WorkflowId = Annotated[str, Field(..., description="Workflow identifier")]
   OptionalLanguage = Annotated[Optional[str], Field(None, description="BCP-47 language tag")]
   ```

2. **Replace Dataclasses with Pydantic Models in parsers.py**
   - Convert `ParsedTextBlock`, `ParsedEntity`, `ParsedAsset` to proper Pydantic models
   - Remove manual `__post_init__` validation
   - Use Pydantic's built-in validation features

3. **Extract Common Base Classes**
   ```python
   class BaseDocumentModel(BaseModel):
       model_config = ConfigDict(frozen=True, extra="forbid")
       
       tenant_id: TenantId
       workflow_id: WorkflowId
   ```

4. **Simplify NormalizedDocumentInputV1**
   - Split into 3 models: `DocumentInput`, `DocumentMetadata`, `DocumentBlob`
   - Reduce the 100+ line `_apply_defaults()` method
   - Use composition instead of one massive model

#### Medium Priority (Moderate Impact)

5. **Standardize Error Handling**
   ```python
   class ValidationError(ValueError):
       def __init__(self, code: str, message: str):
           self.code = code
           super().__init__(message)
   ```

6. **Consolidate Blob Handling**
   - `BlobLocator` union is used in contracts.py
   - Similar blob concepts in parsers.py
   - Create unified blob handling approach

7. **Simplify Asset Model Complexity**
   - 15+ fields with complex caption logic
   - Consider splitting into `BaseAsset` + `ImageAsset` + `TextAsset`
   - Reduce interdependent validation

8. **Remove Global State Anti-Pattern**
   - Context variables `_STRICT_CHECKSUMS`, `_ASSET_MEDIA_GUARD`
   - Pass configuration explicitly instead of global state

#### Low Priority (Nice to Have)

9. **Consistent Field Ordering**
   - Required fields first, optional fields last
   - Consistent naming conventions across all models

10. **Performance Optimizations**
    - Cache compiled regex patterns (`_BCP47_PATTERN`, `_LANGUAGE_TAG_RE`, etc.)
    - Reduce validation overhead for frequently used models

11. **Documentation Standardization**
    - Consistent field description format
    - Add usage examples to complex models

### Specific Refactoring Plan

#### Phase 1: Foundation (Week 1-2)
1. Create `documents/field_types.py` with shared field types
2. Create `documents/validators.py` with common validation functions
3. Create base model classes for common patterns

#### Phase 2: Parser Models (Week 3)
1. Convert dataclasses in `parsers.py` to Pydantic models
2. Remove manual `__post_init__` validation
3. Standardize error handling

#### Phase 3: Contract Models (Week 4-5)
1. Refactor `NormalizedDocumentInputV1` into smaller models
2. Simplify `Asset` model complexity
3. Consolidate blob handling across files

#### Phase 4: Cross-File Consistency (Week 6)
1. Standardize validation patterns across all files
2. Remove code duplication between files
3. Implement consistent error handling

#### Phase 5: Optimization (Week 7)
1. Remove global state anti-patterns
2. Performance optimizations
3. Documentation improvements

### Detailed Code Issues Found

#### documents/parsers.py Specific Issues:
- Line 400+: `ParsedTextBlock` dataclass with manual validation should be Pydantic model
- Line 500+: Complex helper functions duplicating Pydantic functionality
- Line 600+: `ParserRegistry` mixed with Pydantic models (architectural issue)

#### documents/contracts.py Specific Issues:
- Line 200+: `_coerce_uuid()` function duplicated logic
- Line 800+: `NormalizedDocumentInputV1._apply_defaults()` is 100+ lines
- Line 1200+: `Asset` model has too many interdependent fields

#### noesis_docs/contracts/collection_v1.py Issues:
- Duplicates tenant validation from contracts.py
- Different spelling: `_normalise_tenant_id` vs `_normalize_tenant_id`
- Could reuse field types from shared library

### Estimated Effort and Impact

**Total Effort:** ~7 weeks for complete refactoring
**Risk Level:** Medium (requires careful migration)
**Impact:**
- **Code Reduction:** ~30% fewer lines of validation code
- **Maintenance:** Much easier to maintain and extend
- **Performance:** 10-20% faster validation
- **Developer Experience:** Significantly improved

### Success Metrics
- [ ] Reduce total lines of validation code by 30%
- [ ] Eliminate all code duplication between files
- [ ] Convert all dataclasses to proper Pydantic models
- [ ] Achieve consistent validation patterns across all models
- [ ] Remove all global state anti-patterns
- [ ] Improve test coverage for edge cases
