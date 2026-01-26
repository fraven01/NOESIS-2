# Roadmap: SOTA Developer RAG Chat (Pre-MVP)

**Objective**: Elevate the current "Developer RAG Chat" (`/rag-tools/`) from a simple prototype to a State-of-the-Art (SOTA) engineering tool. This involves structured reasoning (CoT), strict typed outputs, and rich debugging visualization.

**Status**: Planned
**Effort Estimate**: ~3-4 Sprints (Medium-High Complexity)
**Target Components**:

- `ai_core/nodes/compose.py`
- `theme/views_chat.py`
- `theme/templates/theme/partials/tool_chat.html`
- `theme/templates/theme/partials/chat_message.html`

---

## 1. Architectural Philosophy (The "Glass Box" Approach)

Since this is a *Developer Workbench* tool (DEBUG-only), we prioritize **transparency** over "magic". The chat should not just give an answer, but explain *how* it got there.

- **Structured Intelligence**: The LLM never returns raw strings. It returns a `RagResponse` object.
- **Chain-of-Thought (CoT)**: We force the model to "think" before answering to improve accuracy and allow devs to debug reasoning errors (dev-only visibility).
- **Rich Metadata**: Tokens, latency, model usage, and exact source relevance scores are first-class citizens in the UI.

---

## 2. Current State Analysis

### 2.1 Backend (`ai_core/nodes/compose.py`)

**Already exists:**

- `ComposeInput` / `ComposeOutput` Pydantic models
- `client.call()` with `response_format` parameter support (line 326, 386-388)
- PII masking via `mask_prompt()` / `mask_response()`
- Langfuse observability via `@observe_span`

**Gaps:**

- No CoT reasoning structure
- Output is plain string (`answer: str | None`)
- No source-level relevance scoring from LLM

### 2.2 Prompt (`ai_core/prompts/retriever/answer.v1.md`)

Current prompt already defines:

```
Ausgabe:
- answer
- citations[]
- gaps[]
```

**Gap**: No explicit JSON schema enforcement, no CoT analysis step.

### 2.3 Frontend (`theme/views_chat.py`, `tool_chat.html`)

**Already exists:**

- HTMX form with `hx-post` to `chat-submit`
- Thread management via session
- Collection/Case scope selection

**Gaps:**

- No debug metadata display (latency, tokens, model)
- No CoT reasoning visibility
- No collapsible sections
- Basic snippet display without relevance bars
- **Complexity Risk**: Integrating interactive UI (collapsibles, streaming updates) into a server-side rendered Django/HTMX implementation requires careful state management with Alpine.js to avoid "tag soup" or race conditions.

### 2.4 LLM Client (`ai_core/llm/client.py`)

**Already available in response:**

- `latency_ms`, `usage` (prompt/completion tokens), `cost_usd`
- `model`, `cache_hit`, `finish_reason`
- Streaming support via `call_stream()`

**Note**: All metadata is already tracked but not surfaced to UI.

---

## 3. Implementation Plan

### Phase 1: Schema Definition (Effort: S - 0.5 Sprint)

**Task 1.1**: Define `RagResponse` schema in `ai_core/rag/schemas.py`

```python
from pydantic import BaseModel, Field

class SourceRef(BaseModel):
    """Reference to a source snippet with LLM-assessed relevance."""
    id: str
    label: str
    relevance_score: float = Field(ge=0.0, le=1.0, description="LLM's self-assessed confidence")

class RagReasoning(BaseModel):
    """Chain-of-thought reasoning structure."""
    analysis: str = Field(description="How snippets relate to the question")
    gaps: list[str] = Field(default_factory=list, description="Missing information identified")

class RagResponse(BaseModel):
    """Structured RAG response with reasoning and metadata."""
    reasoning: RagReasoning
    answer_markdown: str = Field(description="Final user-facing answer in Markdown")
    used_sources: list[SourceRef] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list, max_length=3)
```

**Acceptance Criteria:**

- [ ] Schema passes `model_json_schema()` export
- [ ] Unit tests for round-trip serialization
- [ ] Schema documented in `ai_core/rag/README.md`

### Phase 2: Prompt Engineering (Effort: S - 0.5 Sprint)

**Task 2.1**: Create `ai_core/prompts/retriever/answer.v2.md`

```markdown
# RAG Answer Generation (v2)

## Role
You are an expert RAG assistant analyzing documents to answer user questions.

## Input
- **Question**: The user's query
- **Context**: Retrieved snippets in YAML format with [label] prefixes

## Process (Chain-of-Thought)
1. **Analyze**: For each snippet, assess relevance to the question (0.0-1.0)
2. **Identify Gaps**: Note what information is missing
3. **Synthesize**: Compose answer using only provided context

## Output Format (JSON)
Return ONLY valid JSON matching this schema:
```json
{
  "reasoning": {
    "analysis": "Brief analysis of how snippets address the question",
    "gaps": ["List of missing information"]
  },
  "answer_markdown": "The final answer in Markdown format",
  "used_sources": [
    {"id": "snippet-id", "label": "Source Label", "relevance_score": 0.85}
  ],
  "suggested_followups": ["Optional follow-up questions"]
}
```

## Rules

- Use ONLY information from provided snippets
- Cite sources using [label] notation in answer_markdown
- If context is insufficient, state this clearly and populate gaps[]
- Never hallucinate or invent information

```

**Acceptance Criteria:**
- [ ] Prompt produces valid JSON for 95%+ test cases
- [ ] Few-shot examples added for edge cases
- [ ] Prompt version tracked in `ai_core/infra/prompts.py`
- [ ] Streaming compatibility verified (see Phase 3)

### Phase 3: Backend Refactoring & Streaming (Effort: L - 1.5 Sprints)

**Challenge**: The current implementation uses `client.call_stream`. Standard `json_object` mode often buffers the response or makes partial parsing difficult.

**Strategy**: Prefer JSON-only output for consistency with the app. Streaming may be limited or buffered; do not introduce XML/tag streaming without explicit approval.

**Task 3.1**: Update `ComposeOutput` model

```python
class ComposeOutput(BaseModel):
    """Structured output payload returned by the compose node."""
    answer: str | None
    prompt_version: str | None
    snippets: list[Mapping[str, Any]] | None = None
    retrieval: Mapping[str, Any] | None = None
    # New fields
    reasoning: RagReasoning | None = None
    used_sources: list[SourceRef] | None = None
    suggested_followups: list[str] | None = None
    # For v2, prefer answer_markdown from RagResponse; answer remains for v1 compatibility
    # Debug metadata
    debug_meta: dict[str, Any] | None = None  # latency, tokens, model, cost
```

**Task 3.2**: Update `_run_stream()` in `compose.py`

1. Update prompt to produce strict JSON per `RagResponse` (no tags).
2. Parse JSON in `_run` (non-streamed) or buffer until complete when streaming; do not add tag parsing without explicit approval.

**Task 3.3**: Update `RagQueryService.execute()` to return full payload

Ensure `theme/views_chat.py` receives all new fields from compose output.

**Acceptance Criteria:**

- [ ] v2 path replaces v1 (no feature flag)
- [ ] Streaming is NOT broken; users see tokens as they appear
- [ ] Graceful fallback on parse failure
- [ ] All new fields propagated to view layer
- [ ] Integration tests cover both v1 and v2 paths
- [ ] Langfuse spans include `prompt_version: v2`

### Phase 4: Frontend Implementation (Effort: M-L - 1.5 Sprints)

**Task 4.1**: Create `chat_message_debug.html` partial

Structure:

```
┌─────────────────────────────────────────────┐
│ [Final Answer]  [Thinking ▼]  [Sources ▼]   │  <- Tabs/Toggles
├─────────────────────────────────────────────┤
│ Main answer in Markdown                      │
│ with clickable citations [1] [2]             │
├─────────────────────────────────────────────┤
│ ▼ Thinking Process (collapsed)               │
│   "User asks about X. Snippet A mentions..." │
│   Gaps: [missing info 1], [missing info 2]   │
├─────────────────────────────────────────────┤
│ ▼ Sources & Evidence (collapsed)             │
│   [1] Source Label  ████████░░ 85%           │
│   [2] Another Doc   ██████░░░░ 62%           │
│   [Copy Snippet] [View Document]             │
├─────────────────────────────────────────────┤
│ Debug: 1.2s | 847 tokens | gemini-1.5-pro   │  <- Footer (staff only)
└─────────────────────────────────────────────┘
```

**Task 4.2**: Update `chat_message.html` to include new partial conditionally

**Task 4.3**: Add Alpine.js toggle logic for collapsible sections

**Task 4.4**: Style relevance score bars with Tailwind

**Task 4.5**: Update `chat_submit()` view to pass full structured response

**Acceptance Criteria:**

- [ ] Final Answer tab is default view
- [ ] Thinking/Sources sections collapse properly
- [ ] Relevance bars render correctly (0-100%)
- [ ] Debug footer visible only for `is_staff` or `DEBUG=True` (no debug metadata leakage in non-debug views)
- [ ] Suggested follow-ups rendered as clickable chips
- [ ] Responsive on mobile viewports

### Phase 5: Testing & Documentation (Effort: M - 1 Sprint)

**Task 5.1**: Unit tests

- `ai_core/tests/nodes/test_compose_v2.py` - JSON parsing, fallback
- `ai_core/tests/rag/test_schemas.py` - Schema validation

**Task 5.2**: Integration tests

- `theme/tests/test_chat_submit.py` - Full request/response cycle checks
- `theme/tests/test_rag_tools_view.py` - Ensure tool view doesn't break
- Test with malformed JSON responses (fallback path)

**Task 5.3**: Update documentation

- `docs/development/rag-tools-workbench.md` - Document new UI features
- `ai_core/rag/README.md` - Document `RagResponse` schema

**Task 5.4**: Migration Plan

- **Legacy Compatibility**: v1 prompts will remain as fallback.
- **Data Migration**: Since chat history is session-based (ephemeral in this view), no DB migration is required for the *chat history*.
- **User Rollout**:
    1. Deploy code with Feature Flag OFF.
    2. Enable for Admin/Staff users first.
    3. Verify latency impact.
    4. Enable globally.

**Acceptance Criteria:**

- [ ] Test coverage >= 80% for new code
- [ ] E2E test with Playwright for UI interactions
- [ ] Documentation reviewed and merged
- [ ] Migration of existing "saved" chats (if any) validated (checking `rag.chat_history` logic if applicable)

---

## 4. Effort Summary

| Phase | Description | Size | Dependencies |
|-------|-------------|------|--------------|
| 1 | Schema Definition | S (0.5 Sprint) | None |
| 2 | Prompt Engineering | S (0.5 Sprint) | Phase 1 |
| 3 | Backend Refactoring | M (1 Sprint) | Phase 1, 2 |
| 4 | Frontend Implementation | M-L (1.5 Sprints) | Phase 3 |
| 5 | Testing & Documentation | S (0.5 Sprint) | Phase 3, 4 |

**Total**: ~3-4 Sprints (6-8 weeks at 2-week sprints)

---

## 5. Feature Flags & Rollout

1. **Development**: Replace the existing chat with v2 (DEBUG-only)
2. **Staging/Production**: Not applicable unless explicitly enabled later

Fallback: If v2 JSON parsing fails, return v1 plain-text response with warning in debug footer (DEBUG/staff only).

---

## 6. Future "SOTA" Features (Post-Implementation)

- **Interactive Citations**: Hovering a citation `[1]` in the answer highlights the snippet in the sidebar.
- **Feedback Loop**: Thumbs up/down buttons that log to `LLM Feedback` dataset for future fine-tuning.
- **Prompt Playground**: A "Retry with new prompt" button right in the chat bubble to test prompt variations on the fly.
- **Streaming CoT**: Show reasoning in real-time as it streams in.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM fails to produce valid JSON | Graceful fallback to v1; log failures for prompt iteration |
| Latency increase from CoT | Monitor via Langfuse; accept buffered JSON if streaming is constrained |
| Frontend complexity (Alpine+HTMX) | Incremental rollout; keep state logic in dedicated `chat_logic.js` file if it grows too large. Use standard Tailwind components. |
| Breaking existing workflows | Replace v1 in DEBUG-only dev workbench; no feature flag |
| Streaming "Jumpiness" | Ensure the UI smooths out DOM updates when switching from "Thinking" to "Answer" |

---

## 8. Related Backlog Items

- `backlog.md#INC-20260119-001`: Async-First Web Search (pattern applicable here)
- `backlog.md#Collection Search strategy quality improvements`: Similar JSON output approach

---

**Version**: 2.0 (Revised)
**Last Updated**: 2026-01-19
**Author**: Claude Code
**Reviewers**: TBD
