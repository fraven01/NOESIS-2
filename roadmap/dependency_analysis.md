# Dependency Analysis: Agentic Rebuild Roadmap

This document maps dependencies between work items to enable parallel execution and identify the critical path.

## Work Item Index

| ID | Phase | Title | Short Name |
|----|-------|-------|------------|
| P1.1 | 1 | AgentState, AgentRunRecord, and decision log contracts | AgentState |
| P1.2 | 1 | AgentRuntime lifecycle and run handles | RuntimeLifecycle |
| P2.1 | 2 | AgentRuntime entrypoint hardening (start/resume) | EntrypointHarden |
| P2.2 | 2 | Mandatory dev case and fixed workflow taxonomy | DevCaseTaxonomy |
| P2.3 | 2 | Dependency injection for routers and caches | DependencyInjection |
| P2.4 | 2 | Async run tracking and HITL interrupts | AsyncHITL |
| P3.1 | 3 | Capability descriptor and registry bridge | CapabilityRegistry |
| P3.2 | 3 | New bounded retrieval/search capability | BoundedSearch |
| P3.3 | 3 | RetrievalCapability (bounded) | RetrievalCap |
| P3.4 | 3 | ComposeAnswerCapability (bounded) | ComposeCap |
| P3.5 | 3 | Capability configuration contract (Plan + Validate) | CapabilityConfig |
| P3.6 | 3 | Document storage configuration contract (Plan + Validate) | DocStorageConfig |
| P3.7 | 3 | Crawler configuration contract (Plan + Validate) | CrawlerConfig |
| P3.8 | 3 | WebSearch configuration contract (Plan + Validate) | WebSearchConfig |
| P4.1 | 4 | Minimal dev runner for AgentRuntime flows | DevRunner |
| P4.2 | 4 | Golden query set and diff harness | GoldenHarness |
| P4.3 | 4 | Collection search runtime flow | CollectionFlow |
| P4.4 | 4 | RAG runtime flow | RAGFlow |
| P4.5 | 4 | Framework analysis runtime flow | FrameworkFlow |
| P5.1 | 5 | Theme to runtime adapter (thin, optional) | ThemeAdapter |
| P5.2 | 5 | RagQueryService routing through runtime flow | RagServiceRoute |
| P6.1 | 6 | Deprecate legacy orchestration graphs | DeprecateLegacy |

---

## Dependency Matrix

```
Legend:
  ● = Direct dependency (blocks start)
  ○ = Soft dependency (benefits from, not blocking)

                          DEPENDS ON →
                    P1.1 P1.2 P2.1 P2.2 P2.3 P2.4 P3.1 P3.2 P3.3 P3.4 P3.5 P3.6 P3.7 P3.8 P4.1 P4.2 P4.3 P4.4 P4.5
                    ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ──── ────
P1.1 AgentState      -
P1.2 RuntimeLifecycle●    -
P2.1 EntrypointHarden●    ●    -
P2.2 DevCaseTaxonomy      ○    ●    -
P2.3 DependencyInject●    ●    ○         -
P2.4 AsyncHITL            ●    ○              -
P3.1 CapabilityReg        ●                        -
P3.2 BoundedSearch        ○                        ●    -
P3.3 RetrievalCap         ○                        ●         -
P3.4 ComposeCap           ○                        ●              -
P3.5 CapabilityConfig               ●              ●         ●    ●    -
P3.6 DocStorageConfig               ●              ●                   ●    -
P3.7 CrawlerConfig                  ●              ●                   ●         -
P3.8 WebSearchConfig                ●              ●                   ●              -
P4.1 DevRunner            ●    ●    ●                                                      -
P4.2 GoldenHarness                                                                         ●    -
P4.3 CollectionFlow            ○         ○    ●    ○    ●                   ○    ○    ○    ●    ○    -
P4.4 RAGFlow               ○         ○    ●         ●         ●    ●    ●                   ●    ○         -
P4.5 FrameworkFlow              ○         ○    ●    ●                                       ●    ○              -
P5.1 ThemeAdapter         ●    ●
P5.2 RagServiceRoute                                                                                   ●
P6.1 DeprecateLegacy                                                                              ●    ●    ●    ●
```

---

## Detailed Dependency Graph

### Phase 1: Foundation Layer (Serial)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: FOUNDATION                                                         │
│                                                                             │
│   ┌──────────────┐          ┌───────────────────┐                          │
│   │    P1.1      │          │       P1.2        │                          │
│   │  AgentState  │─────────▶│  RuntimeLifecycle │                          │
│   │  AgentRun    │          │  (start/resume/   │                          │
│   │  DecisionLog │          │   cancel/status)  │                          │
│   └──────────────┘          └───────────────────┘                          │
│                                      │                                      │
│                                      │ UNLOCKS PHASE 2 + 3                  │
│                                      ▼                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Rationale**: P1.1 defines the data contracts (AgentState, AgentRunRecord, DecisionLog) that P1.2 consumes for lifecycle management. These are strictly serial.

---

### Phase 2: Hardening Layer (Partially Parallel)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: HARDENING                                                          │
│                                                                             │
│   From P1.2                                                                 │
│       │                                                                     │
│       ├────────────────────┬────────────────────┬──────────────────────┐   │
│       ▼                    ▼                    ▼                       │   │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐                  │   │
│  │   P2.1   │        │   P2.3   │        │   P2.4   │                  │   │
│  │ Entrypoint│        │   DI     │        │  Async   │                  │   │
│  │ Hardening │        │ Routers  │        │  HITL    │                  │   │
│  └────┬─────┘        └──────────┘        └──────────┘                  │   │
│       │                    │                    │                       │   │
│       │              ┌─────┴────────────────────┘                       │   │
│       ▼              │  (soft deps for tenant isolation tests)          │   │
│  ┌──────────┐        │                                                  │   │
│  │   P2.2   │◀───────┘                                                  │   │
│  │ DevCase  │                                                           │   │
│  │ Taxonomy │                                                           │   │
│  └──────────┘                                                           │   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Parallel Opportunities**:
- P2.1, P2.3, P2.4 can start in parallel after P1.2
- P2.2 requires P2.1 (validation rules depend on hardened entrypoints)

**Dependencies Explained**:
| Item | Depends On | Reason |
|------|-----------|--------|
| P2.1 | P1.1, P1.2 | Hardens `start/resume` APIs defined in P1.2, uses AgentState from P1.1 |
| P2.2 | P2.1 | Validation rules for `case_id`/`workflow_id` enforce hardened entrypoint contracts |
| P2.3 | P1.1, P1.2 | Injects dependencies via runtime capability descriptors |
| P2.4 | P1.2 | Extends runtime lifecycle with cancel tokens and interrupt hooks |

---

### Phase 3: Capability Layer (Parallel After P3.1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: CAPABILITIES                                                       │
│                                                                             │
│   From P1.2                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────┐                                                          │
│  │     P3.1     │                                                          │
│  │  Capability  │                                                          │
│  │   Registry   │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         ├─────────────────┬─────────────────┐                              │
│         ▼                 ▼                 ▼                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │     P3.2     │  │     P3.3     │  │     P3.4     │                      │
│  │   Bounded    │  │  Retrieval   │  │   Compose    │                      │
│  │    Search    │  │  Capability  │  │  Capability  │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│         │                 │                 │                              │
│         │                 └────────┬────────┘                              │
│         │                          │                                        │
│         │                          ▼                                        │
│         │                   ┌──────────────┐                                │
│         │                   │     P3.5     │◀─── P2.3 (DI)                  │
│         │                   │  RAG Config  │                                │
│         │                   │ Plan+Validate│                                │
│         │                   └──────┬───────┘                                │
│         │                          │                                        │
│         │                          ▼                                        │
│         │                   ┌──────────────┐                                │
│         │                   │     P3.6     │◀─── P2.3 (DI)                  │
│         │                   │  Doc Storage │                                │
│         │                   │ Plan+Validate│                                │
│         │                   └──────┬───────┘                                │
│         │                          │                                        │
│         │                          ▼                                        │
│         │                   ┌──────────────┐                                │
│         │                   │     P3.7     │◀─── P2.3 (DI)                  │
│         │                   │   Crawler    │                                │
│         │                   │ Plan+Validate│                                │
│         │                   └──────┬───────┘                                │
│         │                          │                                        │
│         │                          ▼                                        │
│         │                   ┌──────────────┐                                │
│         │                   │     P3.8     │◀─── P2.3 (DI)                  │
│         │                   │  WebSearch   │                                │
│         │                   │ Plan+Validate│                                │
│         │                   └──────────────┘                                │
│         │                          │                                        │
│         ▼                          ▼                                        │
│    CollectionFlow              RAGFlow                                      │
│       (P4.3)                    (P4.4)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Parallel Opportunities**:
- P3.2, P3.3, P3.4 can all run in parallel after P3.1 completes
- P3.5 depends on P3.3, P3.4 (capabilities it configures) and P2.3 (DI for guardrails)
- P3.6 depends on P3.5 (reuses Plan+Validate pattern) and P2.3 (DI for DocumentsRepository)
- P3.7 depends on P3.5 (reuses Plan+Validate pattern) and P2.3 (DI for CrawlerManager)
- P3.8 depends on P3.5 (reuses Plan+Validate pattern) and P2.3 (DI for WebSearchWorker)
- P3.6, P3.7, P3.8 can run in parallel (all follow P3.5 pattern)

**Dependencies Explained**:
| Item | Depends On | Reason |
|------|-----------|--------|
| P3.1 | P1.2 | Registry bridge references runtime resolution APIs |
| P3.2 | P3.1 | Bounded search registers via capability registry |
| P3.3 | P3.1 | RetrievalCapability needs registry bridge for io_spec binding |
| P3.4 | P3.1 | ComposeAnswerCapability needs registry bridge for io_spec binding |
| P3.5 | P3.1, P3.3, P3.4, P2.3 | Configures RAG capabilities; needs DI for GuardrailLimits/QuotaService |
| P3.6 | P3.1, P3.5, P2.3 | Follows P3.5 pattern; needs DI for DocumentsRepository, LifecycleStore |
| P3.7 | P3.1, P3.5, P2.3 | Follows P3.5 pattern; needs DI for CrawlerManager, CrawlerGuardrailLimits |
| P3.8 | P3.1, P3.5, P2.3 | Follows P3.5 pattern; needs DI for WebSearchWorker, WebSearchGuardrailLimits |

---

### Phase 4: Integration Layer (Mixed Dependencies)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: INTEGRATION                                                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ CRITICAL PATH: Dev Runner → Golden Harness                            │  │
│  │                                                                        │  │
│  │   P1.2 ──┐                                                            │  │
│  │   P2.1 ──┼──▶ ┌──────────┐      ┌──────────────┐                     │  │
│  │   P2.2 ──┘    │   P4.1   │─────▶│     P4.2     │                     │  │
│  │               │ DevRunner │      │ GoldenHarness│                     │  │
│  │               └──────────┘      └──────┬───────┘                     │  │
│  │                    │                   │                              │  │
│  └────────────────────┼───────────────────┼──────────────────────────────┘  │
│                       │                   │                                  │
│  ┌────────────────────┼───────────────────┼──────────────────────────────┐  │
│  │ RUNTIME FLOWS      │                   │ (soft dep: harness validates) │  │
│  │                    ▼                   ▼                               │  │
│  │              ┌──────────┐        ┌──────────┐        ┌──────────┐     │  │
│  │   P3.2 ────▶│   P4.3   │        │   P4.4   │◀────── │   P4.5   │     │  │
│  │   P2.4 ────▶│Collection│        │   RAG    │  P3.1──▶│Framework │     │  │
│  │              │   Flow   │        │   Flow   │  P2.4──▶│   Flow   │     │  │
│  │              └──────────┘        └──────────┘        └──────────┘     │  │
│  │                                       ▲                               │  │
│  │                                       │                               │  │
│  │                              P3.3 ────┤                               │  │
│  │                              P3.4 ────┘                               │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Parallel Opportunities**:
- P4.3, P4.4, P4.5 can run in parallel once their respective capability dependencies are met
- P4.2 depends only on P4.1, not on runtime flows

**Dependencies Explained**:
| Item | Depends On | Reason |
|------|-----------|--------|
| P4.1 | P1.2, P2.1, P2.2 | Dev runner needs runtime APIs, hardened entrypoints, and dev case fixture |
| P4.2 | P4.1 | Harness executes via dev runner |
| P4.3 | P3.2, P2.4, P4.1 | Uses bounded search capability, needs HITL interrupts, validated via dev runner |
| P4.4 | P3.3, P3.4, P2.4, P4.1 | Uses retrieval + compose capabilities, needs HITL, validated via dev runner |
| P4.5 | P3.1, P2.4, P4.1 | Uses capability registry, needs HITL, validated via dev runner |

---

### Phase 5 & 6: Client Wiring and Deprecation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: OPTIONAL CLIENTS                                                   │
│                                                                             │
│   P1.2 ──┐      ┌──────────────┐                                           │
│   P2.1 ──┴─────▶│     P5.1     │                                           │
│                 │ ThemeAdapter │                                           │
│                 └──────────────┘                                           │
│                                                                             │
│   P4.4 ────────▶┌──────────────┐                                           │
│                 │     P5.2     │                                           │
│                 │RagServiceRoute│                                           │
│                 └──────────────┘                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: DEPRECATION                                                        │
│                                                                             │
│   P4.2 ──┐                                                                  │
│   P4.3 ──┼─────▶┌──────────────┐                                           │
│   P4.4 ──┤      │     P6.1     │                                           │
│   P4.5 ──┘      │DeprecateLegacy│                                           │
│                 └──────────────┘                                           │
│                                                                             │
│   Gate: Golden harness green for N >= 20 runs                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Path Analysis

The **critical path** determines minimum project duration. Items on the critical path cannot be delayed without delaying the entire project.

```
CRITICAL PATH (longest dependency chain):

P1.1 → P1.2 → P2.1 → P4.1 → P4.2 → P6.1
  │            │
  │            └──→ P2.2 (parallel with P4.1 prep)
  │
  └──→ P3.1 → P3.3 ─┐
              P3.4 ─┴─→ P4.4 → P5.2
                              ↓
                            P6.1

Simplified:
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│P1.1 │──▶│P1.2 │──▶│P2.1 │──▶│P4.1 │──▶│P4.2 │──▶│P6.1 │
└─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘
   1         2         3         4         5         6
                              (+ parallel capability work)
```

**Critical Path Length**: 6 sequential work items minimum

---

## Parallelization Strategy

### Maximum Parallelization Schedule

| Stage | Items | Prerequisites |
|-------|-------|---------------|
| **Stage 1** | P1.1 | None |
| **Stage 2** | P1.2 | P1.1 |
| **Stage 3** | P2.1, P2.3, P2.4, P3.1 | P1.2 |
| **Stage 4** | P2.2, P3.2, P3.3, P3.4, P5.1 | P2.1 (for P2.2), P3.1 (for P3.x), P2.1 (for P5.1) |
| **Stage 5** | P3.5, P4.1 | P3.3, P3.4, P2.3 (for P3.5); P2.1, P2.2 (for P4.1) |
| **Stage 6** | P3.6, P3.7, P3.8, P4.2, P4.3, P4.4, P4.5 | P3.5 (for P3.6, P3.7, P3.8); P4.1, respective capabilities |
| **Stage 7** | P5.2 | P4.4 |
| **Stage 8** | P6.1 | P4.2 green, P4.3, P4.4, P4.5 |

### Visual Schedule (Gantt-style)

```
Stage:  1    2    3              4                   5         6              7    8
        ──── ──── ──────────────  ───────────────────  ────────  ──────────────  ──── ────
P1.1    ████
P1.2         ████
P2.1              ████
P2.2                             ████
P2.3              ████████████
P2.4              ████████████
P3.1              ████
P3.2                             ████████
P3.3                             ████████
P3.4                             ████████
P3.5                                                 ████████
P3.6                                                           ████████
P3.7                                                           ████████
P4.1                                                 ████
P4.2                                                           ████████
P4.3                                                           ████████████
P4.4                                                           ████████████
P4.5                                                           ████████████
P5.1                             ████████████████████████████████████████████
P5.2                                                                        ████
P6.1                                                                             ████
```

---

## Risk Analysis: Blocking Dependencies

### High-Risk Bottlenecks

| Bottleneck | Blocks | Mitigation |
|------------|--------|------------|
| **P1.1 (AgentState)** | Everything | Start immediately; prototype contracts early |
| **P1.2 (RuntimeLifecycle)** | P2.x, P3.1, P4.x, P5.1 | Define API surface early; allow parallel impl of consumers with mocks |
| **P3.1 (CapabilityRegistry)** | P3.2, P3.3, P3.4, P3.5, P3.6, P3.7 | Keep registry bridge minimal; define interface contracts upfront |
| **P4.1 (DevRunner)** | P4.2, P4.3, P4.4, P4.5 | Prioritize minimal working runner; harness can validate iteratively |
| **P4.2 (GoldenHarness)** | P6.1 | Harness is gate for deprecation; ensure artifact schema is stable early |

### Dependency Injection (P2.3) Consideration

The user's selected line (73) references **P2.3: Dependency injection for routers and caches**. This item:
- **Does NOT block** critical path items (P4.1, P4.2, P6.1)
- **DOES enable** tenant isolation testing for P4.3 (CollectionFlow)
- **Can run in parallel** with P2.1, P2.4 after P1.2

**Recommendation**: Start P2.3 in Stage 3 but treat as lower priority than P2.1. It becomes critical only when P4.3 integration tests require tenant isolation validation.

---

## Suggested Implementation Order

### Phase 1 (Foundation) - Serial
1. **P1.1** AgentState, AgentRunRecord, decision log contracts
2. **P1.2** AgentRuntime lifecycle and run handles

### Phase 2 (Hardening) - Parallel Start
3. **P2.1** EntrypointHarden *(critical path)*
4. **P2.3** DependencyInjection *(parallel, lower priority)*
5. **P2.4** AsyncHITL *(parallel)*
6. **P2.2** DevCaseTaxonomy *(after P2.1)*

### Phase 3 (Capabilities) - Parallel
7. **P3.1** CapabilityRegistry *(critical for P3.2-3.7)*
8. **P3.2** BoundedSearch *(parallel after P3.1)*
9. **P3.3** RetrievalCapability *(parallel after P3.1)*
10. **P3.4** ComposeAnswerCapability *(parallel after P3.1)*
11. **P3.5** CapabilityConfig *(after P3.3, P3.4, P2.3 - defines Plan+Validate pattern for RAG)*
12. **P3.6** DocStorageConfig *(after P3.5, P2.3 - applies Plan+Validate pattern to Document Storage)*
13. **P3.7** CrawlerConfig *(after P3.5, P2.3 - applies Plan+Validate pattern to Crawler)*

### Phase 4 (Integration) - Mixed
14. **P4.1** DevRunner *(critical path)*
15. **P4.2** GoldenHarness *(critical path - gates P6.1)*
16. **P4.3** CollectionFlow *(parallel with P4.4, P4.5; can use P3.6/P3.7 for doc storage + crawler)*
17. **P4.4** RAGFlow *(parallel, uses P3.5 config pattern)*
18. **P4.5** FrameworkFlow *(parallel)*

### Phase 5 (Clients) - Optional
19. **P5.1** ThemeAdapter *(can start after P2.1, low priority)*
20. **P5.2** RagServiceRoute *(after P4.4)*

### Phase 6 (Deprecation)
21. **P6.1** DeprecateLegacy *(after harness green for N >= 20 runs)*

---

## Cross-Reference: Review Findings Coverage

| Review Finding | Addressed By |
|---------------|--------------|
| Review 1: No AgentState, success conflated | P1.1, P1.2 |
| Review 1: Compile-time control flow | P1.2, P4.3, P4.4, P4.5 |
| Review 1: HITL blocking, async tracking | P2.4, P4.1, P4.3 |
| Review 1: Global mutable routers | **P2.3** |
| Review 2: Capability-ready graphs | P3.1, P3.2, P3.3, P3.4 |
| Review 2: CollectionSearchAdapter unusable | P3.2, P4.3 |
| Review 2: Agentic smell in RAG graph | P3.3, P3.4, P4.4 |
| Review 3: ID leaks, implicit context | P2.1, P2.2 |
| Review 3: Hidden dependencies | **P2.3**, P2.1, P3.5, P3.6, P3.7 |
| Review 3: Deterministic state shape | P1.1, P4.2, P3.5, P3.6, P3.7 |

---

## Appendix: Dependency Adjacency List

```yaml
P1.1:
  depends_on: []
  enables: [P1.2, P2.1, P2.3]

P1.2:
  depends_on: [P1.1]
  enables: [P2.1, P2.3, P2.4, P3.1, P4.1, P5.1]

P2.1:
  depends_on: [P1.1, P1.2]
  enables: [P2.2, P4.1, P5.1]

P2.2:
  depends_on: [P2.1]
  enables: [P4.1]

P2.3:
  depends_on: [P1.1, P1.2]
  enables: [P3.5, P3.6, P3.7, P4.3]  # P3.5/P3.6/P3.7 need DI for guardrails; P4.3 needs tenant isolation

P2.4:
  depends_on: [P1.2]
  enables: [P4.3, P4.4, P4.5]

P3.1:
  depends_on: [P1.2]
  enables: [P3.2, P3.3, P3.4, P3.5, P4.5]

P3.2:
  depends_on: [P3.1]
  enables: [P4.3]

P3.3:
  depends_on: [P3.1]
  enables: [P3.5, P4.4]

P3.4:
  depends_on: [P3.1]
  enables: [P3.5, P4.4]

P3.5:
  depends_on: [P3.1, P3.3, P3.4, P2.3]
  enables: [P3.6, P3.7, P4.4]  # P3.6/P3.7 follow the pattern; RAGFlow uses the Plan+Validate config pattern

P3.6:
  depends_on: [P3.1, P3.5, P2.3]
  enables: [P4.3]  # CollectionFlow can use document storage capability for ingestion triggering

P3.7:
  depends_on: [P3.1, P3.5, P2.3]
  enables: [P4.3]  # CollectionFlow can use crawler capability for web acquisition

P4.1:
  depends_on: [P1.2, P2.1, P2.2]
  enables: [P4.2, P4.3, P4.4, P4.5]

P4.2:
  depends_on: [P4.1]
  enables: [P6.1]

P4.3:
  depends_on: [P3.2, P2.4, P4.1]  # Soft deps: P3.6, P3.7 for doc storage and crawler capabilities
  enables: [P6.1]

P4.4:
  depends_on: [P3.3, P3.4, P3.5, P2.4, P4.1]
  enables: [P5.2, P6.1]

P4.5:
  depends_on: [P3.1, P2.4, P4.1]
  enables: [P6.1]

P5.1:
  depends_on: [P1.2, P2.1]
  enables: []

P5.2:
  depends_on: [P4.4]
  enables: []

P6.1:
  depends_on: [P4.2, P4.3, P4.4, P4.5]
  enables: []
```

---

**Version**: 1.3
**Created**: 2026-01-26
**Updated**: 2026-01-26 - Added P3.7 (Crawler configuration contract)
**Source**: [agentic_rebuild_deprecate_roadmap.md](agentic_rebuild_deprecate_roadmap.md)
