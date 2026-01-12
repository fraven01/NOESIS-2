# Documentation (`docs/`)

This directory contains explanatory documentation that references implementation in the repository. Runtime behavior, validation, and contracts are defined in code.

LLM entry point and code navigation: `AGENTS.md`.

## Quick navigation

- Getting started / local workflow: `docs/development/onboarding.md` and `README.md`
- Architecture lens + inventory: `docs/architecture/4-layer-firm-hierarchy.md`, `docs/architecture/architecture-reality.md`
- AI Core graphs: `ai_core/graph/README.md`
- AI Core tool envelopes: `docs/agents/tool-contracts.md` (mirrors `ai_core/tool_contracts/base.py`)
- ID normalization & headers: `docs/architecture/id-guide-for-agents.md` (points to `ai_core/ids/*` and `common/constants.py`)
- RAG: `docs/rag/overview.md`, `docs/rag/ingestion.md`, `docs/rag/retrieval-contracts.md`
- Documents subsystem: `docs/documents/` (contracts, parsing, persistence notes)
- Ops/runbooks: `docs/runbooks/`, `docs/cicd/`, `docs/operations/`

## Directory overview

```
docs/
  README.md
  agents/          # Agent + tool docs (explanatory)
  architecture/    # IDs, ADRs, design notes (explanatory)
  documents/       # Document subsystem docs
  rag/             # Retrieval/ingestion docs
  crawler/         # Crawler docs
  development/     # Developer guides
  observability/   # Langfuse, ELK, telemetry docs
  api/             # API reference/changelog docs
  cicd/            # CI/CD pipeline docs
  operations/      # Scaling/deployment docs
  runbooks/        # Operational runbooks
  security/        # Security notes/secrets docs
  domain/          # Domain model notes
  docker/          # Docker/compose notes
  audit/           # Audit reports
```

Legacy alias: `docs/architektur/*` now redirects to `docs/architecture/*`; use `docs/architecture/` going forward.
