# NOESIS 2 Documentation

**Entry Point**: [Root `/AGENTS.md`](../AGENTS.md) - Single Source of Truth for Architecture, IDs, Contracts

---

## Navigation Guide

### For Developers (Getting Started)

1. **Onboarding**: [development/onboarding.md](development/onboarding.md)
2. **Architecture Overview**: [Root `/AGENTS.md`](../AGENTS.md) (4-Layer Model, ID Semantics, Tool Contracts)
3. **Development Setup**: [Root `/CLAUDE.md`](../CLAUDE.md) or [Root `/Gemini.md`](../Gemini.md) (Operational Commands)

### For Architects

1. **4-Layer Architecture**: [Root `/AGENTS.md`](../AGENTS.md)
2. **System Overview**: [architektur/overview.md](architektur/overview.md)
3. **ID Semantics & Propagation**:
   - [architecture/id-semantics.md](architecture/id-semantics.md) (Definitions)
   - [architecture/id-propagation.md](architecture/id-propagation.md) (End-to-End Flows)
   - [architecture/id-guide-for-agents.md](architecture/id-guide-for-agents.md) (Implementation Guide)
4. **Multi-Tenancy**: [multi-tenancy.md](multi-tenancy.md)

### For AI/ML Engineers

1. **Agents Overview**: [agents/overview.md](agents/overview.md)
   - RAG Graph
   - External Knowledge Graph
   - Framework Analysis Graph
2. **RAG Pipeline**:
   - [rag/overview.md](rag/overview.md) (Architecture)
   - [rag/ingestion.md](rag/ingestion.md) (Upload & Crawler Ingestion Graphs)
   - [rag/retrieval-contracts.md](rag/retrieval-contracts.md) (Query Interface)
3. **Tool Contracts**: [agents/tool-contracts.md](agents/tool-contracts.md)
4. **Observability**: [observability/langfuse.md](observability/langfuse.md)

### For Document Processing Engineers

1. **Subsystem Overview**: [architektur/documents-subsystem.md](architektur/documents-subsystem.md)
2. **Contracts Reference**: [documents/contracts-reference.md](documents/contracts-reference.md)
3. **Parser Reference**: [documents/parser-referenz.md](documents/parser-referenz.md)
4. **Assets & Multimodality**: [documents/assets-und-multimodalitaet.md](documents/assets-und-multimodalitaet.md)
5. **Telemetry**: [observability/documents-telemetry.md](observability/documents-telemetry.md)

### For DevOps/SRE

1. **CI/CD Pipeline**: [cicd/pipeline.md](cicd/pipeline.md)
2. **Scaling**: [operations/scaling.md](operations/scaling.md)
3. **Runbooks**:
   - [runbooks/migrations.md](runbooks/migrations.md)
   - [runbooks/rag_delete.md](runbooks/rag_delete.md)
   - [runbooks/incidents.md](runbooks/incidents.md)
4. **Cloud Deployments**:
   - [cloud/gcp-prod.md](cloud/gcp-prod.md)
   - [cloud/gcp-staging.md](cloud/gcp-staging.md)
5. **Docker Conventions**: [docker/conventions.md](docker/conventions.md)

---

## Documentation Structure

```
docs/
├── README.md (THIS FILE)
├── agents/          # LangGraph agents & tools
├── architecture/    # System design, ADRs, IDs
├── architektur/     # German architecture docs (to be merged with architecture/)
├── documents/       # Document subsystem (parsing, assets, persistence)
├── rag/             # Retrieval-Augmented Generation
├── crawler/         # Web crawler
├── development/     # Developer guides
├── audit/           # Audit reports (contract divergence, cleanup)
├── observability/   # Langfuse, ELK, telemetry
├── api/             # API reference, changelog
├── cicd/            # CI/CD pipeline docs
├── operations/      # Scaling, deployment
├── runbooks/        # Operational runbooks
├── security/        # Security policies, secrets
├── domain/          # Domain models (Cases, etc.)
├── llm/             # LLM generation policies
├── qa/              # QA checklists
├── cloud/           # Cloud-specific guides (GCP)
├── docker/          # Docker conventions
├── environments/    # Environment matrix
├── litellm/         # LiteLLM admin
└── *.md             # Misc root docs (multi-tenancy, PII, frontend, demo)
```

---

## Quick Links

### Architecture Decision Records (ADRs)
- [ADR-001: Case ID Semantics](architecture/adrs/ADR-001-case-id-semantics.md)
- [ADR-002: Workflow ID Generation](architecture/adrs/ADR-002-workflow-id-generation.md)
- [ADR-003: Tenant ID Propagation](architecture/adrs/ADR-003-tenant-id-propagation.md)
- [ADR-004: Lifecycle Configuration](architecture/adrs/ADR-004-lifecycle-configuration.md)
- [ADR-005: Validation Policy](architecture/adrs/ADR-005-validation-policy.md)

### Contracts
- **Tool Contracts**: [agents/tool-contracts.md](agents/tool-contracts.md)
- **Document Contracts**: [documents/contracts-reference.md](documents/contracts-reference.md)
- **RAG Retrieval Contracts**: [rag/retrieval-contracts.md](rag/retrieval-contracts.md)
- **Reranking Contracts**: [agents/reranking-contracts.md](agents/reranking-contracts.md)

### Developer Tools
- **RAG Developer Workbench**: [development/rag-tools-workbench.md](development/rag-tools-workbench.md)
- **Manual Setup**: [development/manual-setup.md](development/manual-setup.md)

### API & SDK
- **API Reference**: [api/reference.md](api/reference.md)
- **API Changelog**: [api/CHANGELOG.md](api/CHANGELOG.md)
- **LiteLLM Admin**: [api/litellm-admin.md](api/litellm-admin.md)

---

## Contributing to Documentation

### Documentation Principles

1. **Single Source of Truth**: Root `/AGENTS.md` is the authoritative source for architecture, IDs, and contracts. Do not duplicate this content in other files—link to it.

2. **Pointer Principle**: Operational guides ([`/CLAUDE.md`](../CLAUDE.md), [`/Gemini.md`](../Gemini.md)) only contain agent-specific commands and point to `/AGENTS.md` for architecture.

3. **No Redundancy**: Each concept is documented exactly once. Use links to connect related docs.

4. **Detail Hierarchy**:
   - **High-Level**: Root `/AGENTS.md` (architecture, contracts, glossar)
   - **Mid-Level**: `docs/` subdirectories (detailed guides, implementations)
   - **Low-Level**: Code docstrings, inline comments

### Before Adding a New Doc

1. Check if content already exists in:
   - Root `/AGENTS.md`
   - Existing `docs/` subdirectories
2. If overlap exists, **link** instead of duplicating
3. If truly new content, place in appropriate subdirectory
4. Update this `README.md` navigation section

### Before Updating an Existing Doc

1. Verify it's not superseded by Root `/AGENTS.md`
2. Check for broken links after updates
3. Run `npm run lint:fix` (Prettier) before committing

---

## Recent Changes

- **2025-12-15**: Major cleanup - Removed 23 redundant files, established `/AGENTS.md` as Single Source of Truth ([audit/cleanup_report.md](audit/cleanup_report.md))
- **2025-12-15**: Added comprehensive documentation for:
  - Upload Ingestion Graph ([rag/ingestion.md](rag/ingestion.md))
  - External Knowledge Graph ([agents/overview.md](agents/overview.md))
  - RAG Developer Workbench ([development/rag-tools-workbench.md](development/rag-tools-workbench.md))
- **2025-12-15**: Contract divergence audit completed ([audit/contract_divergence_report.md](audit/contract_divergence_report.md))

---

## Support

- **Questions about Architecture/IDs/Contracts?** → Start with [Root `/AGENTS.md`](../AGENTS.md)
- **Questions about Development Setup?** → [`/CLAUDE.md`](../CLAUDE.md) or [`/Gemini.md`](../Gemini.md)
- **Questions about Specific Feature?** → Use navigation above to find relevant doc
- **Found a Documentation Bug?** → Report in GitHub Issues or fix directly (PR welcome)

---

**Last Updated**: 2025-12-15
**Maintained By**: NOESIS 2 Documentation Team
