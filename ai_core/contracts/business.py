"""Business domain context for tool and graph invocations.

BusinessContext captures the domain-specific identifiers that describe
WHAT is being processed, independent of WHO is processing it (ScopeContext)
or HOW it's being processed (ToolContext runtime metadata).

Separation rationale (Option A - Strict Separation):
- Business IDs (case, document, collection) are domain concepts
- Scope IDs (tenant, trace, invocation) are infrastructure concerns
- Clear separation enables independent evolution and testing
- Single Responsibility: each context type has ONE clear purpose

This contract is part of the Option A migration to eliminate redundancies
between ScopeContext, ToolContext, and Tool-Input models.

See: OPTION_A_CONTRACT_IMPACT.md, OPTION_A_SOURCE_CODE_ANALYSIS.md
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Type aliases for clarity
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

    Golden Rule (Operationalized):
    - Tool-Inputs contain only functional parameters
    - Context contains Scope, Business, and Runtime Permissions
    - Tool-Run functions read identifiers exclusively from context, not params

    Migration notes:
    - Extracted from ScopeContext (previously mixed Scope + Business)
    - Tool-Inputs no longer contain these IDs (RetrieveInput, FrameworkAnalysisInput)
    - ToolContext composes ScopeContext + BusinessContext
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
