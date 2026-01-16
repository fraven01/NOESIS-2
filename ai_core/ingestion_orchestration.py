"""Ingestion orchestration components with separated concerns.

Refactored from run_ingestion_graph god-function into:
- IngestionContextBuilder: Defensive metadata extraction
- ObservabilityWrapper: Tracing lifecycle management
- Cleaner orchestration function
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Dict
from collections.abc import Mapping as MappingABC

from ai_core.tool_contracts.base import tool_context_from_meta


@dataclass(frozen=True)
class IngestionContext:
    """Normalized ingestion context extracted from various sources."""

    tenant_id: Optional[str]
    case_id: Optional[str]
    workflow_id: Optional[str]
    trace_id: Optional[str]
    collection_id: Optional[str]
    source: Optional[str]
    document_id: Optional[str]
    run_id: Optional[str]
    ingestion_run_id: Optional[str]
    raw_payload_path: Optional[str]


class IngestionContextBuilder:
    """Builder for defensive metadata extraction from nested dictionaries.

    Handles the complex fallback chains for extracting context from:
    - state dict
    - meta dict
    - trace_context dict
    - nested raw_document/metadata dicts
    """

    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        """Coerce value to string or None.

        Args:
            value: Any value

        Returns:
            String representation or None if empty/None
        """
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    @staticmethod
    def _extract_from_mapping(mapping: Any, key: str) -> Any:
        """Safely extract key from mapping-like object.

        Args:
            mapping: Dict-like object or None
            key: Key to extract

        Returns:
            Value or None if not found
        """
        if not isinstance(mapping, Mapping):
            return None
        return mapping.get(key)

    @staticmethod
    def _extract_tool_context(meta: Optional[Mapping[str, Any]]) -> Any:
        if not isinstance(meta, MappingABC):
            return None
        try:
            return tool_context_from_meta(meta)
        except (TypeError, ValueError):
            return None

    def build_from_state(
        self,
        state: Mapping[str, Any],
        meta: Optional[Mapping[str, Any]],
        trace_context: Mapping[str, Any],
    ) -> IngestionContext:
        """Build context from state, meta, and trace_context.

        Args:
            state: Graph state dict
            meta: Optional metadata dict
            trace_context: Trace context dict

        Returns:
            IngestionContext with extracted values

        Business logic:
        - Priority: trace_context > meta > state > nested objects
        - Defensive: Multiple fallback chains for each field
        - Raw payload path: Special extraction from state
        """
        state_meta = self._extract_from_mapping(state, "meta")
        tool_context = self._extract_tool_context(meta)
        scope_context = tool_context.scope if tool_context else None
        business_context = tool_context.business if tool_context else None
        raw_reference = self._extract_from_mapping(state, "raw_document")
        raw_metadata = None
        if isinstance(raw_reference, MappingABC):
            raw_metadata = raw_reference.get("metadata")
            if not isinstance(raw_metadata, MappingABC):
                raw_metadata = None

        # Extract tenant_id
        tenant_id = self._coerce_str(
            trace_context.get("tenant_id")
            or (scope_context.tenant_id if scope_context else None)
            or self._extract_from_mapping(state, "tenant_id")
        )

        # Extract case_id (BREAKING CHANGE: now in business_context)
        case_id = self._coerce_str(
            trace_context.get("case_id")
            or (business_context.case_id if business_context else None)
            or self._extract_from_mapping(state, "case_id")
        )

        # Extract workflow_id (BREAKING CHANGE: now in business_context)
        workflow_id = self._coerce_str(
            trace_context.get("workflow_id")
            or (business_context.workflow_id if business_context else None)
            or self._extract_from_mapping(state_meta, "workflow_id")
            or self._extract_from_mapping(state, "workflow_id")
            or self._extract_from_mapping(raw_reference, "workflow_id")
            or self._extract_from_mapping(raw_metadata, "workflow_id")
        )

        # Extract trace_id
        trace_id = self._coerce_str(trace_context.get("trace_id"))

        # Extract collection_id (BREAKING CHANGE: now in business_context)
        collection_id = self._coerce_str(
            trace_context.get("collection_id")
            or (business_context.collection_id if business_context else None)
            or self._extract_from_mapping(state, "collection_id")
            or self._extract_from_mapping(raw_metadata, "collection_id")
        )

        # Extract source
        source = self._coerce_str(
            self._extract_from_mapping(meta, "source")
            or self._extract_from_mapping(state_meta, "source")
            or self._extract_from_mapping(state, "source")
            or self._extract_from_mapping(raw_reference, "source")
            or self._extract_from_mapping(raw_metadata, "source")
        )

        # Extract document_id
        document_id = self._coerce_str(trace_context.get("document_id"))

        # Extract run_id
        run_id = self._coerce_str(
            trace_context.get("run_id")
            or (scope_context.run_id if scope_context else None)
            or self._extract_from_mapping(state_meta, "run_id")
        )

        # Extract ingestion_run_id
        ingestion_run_id = self._coerce_str(
            trace_context.get("ingestion_run_id")
            or (scope_context.ingestion_run_id if scope_context else None)
            or self._extract_from_mapping(state_meta, "ingestion_run_id")
        )

        # Extract raw_payload_path (special logic)
        raw_payload_path = None
        if isinstance(state, MappingABC):
            candidate = state.get("raw_payload_path")
            if isinstance(candidate, str) and candidate.strip():
                raw_payload_path = candidate.strip()
            else:
                if isinstance(raw_reference, MappingABC):
                    nested_candidate = raw_reference.get("payload_path")
                    if isinstance(nested_candidate, str) and nested_candidate.strip():
                        raw_payload_path = nested_candidate.strip()

        return IngestionContext(
            tenant_id=tenant_id,
            case_id=case_id,
            workflow_id=workflow_id,
            trace_id=trace_id,
            collection_id=collection_id,
            source=source,
            document_id=document_id,
            run_id=run_id,
            ingestion_run_id=ingestion_run_id,
            raw_payload_path=raw_payload_path,
        )


@dataclass
class ObservabilityContext:
    """Context for observability tracing."""

    trace_name: str
    user_id: Optional[str]
    session_id: Optional[str]
    metadata: Dict[str, str]
    task_identifier: Optional[str] = None


class ObservabilityWrapper:
    """Wrapper for observability lifecycle management.

    Handles:
    - Trace start/end
    - Metadata collection
    - Celery task ID tracking
    """

    def __init__(self, observability_helpers: Any):
        """Initialize with observability helpers.

        Args:
            observability_helpers: Observability helper module/object
        """
        self._helpers = observability_helpers

    def create_context(
        self,
        ingestion_ctx: IngestionContext,
        trace_context: Mapping[str, Any],
        task_request: Optional[Any] = None,
    ) -> ObservabilityContext:
        """Create observability context from ingestion context.

        Args:
            ingestion_ctx: Ingestion context
            trace_context: Trace context dict (for trace_id)
            task_request: Optional Celery task request object

        Returns:
            ObservabilityContext ready for tracing
        """
        # Extract Celery task ID if available
        task_identifier = None
        if task_request is not None:
            task_id = getattr(task_request, "id", None)
            if task_id:
                task_identifier = str(task_id).strip() or None

        # Build metadata dict
        metadata: Dict[str, str] = {}

        # Add trace_id from trace_context
        trace_id = trace_context.get("trace_id")
        if trace_id:
            metadata["trace_id"] = str(trace_id)

        if ingestion_ctx.tenant_id:
            metadata["tenant_id"] = ingestion_ctx.tenant_id
        user_id = trace_context.get("user_id")
        if user_id:
            metadata["user_id"] = str(user_id)
        service_id = trace_context.get("service_id")
        if service_id:
            metadata["service_id"] = str(service_id)
        if ingestion_ctx.case_id:
            metadata["case_id"] = ingestion_ctx.case_id
        if ingestion_ctx.document_id:
            metadata["document_id"] = ingestion_ctx.document_id
        if ingestion_ctx.run_id:
            metadata["run_id"] = ingestion_ctx.run_id
        if ingestion_ctx.ingestion_run_id:
            metadata["ingestion_run_id"] = ingestion_ctx.ingestion_run_id

        if task_identifier:
            metadata["celery.task_id"] = task_identifier

        return ObservabilityContext(
            trace_name="crawler.ingestion",
            user_id=str(user_id) if user_id else None,
            session_id=ingestion_ctx.case_id,
            metadata=metadata,
            task_identifier=task_identifier,
        )

    def start_trace(self, obs_ctx: ObservabilityContext) -> None:
        """Start observability trace.

        Args:
            obs_ctx: Observability context
        """
        # Add trace_id if available from external context
        # (Would be added by caller from trace_context)
        self._helpers.start_trace(
            name=obs_ctx.trace_name,
            user_id=obs_ctx.user_id,
            session_id=obs_ctx.session_id,
            metadata=obs_ctx.metadata or None,
        )

        if obs_ctx.task_identifier:
            self.update_observation({"celery.task_id": obs_ctx.task_identifier})

    def end_trace(self) -> None:
        """End observability trace."""
        self._helpers.end_trace()

    def update_observation(self, metadata: Dict[str, Any]) -> None:
        """Update observation metadata.

        Args:
            metadata: Metadata to update
        """
        # Delegate to observability helpers if method exists
        if hasattr(self._helpers, "update_observation"):
            self._helpers.update_observation(metadata=metadata)
