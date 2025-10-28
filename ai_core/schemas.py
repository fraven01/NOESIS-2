from __future__ import annotations
from typing import Any, Mapping, Literal, Annotated, Sequence
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict, ValidationInfo
from pydantic_core import PydanticCustomError

# NOTE: This schema uses Pydantic v2 features (e.g., Field(min_length) on lists).
# The project's requirements should be pinned to Pydantic >=2.0,<3.0.


def _normalise_optional_uuid(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        try:
            return str(UUID(trimmed))
        except (TypeError, ValueError):
            raise PydanticCustomError(
                f"invalid_{field_name}",
                f"{field_name} must be a valid UUID string.",
            )
    raise PydanticCustomError(
        f"invalid_{field_name}",
        f"{field_name} must be a valid UUID string.",
    )


def _normalise_collection_id_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PydanticCustomError(
            "invalid_collection_ids",
            "collection_ids must be provided as a list of UUID strings.",
        )

    normalised: list[str] = []
    seen: set[str] = set()
    for item in value:
        parsed = _normalise_optional_uuid(item, "collection_id")
        if parsed is None:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        normalised.append(parsed)

    return normalised


class RagIngestionRunRequest(BaseModel):
    """
    Request model for queueing a RAG ingestion run.
    """

    document_ids: list[Annotated[str, Field(min_length=1)]] = Field(
        ...,
        min_length=1,
        description="A non-empty list of non-empty document IDs to be ingested.",
    )
    priority: Literal["low", "normal", "high"] = "normal"
    embedding_profile: str
    collection_id: str | None = None

    @field_validator("document_ids", mode="after")
    @classmethod
    def _normalise_document_ids(cls, value: list[object]) -> list[str]:
        normalized: list[str] = []
        for raw in value:
            if isinstance(raw, UUID):
                normalized.append(str(raw))
                continue
            if not isinstance(raw, str):
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain valid UUID strings.",
                )
            trimmed = raw.strip()
            if not trimmed:
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain valid UUID strings.",
                )
            try:
                normalized.append(str(UUID(trimmed)))
            except (TypeError, ValueError):
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain valid UUID strings.",
                )
        return normalized

    @field_validator("embedding_profile", mode="before")
    @classmethod
    def _normalise_embedding_profile(cls, value: object) -> str:
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_embedding_profile",
                "embedding_profile must be a non-empty string.",
            )
        trimmed = value.strip()
        if not trimmed:
            raise PydanticCustomError(
                "invalid_embedding_profile",
                "embedding_profile must be a non-empty string.",
            )
        return trimmed

    @field_validator("collection_id", mode="before")
    @classmethod
    def _normalise_collection_id(cls, value: object) -> str | None:
        return _normalise_optional_uuid(value, "collection_id")


class RagHardDeleteAdminRequest(BaseModel):
    """Request payload for administrative hard delete operations."""

    tenant_id: str
    document_ids: list[str] = Field(..., min_length=1)
    reason: str
    ticket_ref: str
    operator_label: str | None = None
    tenant_schema: str | None = None

    @field_validator("tenant_id", mode="before")
    @classmethod
    def _normalize_tenant_id(cls, value: object) -> str:
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise PydanticCustomError(
                "invalid_tenant_id",
                "tenant_id must be a valid UUID string.",
            )
        try:
            return str(UUID(str(value)))
        except (TypeError, ValueError):
            raise PydanticCustomError(
                "invalid_tenant_id",
                "tenant_id must be a valid UUID string.",
            )

    @field_validator("document_ids", mode="before")
    @classmethod
    def _ensure_document_ids_list(cls, value: object) -> object:
        if not isinstance(value, list) or not value:
            raise PydanticCustomError(
                "invalid_document_ids",
                "document_ids must be a non-empty list.",
            )
        return value

    @field_validator("document_ids", mode="after")
    @classmethod
    def _normalize_document_ids(cls, value: list[object]) -> list[str]:
        normalised: list[str] = []
        for raw in value:
            if isinstance(raw, UUID):
                normalised.append(str(raw))
                continue
            if not isinstance(raw, str):
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain valid UUID strings.",
                )
            trimmed = raw.strip()
            if not trimmed:
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain valid UUID strings.",
                )
            try:
                normalised.append(str(UUID(trimmed)))
            except (TypeError, ValueError):
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain valid UUID strings.",
                )
        return normalised

    @field_validator("reason", "ticket_ref", mode="before")
    @classmethod
    def _validate_required_strings(cls, value: object, info: ValidationInfo) -> str:
        if not isinstance(value, str):
            raise PydanticCustomError(
                f"invalid_{info.field_name}",
                f"{info.field_name} must be a non-empty string.",
            )
        trimmed = value.strip()
        if not trimmed:
            raise PydanticCustomError(
                f"invalid_{info.field_name}",
                f"{info.field_name} must be a non-empty string.",
            )
        return trimmed

    @field_validator("operator_label", mode="before")
    @classmethod
    def _normalise_optional_label(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return None

    @field_validator("tenant_schema", mode="before")
    @classmethod
    def _normalise_optional_schema(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return None


class CrawlerRunLimits(BaseModel):
    """Optional limits that influence crawler runtime decisions."""

    max_document_bytes: int | None = None

    @field_validator("max_document_bytes", mode="before")
    @classmethod
    def _coerce_max_document_bytes(cls, value: object) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise PydanticCustomError(
                "invalid_max_document_bytes",
                "max_document_bytes must be a positive integer.",
            )
        if parsed < 0:
            raise PydanticCustomError(
                "invalid_max_document_bytes",
                "max_document_bytes must be a positive integer.",
            )
        return parsed


class CrawlerSnapshotOptions(BaseModel):
    """Snapshot configuration shared across crawler origins."""

    enabled: bool = False
    label: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_snapshot(cls, value: object) -> Mapping[str, object] | object:
        if value is None:
            return {}
        if isinstance(value, bool):
            return {"enabled": value}
        if isinstance(value, Mapping):
            return value
        raise PydanticCustomError(
            "invalid_snapshot",
            "snapshot must be provided as an object or boolean.",
        )

    @field_validator("label", mode="before")
    @classmethod
    def _trim_label(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_snapshot_label",
                "snapshot label must be a string when provided.",
            )
        trimmed = value.strip()
        return trimmed or None


class CrawlerOriginConfig(BaseModel):
    """Origin specific configuration for crawler runs."""

    url: str
    provider: str | None = None
    document_id: str | None = None
    title: str | None = None
    language: str | None = None
    content: str | None = None
    content_type: str | None = None
    fetch: bool | None = None
    tags: list[str] | None = None
    limits: CrawlerRunLimits | None = None
    snapshot: CrawlerSnapshotOptions | None = None
    review: Literal["required", "approved", "rejected"] | None = None
    dry_run: bool | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _normalise_url(cls, value: object) -> str:
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_origin_url",
                "origin URL must be provided as a non-empty string.",
            )
        candidate = value.strip()
        if not candidate:
            raise PydanticCustomError(
                "invalid_origin_url",
                "origin URL must be provided as a non-empty string.",
            )
        return candidate

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_provider",
                "provider must be a non-empty string when provided.",
            )
        candidate = value.strip()
        if not candidate:
            raise PydanticCustomError(
                "invalid_provider",
                "provider must be a non-empty string when provided.",
            )
        return candidate

    @field_validator("document_id", "title", "language", mode="before")
    @classmethod
    def _trim_optional_text(cls, value: object, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                f"invalid_{info.field_name}",
                f"{info.field_name} must be a string when provided.",
            )
        trimmed = value.strip()
        return trimmed or None

    @field_validator("content", mode="before")
    @classmethod
    def _ensure_content(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_content",
                "content must be provided as a non-empty string.",
            )
        candidate = value.strip()
        if not candidate:
            raise PydanticCustomError(
                "invalid_content",
                "content must be provided as a non-empty string.",
            )
        return candidate

    @field_validator("content_type", mode="before")
    @classmethod
    def _normalise_content_type(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_content_type",
                "content_type must be a non-empty string when provided.",
            )
        candidate = value.strip()
        if not candidate:
            raise PydanticCustomError(
                "invalid_content_type",
                "content_type must be a non-empty string when provided.",
            )
        return candidate

    @field_validator("tags", mode="before")
    @classmethod
    def _normalise_tags(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            entries = [entry.strip() for entry in value.split(",")]
        elif isinstance(value, Sequence):
            entries = [str(entry).strip() for entry in value]
        else:
            raise PydanticCustomError(
                "invalid_tags",
                "tags must be provided as a list or comma separated string.",
            )
        normalised = [entry for entry in entries if entry]
        return normalised or None

    @field_validator("review", mode="before")
    @classmethod
    def _normalise_review(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip().lower()
            if not candidate:
                return None
            if candidate not in {"required", "approved", "rejected"}:
                raise PydanticCustomError(
                    "invalid_review",
                    "review must be one of required, approved, rejected.",
                )
            return candidate
        raise PydanticCustomError(
            "invalid_review",
            "review must be a string when provided.",
        )


class CrawlerRunRequest(BaseModel):
    """Request payload used by the crawler LangGraph runner."""

    workflow_id: str | None = None
    mode: Literal["live", "manual"] = "live"
    origins: list[CrawlerOriginConfig] | None = None
    origin_url: str | None = None
    provider: str = "web"
    document_id: str | None = None
    title: str | None = None
    language: str | None = None
    content: str | None = None
    content_type: str = "text/html"
    fetch: bool = True
    snapshot: CrawlerSnapshotOptions = Field(default_factory=CrawlerSnapshotOptions)
    snapshot_label: str | None = None
    tags: list[str] | None = None
    shadow_mode: bool = False
    dry_run: bool = False
    review: Literal["required", "approved", "rejected"] | None = None
    manual_review: Literal["required", "approved", "rejected"] | None = None
    force_retire: bool = False
    recompute_delta: bool = False
    max_document_bytes: int | None = None
    limits: CrawlerRunLimits | None = None
    collection_id: str | None = None

    @field_validator("origin_url", mode="before")
    @classmethod
    def _normalise_origin_url(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_origin_url",
                "origin_url must be provided as a non-empty string.",
            )
        candidate = value.strip()
        if not candidate:
            raise PydanticCustomError(
                "invalid_origin_url",
                "origin_url must be provided as a non-empty string.",
            )
        return candidate

    @field_validator("provider", "content_type", mode="before")
    @classmethod
    def _normalise_simple_text(cls, value: object, info: ValidationInfo) -> str:
        if not isinstance(value, str):
            raise PydanticCustomError(
                f"invalid_{info.field_name}",
                f"{info.field_name} must be a non-empty string.",
            )
        candidate = value.strip()
        if not candidate:
            raise PydanticCustomError(
                f"invalid_{info.field_name}",
                f"{info.field_name} must be a non-empty string.",
            )
        return candidate

    @field_validator("workflow_id", "document_id", "title", "language", mode="before")
    @classmethod
    def _trim_optional_text(cls, value: object, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                f"invalid_{info.field_name}",
                f"{info.field_name} must be a string when provided.",
            )
        trimmed = value.strip()
        return trimmed or None

    @field_validator("content", mode="before")
    @classmethod
    def _ensure_content(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_content",
                "content must be provided as a non-empty string.",
            )
        candidate = value.strip()
        if not candidate:
            raise PydanticCustomError(
                "invalid_content",
                "content must be provided as a non-empty string.",
            )
        return candidate

    @field_validator("snapshot_label", mode="before")
    @classmethod
    def _normalise_snapshot_label(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_snapshot_label",
                "snapshot_label must be a string when provided.",
            )
        trimmed = value.strip()
        return trimmed or None

    @field_validator("tags", mode="before")
    @classmethod
    def _normalise_tags(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            entries = [entry.strip() for entry in value.split(",")]
        elif isinstance(value, Sequence):
            entries = [str(entry).strip() for entry in value]
        else:
            raise PydanticCustomError(
                "invalid_tags",
                "tags must be provided as a list or comma separated string.",
            )
        normalised = [entry for entry in entries if entry]
        return normalised or None

    @field_validator("review", "manual_review", mode="before")
    @classmethod
    def _normalise_review(cls, value: object, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip().lower()
            if not candidate:
                return None
            if candidate not in {"required", "approved", "rejected"}:
                raise PydanticCustomError(
                    f"invalid_{info.field_name}",
                    f"{info.field_name} must be one of required, approved, rejected.",
                )
            return candidate
        raise PydanticCustomError(
            f"invalid_{info.field_name}",
            f"{info.field_name} must be a string when provided.",
        )

    @field_validator("max_document_bytes", mode="before")
    @classmethod
    def _coerce_max_document_bytes(cls, value: object) -> int | None:
        return CrawlerRunLimits._coerce_max_document_bytes(value)  # type: ignore[arg-type]

    @field_validator("collection_id", mode="before")
    @classmethod
    def _normalise_collection_id(cls, value: object) -> str | None:
        return _normalise_optional_uuid(value, "collection_id")

    @model_validator(mode="after")
    def _finalise(self) -> "CrawlerRunRequest":
        if self.review and self.manual_review and self.review != self.manual_review:
            raise PydanticCustomError(
                "invalid_review",
                "review and manual_review must match when both are provided.",
            )
        if self.review and not self.manual_review:
            object.__setattr__(self, "manual_review", self.review)
        elif self.manual_review and not self.review:
            object.__setattr__(self, "review", self.manual_review)

        snapshot_label = self.snapshot_label or self.snapshot.label
        if snapshot_label and not self.snapshot.label:
            object.__setattr__(
                self,
                "snapshot",
                self.snapshot.model_copy(update={"label": snapshot_label}),
            )

        origins = list(self.origins or [])
        if not origins:
            if not self.origin_url:
                raise PydanticCustomError(
                    "missing_origins",
                    "At least one origin must be provided via origins or origin_url.",
                )
            limits = self.limits
            if limits is None and self.max_document_bytes is not None:
                limits = CrawlerRunLimits(max_document_bytes=self.max_document_bytes)
            origin_snapshot: CrawlerSnapshotOptions | None = None
            if self.snapshot.enabled or self.snapshot.label:
                origin_snapshot = self.snapshot
            origin = CrawlerOriginConfig(
                url=self.origin_url,
                provider=self.provider,
                document_id=self.document_id,
                title=self.title,
                language=self.language,
                content=self.content,
                content_type=self.content_type,
                fetch=self.fetch,
                tags=self.tags,
                limits=limits,
                snapshot=origin_snapshot,
                review=self.review,
                dry_run=self.dry_run,
            )
            origins = [origin]
        else:
            normalised: list[CrawlerOriginConfig] = []
            default_limits = self.limits
            if default_limits is None and self.max_document_bytes is not None:
                default_limits = CrawlerRunLimits(
                    max_document_bytes=self.max_document_bytes
                )
            for origin in origins:
                update_payload: dict[str, object] = {}
                if origin.provider is None:
                    update_payload["provider"] = self.provider
                if origin.content_type is None:
                    update_payload["content_type"] = self.content_type
                if origin.fetch is None:
                    update_payload["fetch"] = self.fetch
                if origin.tags is None and self.tags is not None:
                    update_payload["tags"] = list(self.tags)
                if origin.limits is None and default_limits is not None:
                    update_payload["limits"] = default_limits.model_copy()
                if origin.snapshot is None and (
                    self.snapshot.enabled or self.snapshot.label
                ):
                    update_payload["snapshot"] = self.snapshot.model_copy()
                if origin.review is None and self.review is not None:
                    update_payload["review"] = self.review
                if origin.dry_run is None:
                    update_payload["dry_run"] = self.dry_run
                if update_payload:
                    origin = origin.model_copy(update=update_payload)
                normalised.append(origin)
            origins = normalised

        if self.mode == "manual":
            manual_origins: list[CrawlerOriginConfig] = []
            for origin in origins:
                updated = origin
                if updated.fetch not in (None, False):
                    raise PydanticCustomError(
                        "invalid_manual_mode",
                        "Manual mode does not support remote fetching.",
                    )
                if not updated.content and not self.content:
                    raise PydanticCustomError(
                        "content_required_when_fetch_disabled",
                        "Manual mode requires inline content for each origin.",
                    )
                if updated.content is None and self.content:
                    updated = updated.model_copy(update={"content": self.content})
                if updated.content_type is None:
                    updated = updated.model_copy(
                        update={"content_type": self.content_type}
                    )
                if updated.fetch is None:
                    updated = updated.model_copy(update={"fetch": False})
                manual_origins.append(updated)
            origins = manual_origins

        for origin in origins:
            if origin.fetch is False and origin.content is None:
                raise PydanticCustomError(
                    "content_required_when_fetch_disabled",
                    "content must be provided when fetch is disabled.",
                )

        object.__setattr__(self, "origins", origins)
        return self


class RagUploadMetadata(BaseModel):
    """Metadata contract for document uploads."""

    model_config = ConfigDict(extra="allow")

    external_id: str | None = None
    collection_id: str | None = None

    @field_validator("external_id", mode="before")
    @classmethod
    def _normalise_external_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return None

    @field_validator("collection_id", mode="before")
    @classmethod
    def _normalise_collection_id(cls, value: object) -> str | None:
        return _normalise_optional_uuid(value, "collection_id")


class _GraphStateBase(BaseModel):
    """Shared base for agent graph request payloads."""

    model_config = ConfigDict(extra="allow")

    prompt: str | None = None
    metadata: dict[str, Any] | None = None
    scope: str | None = None
    needs_input: list[Any] | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def _ensure_metadata_mapping(cls, value: object) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        raise PydanticCustomError(
            "invalid_metadata",
            "metadata must be a JSON object when provided.",
        )

    @field_validator("needs_input", mode="before")
    @classmethod
    def _ensure_needs_list(cls, value: object) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        raise PydanticCustomError(
            "invalid_needs_input",
            "needs_input must be provided as a list.",
        )


class InfoIntakeRequest(_GraphStateBase):
    """Initial payload accepted by the intake workflow."""


class ScopeCheckRequest(_GraphStateBase):
    """State updates applied during the scope validation step."""


class NeedsMappingRequest(_GraphStateBase):
    """State updates consumed by the needs mapping graph."""


class SystemDescriptionRequest(_GraphStateBase):
    """Terminal request payload for the system description graph."""


class RagQueryRequest(_GraphStateBase):
    """Request payload accepted by the production retrieval graph."""

    question: str | None = None
    query: str | None = None
    filters: dict[str, Any] | None = None
    process: str | None = None
    doc_class: str | None = None
    visibility: str | None = None
    visibility_override_allowed: bool | None = None
    hybrid: dict[str, Any] | None = None
    collection_id: str | None = None

    @field_validator(
        "question", "query", "process", "doc_class", "visibility", mode="before"
    )
    @classmethod
    def _normalise_optional_text(
        cls, value: object, info: ValidationInfo
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        raise PydanticCustomError(
            f"invalid_{info.field_name}",
            f"{info.field_name} must be a string when provided.",
        )

    @field_validator("filters", "hybrid", mode="before")
    @classmethod
    def _ensure_mapping(
        cls, value: object, info: ValidationInfo
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        raise PydanticCustomError(
            f"invalid_{info.field_name}",
            f"{info.field_name} must be a JSON object when provided.",
        )

    @field_validator("collection_id", mode="before")
    @classmethod
    def _normalise_collection_id(cls, value: object) -> str | None:
        # Accept missing/empty values to allow global queries across all documents.
        return _normalise_optional_uuid(value, "collection_id")

    @model_validator(mode="after")
    def _apply_collection_scope(self) -> "RagQueryRequest":
        filters: dict[str, Any] | None = None
        if self.filters is not None:
            filters = dict(self.filters)

        if filters is None:
            filters = {}

        existing_collection_ids = _normalise_collection_id_list(
            filters.get("collection_ids")
        )
        if existing_collection_ids is None:
            existing_collection_ids = []

        filter_collection_id = filters.get("collection_id")
        if filter_collection_id is not None:
            normalised_filter_id = _normalise_optional_uuid(
                filter_collection_id, "collection_id"
            )
            if normalised_filter_id:
                if normalised_filter_id not in existing_collection_ids:
                    existing_collection_ids.insert(0, normalised_filter_id)
            filters.pop("collection_id", None)

        # Only apply a concrete collection scope if provided either via the
        # request body (self.collection_id) or as legacy filter fields above.
        if self.collection_id is not None:
            if self.collection_id not in existing_collection_ids:
                existing_collection_ids.insert(0, self.collection_id)
            filters["collection_id"] = self.collection_id

        # Persist list only when non-empty; otherwise omit to indicate global search
        if existing_collection_ids:
            filters["collection_ids"] = existing_collection_ids
        else:
            filters.pop("collection_ids", None)
            filters.pop("collection_id", None)

        if filters:
            self.filters = filters
        else:
            self.filters = None

        return self

    @model_validator(mode="after")
    def _ensure_question_query(self) -> "RagQueryRequest":
        question = self.question
        query = self.query
        if not question and not query:
            raise PydanticCustomError(
                "missing_question",
                "Either question or query must be provided.",
            )
        if question and not query:
            self.query = question
        elif query and not question:
            self.question = query
        if self.hybrid is None:
            raise PydanticCustomError(
                "missing_hybrid",
                "hybrid configuration must be provided.",
            )
        return self
