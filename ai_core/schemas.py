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
    """Optional guardrail limit overrides for crawler runs."""

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


class CrawlerRunSnapshot(BaseModel):
    """Snapshot configuration for crawler runs."""

    enabled: bool = False
    label: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_bool(cls, value: object) -> Mapping[str, object] | object:
        if isinstance(value, bool):
            return {"enabled": value}
        return value

    @field_validator("label", mode="before")
    @classmethod
    def _normalise_label(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_snapshot_label",
                "snapshot label must be a string when provided.",
            )
        trimmed = value.strip()
        return trimmed or None


class CrawlerRunOrigin(BaseModel):
    """Origin description for crawler runs."""

    url: str
    provider: str = "web"
    document_id: str | None = None
    title: str | None = None
    language: str | None = None
    content: str | None = None
    content_type: str = "text/html"
    snapshot_label: str | None = None
    tags: list[str] | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _normalise_url(cls, value: object) -> str:
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_origin_url",
                "origin url must be provided as a non-empty string.",
            )
        url = value.strip()
        if not url:
            raise PydanticCustomError(
                "invalid_origin_url",
                "origin url must be provided as a non-empty string.",
            )
        return url

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

    @field_validator(
        "document_id", "title", "language", "content", "snapshot_label", mode="before"
    )
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
        if info.field_name == "content" and not trimmed:
            raise PydanticCustomError(
                "invalid_content",
                "content must be provided as a non-empty string.",
            )
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
        normalised: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            if not entry or entry in seen:
                continue
            seen.add(entry)
            normalised.append(entry)
        return normalised or None


class CrawlerRunRequest(BaseModel):
    """Request payload used by the crawler LangGraph runner."""

    model_config = ConfigDict(extra="ignore")

    workflow_id: str | None = None
    origins: list[CrawlerRunOrigin] = Field(..., min_length=1)
    mode: Literal["live", "content"] = "live"
    collection_id: str | None = None
    limits: CrawlerRunLimits | None = None
    review: Literal["required", "approved", "rejected"] | None = None
    snapshot: CrawlerRunSnapshot = Field(default_factory=CrawlerRunSnapshot)
    tags: list[str] | None = None
    shadow_mode: bool = False
    dry_run: bool = False
    force_retire: bool = False
    recompute_delta: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_payload(cls, value: object) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise PydanticCustomError(
                "invalid_request",
                "Crawler request must be provided as an object.",
            )
        data = dict(value)
        if "origins" not in data and data.get("origin_url"):
            origin: dict[str, object] = {"url": data.get("origin_url")}
            if data.get("provider"):
                origin["provider"] = data.get("provider")
            for field in ("document_id", "title", "language", "content"):
                if data.get(field) is not None:
                    origin[field] = data[field]
            if data.get("content_type"):
                origin["content_type"] = data["content_type"]
            if data.get("snapshot_label"):
                origin["snapshot_label"] = data["snapshot_label"]
            if data.get("tags"):
                origin["tags"] = data["tags"]
            data["origins"] = [origin]
        if "limits" not in data and data.get("max_document_bytes") is not None:
            data["limits"] = {"max_document_bytes": data["max_document_bytes"]}
        if "snapshot" not in data:
            snapshot_enabled = data.get("snapshot") if "snapshot" in data else None
            label = data.get("snapshot_label")
            if snapshot_enabled is not None or label is not None:
                data["snapshot"] = {
                    "enabled": bool(snapshot_enabled),
                    "label": label,
                }
        if "review" not in data and data.get("manual_review") is not None:
            data["review"] = data.get("manual_review")
        if "mode" not in data and data.get("fetch") is not None:
            data["mode"] = "live" if data.get("fetch") else "content"
        return data

    @field_validator("workflow_id", mode="before")
    @classmethod
    def _trim_workflow(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise PydanticCustomError(
                "invalid_workflow_id",
                "workflow_id must be a string when provided.",
            )
        trimmed = value.strip()
        return trimmed or None

    @field_validator("collection_id", mode="before")
    @classmethod
    def _normalise_collection_id(cls, value: object) -> str | None:
        return _normalise_optional_uuid(value, "collection_id")

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
        normalised = []
        seen: set[str] = set()
        for entry in entries:
            if not entry:
                continue
            if entry in seen:
                continue
            seen.add(entry)
            normalised.append(entry)
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

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> "CrawlerRunRequest":
        if self.mode == "content":
            for origin in self.origins:
                if origin.content is None:
                    raise PydanticCustomError(
                        "content_required",
                        "content mode requires inline content for every origin.",
                    )
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
