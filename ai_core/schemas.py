from __future__ import annotations
from typing import Any, Mapping, Literal, Annotated
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict, FieldValidationInfo
from pydantic_core import PydanticCustomError

# NOTE: This schema uses Pydantic v2 features (e.g., Field(min_length) on lists).
# The project's requirements should be pinned to Pydantic >=2.0,<3.0.


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

    @field_validator("document_ids", mode="after")
    @classmethod
    def _normalise_document_ids(cls, value: list[object]) -> list[str]:
        normalized: list[str] = []
        for raw in value:
            if not isinstance(raw, str):
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain non-empty document identifiers.",
                )
            trimmed = raw.strip()
            if not trimmed:
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain non-empty document identifiers.",
                )
            normalized.append(trimmed)
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
            if not isinstance(raw, str):
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain non-empty UUID strings.",
                )
            trimmed = raw.strip()
            if not trimmed:
                raise PydanticCustomError(
                    "invalid_document_ids",
                    "document_ids must contain non-empty UUID strings.",
                )
            normalised.append(trimmed)
        return normalised

    @field_validator("reason", "ticket_ref", mode="before")
    @classmethod
    def _validate_required_strings(
        cls, value: object, info: FieldValidationInfo
    ) -> str:
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


class RagUploadMetadata(BaseModel):
    """Metadata contract for document uploads."""

    model_config = ConfigDict(extra="allow")

    external_id: str | None = None

    @field_validator("external_id", mode="before")
    @classmethod
    def _normalise_external_id(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return None


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

    @field_validator(
        "question", "query", "process", "doc_class", "visibility", mode="before"
    )
    @classmethod
    def _normalise_optional_text(
        cls, value: object, info: FieldValidationInfo
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
        cls, value: object, info: FieldValidationInfo
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        raise PydanticCustomError(
            f"invalid_{info.field_name}",
            f"{info.field_name} must be a JSON object when provided.",
        )

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
        return self
