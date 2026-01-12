from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


def _strip_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


class RagChatPayload(BaseModel):
    message: str
    tenant_id: str
    tenant_schema: str | None = None
    case_id: str | None = None
    collection_id: str | None = None
    thread_id: str | None = None
    global_search: bool | None = None

    # Hybrid tuning fields (existing client payloads)
    alpha: float | None = None
    min_sim: float | None = None
    top_k: int | None = Field(default=None, ge=1)
    vec_limit: int | None = Field(default=None, ge=1)
    lex_limit: int | None = Field(default=None, ge=1)
    trgm_limit: float | None = None
    max_candidates: int | None = Field(default=None, ge=1)
    diversify_strength: float | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("message", "tenant_id", mode="before")
    @classmethod
    def _require_text(cls, value: object, info: ValidationInfo) -> str:
        text = _strip_text(value)
        if not text:
            raise ValueError(f"{info.field_name} is required")
        return text

    @field_validator(
        "tenant_schema",
        "case_id",
        "collection_id",
        "thread_id",
        mode="before",
    )
    @classmethod
    def _optional_text(cls, value: object) -> str | None:
        text = _strip_text(value)
        return text or None
