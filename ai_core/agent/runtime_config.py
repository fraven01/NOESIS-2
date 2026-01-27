from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, model_validator


_FORBIDDEN_ID_FIELDS = {"tenant_id", "user_id", "case_id", "workflow_id"}


class RuntimeConfig(BaseModel):
    execution_scope: str | None = None
    budget_tokens: int | None = None
    timeouts_ms: int | None = None
    model_label: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def reject_id_fields(cls, data: object) -> object:
        if isinstance(data, Mapping):
            forbidden = sorted(key for key in _FORBIDDEN_ID_FIELDS if key in data)
            if forbidden:
                raise ValueError(
                    "RuntimeConfig must not include ID fields: " + ", ".join(forbidden)
                )
        return data


__all__ = ["RuntimeConfig"]
