from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


def _strip_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = _strip_text(value).lower()
    if not token:
        return default
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


class DocumentSpaceQueryParams(BaseModel):
    """Validate document space query params."""

    collection: str | None = None
    limit: int = 25
    latest: bool = True
    show_retired: bool = False
    q: str = ""
    cursor: str = ""
    workflow: str = ""

    model_config = ConfigDict(extra="ignore")

    @field_validator("collection", mode="before")
    @classmethod
    def _parse_collection(cls, value: object) -> str | None:
        text = _strip_text(value)
        return text or None

    @field_validator("q", "cursor", "workflow", mode="before")
    @classmethod
    def _parse_text_fields(cls, value: object) -> str:
        return _strip_text(value)

    @field_validator("limit", mode="before")
    @classmethod
    def _parse_limit(cls, value: object) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = 25
        return max(5, min(200, numeric))

    @field_validator("latest", mode="before")
    @classmethod
    def _parse_latest(cls, value: object) -> bool:
        return _parse_bool(value, default=True)

    @field_validator("show_retired", mode="before")
    @classmethod
    def _parse_show_retired(cls, value: object) -> bool:
        return _parse_bool(value, default=False)


class SearchQualityParams(BaseModel):
    """Validate RAG quality tuning params."""

    quality_mode: str = "standard"
    max_candidates: int = 20
    purpose: str | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("quality_mode", mode="before")
    @classmethod
    def _parse_quality_mode(cls, value: object) -> str:
        candidate = _strip_text(value).lower()
        if candidate in {"standard", "premium", "fast"}:
            return candidate
        return "standard"

    @field_validator("max_candidates", mode="before")
    @classmethod
    def _parse_max_candidates(cls, value: object) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = 20
        return max(5, min(40, numeric))

    @field_validator("purpose", mode="before")
    @classmethod
    def _parse_purpose(cls, value: object) -> str | None:
        text = _strip_text(value)
        return text or None
