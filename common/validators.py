from __future__ import annotations

from typing import Any, TypeVar

TSeq = TypeVar("TSeq", tuple[str, ...], list[str])


def require_trimmed_str(value: Any, *, field_name: str = "value") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{field_name} must not be empty")
    return candidate


def optional_str(value: Any, *, field_name: str = "value") -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    candidate = value.strip()
    return candidate or None


def normalise_str_sequence(
    value: Any,
    *,
    field_name: str = "value",
    error_message: str | None = None,
    return_type: type[TSeq] = tuple,
) -> TSeq:
    if error_message is None:
        error_message = f"{field_name} must be a sequence of strings"
    if value in (None, "", (), []):
        return return_type()
    if isinstance(value, str):
        candidate = value.strip()
        return return_type([candidate] if candidate else [])
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(error_message)
    cleaned: list[str] = []
    for item in value:
        if item in (None, ""):
            continue
        cleaned_item = str(item).strip()
        if cleaned_item:
            cleaned.append(cleaned_item)
    return return_type(cleaned)
