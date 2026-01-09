from __future__ import annotations

from typing import Any

from pydantic_core import to_jsonable_python


def _jsonable_fallback(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_jsonable_python(
                dict(to_dict()),
                by_alias=False,
                bytes_mode="base64",
                serialize_unknown=True,
                fallback=str,
            )
        except Exception:
            return str(value)
    return str(value)


def to_jsonable(value: Any) -> Any:
    """Normalize values into JSON-friendly primitives."""
    return to_jsonable_python(
        value,
        by_alias=False,
        bytes_mode="base64",
        serialize_unknown=True,
        fallback=_jsonable_fallback,
    )
