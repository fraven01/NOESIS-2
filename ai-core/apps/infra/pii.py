"""PII masking stub that marks tokens as [PII]."""

from __future__ import annotations

import re


def mask(text: str) -> str:
    """Replace alphanumeric tokens with ``[PII]`` placeholders."""
    return re.sub(r"\w+", "[PII]", text)
