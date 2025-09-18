from __future__ import annotations


def mask(text: str) -> str:
    """Placeholder implementation for PII masking."""

    return "".join("X" if ch.isdigit() else ch for ch in text)


def mask_prompt(text: str, *, placeholder_only: bool = False) -> str:
    """Mask prompt text and optionally collapse it to a redaction placeholder."""

    masked = mask(text)
    if not masked:
        return masked
    if placeholder_only:
        return "XXXX"
    if "XXXX" in masked:
        return masked
    return f"{masked}\n\nXXXX"
