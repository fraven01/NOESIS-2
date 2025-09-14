from __future__ import annotations


def mask(text: str) -> str:
    """Placeholder implementation for PII masking.

    Currently replaces any digit with ``X`` as a simplistic stand-in for a real
    anonymisation routine.
    """

    return "".join("X" if ch.isdigit() else ch for ch in text)
