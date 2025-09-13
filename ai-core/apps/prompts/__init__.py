"""Prompt loading utilities."""

from __future__ import annotations

from ..infra.config import settings


def load_prompt(path_alias: str) -> dict:
    """Return prompt version and text for the given alias."""
    base = settings.PROMPTS_DIR / path_alias
    files = sorted(base.parent.glob(f"{base.name}.v*.md"))
    if not files:
        raise FileNotFoundError(f"prompt not found for alias {path_alias}")
    file_path = files[-1]
    version = file_path.stem.split(".")[-1]
    return {"version": version, "text": file_path.read_text()}
