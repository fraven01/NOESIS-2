"""Configuration container for AI Core."""

from __future__ import annotations

from pathlib import Path


class Settings:
    PROJECT_NAME = "ai-core"
    APP_DIR = Path(__file__).resolve().parents[1]
    PROMPTS_DIR = APP_DIR / "prompts"


settings = Settings()
