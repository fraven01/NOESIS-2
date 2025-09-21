from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Dict


@lru_cache(maxsize=None)
def load(alias: str) -> Dict[str, str]:
    """Load a prompt by alias.

    The alias maps to a markdown file under ``ai_core/prompts`` where
    files are versioned ``<name>.v<version>.md``. The function returns the
    prompt text along with its version string (e.g. ``"v1"``).
    """
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    base = prompts_dir / alias
    candidates = sorted(base.parent.glob(f"{base.name}.v*.md"))
    if not candidates:
        raise FileNotFoundError(f"No prompt for alias '{alias}'")

    prompt_file = candidates[-1]
    match = re.search(r"\.v(\d+)\.md$", prompt_file.name)
    if not match:
        raise ValueError(f"Invalid prompt filename: {prompt_file.name}")
    version = f"v{match.group(1)}"
    text = prompt_file.read_text(encoding="utf-8")
    return {"version": version, "text": text}
