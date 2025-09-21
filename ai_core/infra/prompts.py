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

    valid_candidates = []
    for candidate in candidates:
        match = re.search(r"\.v(\d+)\.md$", candidate.name)
        if not match:
            continue
        valid_candidates.append((int(match.group(1)), match.group(1), candidate))

    if not valid_candidates:
        invalid_files = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"No valid prompt filename found for alias '{alias}'. Candidates: {invalid_files}"
        )

    _, version_str, prompt_file = max(valid_candidates, key=lambda item: item[0])
    version = f"v{version_str}"
    text = prompt_file.read_text(encoding="utf-8")
    return {"version": version, "text": text}
