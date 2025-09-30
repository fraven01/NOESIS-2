from __future__ import annotations

from pathlib import Path


def test_api_reference_describes_fused_scores() -> None:
    reference = Path("docs/api/reference.md").read_text(encoding="utf-8")
    lower = reference.lower()

    assert "fused" in lower or "fusion" in lower
    assert "cosine" in lower
    assert "1 / (1 + distance)" not in reference
