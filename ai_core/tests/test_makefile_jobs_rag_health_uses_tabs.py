from pathlib import Path

import pytest


def test_makefile_jobs_rag_health_uses_tabs():
    makefile_path = Path(__file__).resolve().parents[2] / "Makefile"
    lines = makefile_path.read_text().splitlines()

    try:
        target_index = lines.index("jobs\\:rag\\:health:")
    except ValueError:  # pragma: no cover - guard against missing target
        pytest.fail("jobs:rag:health target not found in Makefile")

    for next_line in lines[target_index + 1 :]:
        if not next_line.strip():
            continue
        assert next_line.startswith("\t"), "jobs:rag:health recipe must start with a tab"
        break
    else:  # pragma: no cover - guard against missing recipe lines
        pytest.fail("jobs:rag:health has no recipe lines")
