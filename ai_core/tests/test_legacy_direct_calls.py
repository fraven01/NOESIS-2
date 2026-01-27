from __future__ import annotations

from pathlib import Path
import re

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_allowlist() -> set[str]:
    allowlist_path = _repo_root() / "roadmap" / "legacy_direct_entrypoints.yaml"
    data = yaml.safe_load(allowlist_path.read_text(encoding="utf-8")) or []
    return {str(entry["path"]).replace("\\", "/") for entry in data}


def _scan_direct_call_paths() -> set[str]:
    root = _repo_root()
    targets = [
        root / "ai_core" / "services",
        root / "ai_core" / "graphs",
    ]
    pattern = re.compile(
        r"\b(run_retrieval_augmented_generation\(|build_graph\(\)\.run|_graph\.run|graph\.run)"
    )
    matches: set[str] = set()
    for base in targets:
        for path in base.rglob("*.py"):
            if path.name.endswith("_README.md") or path.name.endswith(".md"):
                continue
            content = path.read_text(encoding="utf-8")
            if pattern.search(content):
                rel_path = path.relative_to(root).as_posix()
                matches.add(rel_path)
    return matches


def test_direct_graph_calls_are_allowlisted():
    allowlist = _load_allowlist()
    found = _scan_direct_call_paths()

    missing = sorted(found - allowlist)
    assert not missing, f"Direct graph calls missing allowlist entries: {missing}"
