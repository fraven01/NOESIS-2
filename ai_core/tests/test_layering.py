"""Layering tests to enforce import direction between graph packages."""

from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_TECHNICAL = _ROOT / "ai_core" / "graphs" / "technical"
_BUSINESS = _ROOT / "ai_core" / "graphs" / "business"


def _iter_python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if path.is_file()]


def _find_forbidden_imports(path: Path, forbidden_prefix: str) -> list[str]:
    violations: list[str] = []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(forbidden_prefix):
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(forbidden_prefix):
                violations.append(node.module)

    return violations


def test_technical_graphs_do_not_import_business_graphs() -> None:
    """Technical graphs must not import business graphs."""
    forbidden = "ai_core.graphs.business"
    violations: dict[str, list[str]] = {}

    for path in _iter_python_files(_TECHNICAL):
        found = _find_forbidden_imports(path, forbidden)
        if found:
            violations[str(path)] = found

    assert not violations, (
        "Technical graph imports business graph modules, violating direction:\n"
        + "\n".join(
            f"{path}: {', '.join(sorted(set(modules)))}"
            for path, modules in sorted(violations.items())
        )
    )
