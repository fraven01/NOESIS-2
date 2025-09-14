from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Chunk:
    """A chunk of knowledge used for retrieval."""

    content: str
    meta: Dict[str, Any]
    """Metadata with keys: tenant, case, source, hash."""
