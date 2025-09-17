from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A chunk of knowledge used for retrieval."""

    content: str
    meta: Dict[str, Any]
    """Metadata with keys: tenant, case, source, hash."""
    embedding: Optional[List[float]] = None
