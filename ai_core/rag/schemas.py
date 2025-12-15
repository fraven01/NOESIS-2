from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A chunk of knowledge used for retrieval."""

    content: str
    meta: Dict[str, Any]
    """Metadata including tenant_id, case_id, source, hash and related fields."""
    embedding: Optional[List[float]] = None
    parents: Optional[Dict[str, Dict[str, Any]]] = None
    """Optional mapping of parent node identifiers to their metadata payloads."""

