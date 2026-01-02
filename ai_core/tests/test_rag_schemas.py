import pytest
from pydantic import ValidationError

from ai_core.rag.schemas import Chunk


def test_chunk_frozen_config_prevents_field_reassignment():
    """Verify frozen=True blocks assignment to chunk fields."""
    chunk = Chunk(content="Test", meta={"id": "doc-1"})

    with pytest.raises(ValidationError):
        chunk.meta = {"id": "doc-2"}

    with pytest.raises(ValidationError):
        chunk.content = "Modified"


def test_chunk_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Chunk(content="Test", meta={"id": "doc-1"}, extra_field="nope")
