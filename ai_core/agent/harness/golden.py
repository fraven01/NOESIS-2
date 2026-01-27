from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GoldenQueryInput(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=50)

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GoldenQuery(BaseModel):
    id: str
    flow_name: str
    input: GoldenQueryInput
    notes: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


_DEFAULT_PATH = Path(__file__).with_name("golden_queries_v0.json")


def load_golden_queries(path: str | Path = _DEFAULT_PATH) -> list[GoldenQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden query dataset must be a list")
    return [GoldenQuery.model_validate(item) for item in data]


__all__ = ["GoldenQuery", "GoldenQueryInput", "load_golden_queries"]
