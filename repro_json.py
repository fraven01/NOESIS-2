import json
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Mapping
from uuid import UUID, uuid4
from datetime import datetime, timezone
from pydantic import BaseModel


# Mock classes
class Metadata(BaseModel):
    id: UUID
    created_at: datetime


@dataclass
class Context:
    metadata: Metadata


@dataclass
class Artifact:
    context: Context
    stats: Mapping[str, Any]


def _serialize_artifact(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize_artifact(v) for k, v in asdict(obj).items()}
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (list, tuple)):
        return [_serialize_artifact(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize_artifact(v) for k, v in obj.items()}
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


try:
    m = Metadata(id=uuid4(), created_at=datetime.now(timezone.utc))
    c = Context(metadata=m)
    a = Artifact(context=c, stats={"foo": "bar"})

    print(f"Original: {a}")
    serialized = _serialize_artifact(a)
    print(f"Serialized: {serialized}")
    json_str = json.dumps(serialized)
    print(f"JSON: {json_str}")
    print("SUCCESS")
except Exception as e:
    print(f"FAILURE: {e}")
