import copy
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class DocumentPipelineConfig:
    caption_min_confidence_by_collection: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "caption_min_confidence_by_collection",
            dict(self.caption_min_confidence_by_collection),
        )


try:
    c = DocumentPipelineConfig(caption_min_confidence_by_collection={"a": 0.5})
    print(f"Config: {c}")
    c_copied = copy.deepcopy(c)
    print(f"Config Copied: {c_copied}")
    print("SUCCESS")
except Exception as e:
    print(f"FAILURE: {e}")
