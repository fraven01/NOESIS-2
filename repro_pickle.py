import copy
from dataclasses import dataclass
from typing import Mapping, Any, Tuple


@dataclass(frozen=True)
class DocumentParseArtifact:
    text_blocks: Tuple[Mapping[str, Any], ...]
    statistics: Mapping[str, Any]

    def __post_init__(self) -> None:
        # No MappingProxyType anymore
        object.__setattr__(self, "statistics", dict(self.statistics))


try:
    d = {"a": 1}
    a = DocumentParseArtifact(text_blocks=(), statistics=d)
    print(f"Artifact: {a}")
    a_copied = copy.deepcopy(a)
    print(f"Artifact Copied: {a_copied}")
    print("SUCCESS")
except Exception as e:
    print(f"FAILURE: {e}")
