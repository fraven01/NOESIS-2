import json
from dataclasses import is_dataclass
from documents.parsers import ParsedResult, ParsedTextBlock
from ai_core.services import _dump_jsonable


def test_serialization():
    print("Testing ParsedResult serialization...")

    # Create a dummy ParsedResult
    block = ParsedTextBlock(text="Hello", kind="paragraph")
    result = ParsedResult(text_blocks=(block,))

    print(f"Is dataclass? {is_dataclass(result)}")
    print(f"Type: {type(result)}")

    try:
        safe = _dump_jsonable(result)
        print(f"Safe representation: {safe}")
        json_str = json.dumps(safe)
        print(f"JSON: {json_str}")
    except Exception as e:
        print(f"Serialization failed: {e}")


if __name__ == "__main__":
    test_serialization()
