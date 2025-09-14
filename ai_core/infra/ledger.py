from __future__ import annotations

import json
import sys
from typing import Any


def record(meta: dict[str, Any]) -> None:
    """Log metadata as a JSON line to stdout."""

    print(json.dumps(meta), file=sys.stdout)
