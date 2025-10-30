from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from common import object_store_defaults as _defaults

FilesystemObjectStore = _defaults.FilesystemObjectStore
put_bytes = _defaults.put_bytes
read_bytes = _defaults.read_bytes
read_json = _defaults.read_json
safe_filename = _defaults.safe_filename
sanitize_identifier = _defaults.sanitize_identifier
write_bytes = _defaults.write_bytes
write_json = _defaults.write_json


class _ObjectStoreModule(ModuleType):
    @property
    def BASE_PATH(self) -> Path:
        return _defaults.BASE_PATH

    @BASE_PATH.setter
    def BASE_PATH(self, value: Path) -> None:
        _defaults.BASE_PATH = value


# Expose BASE_PATH for static analyzers, then rely on the module property at runtime.
BASE_PATH: Path = _defaults.BASE_PATH
_module = sys.modules[__name__]
delattr(_module, "BASE_PATH")
_module.__class__ = _ObjectStoreModule


__all__ = [
    "BASE_PATH",
    "FilesystemObjectStore",
    "put_bytes",
    "read_bytes",
    "read_json",
    "safe_filename",
    "sanitize_identifier",
    "write_bytes",
    "write_json",
]
