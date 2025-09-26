from importlib import import_module


info_intake = import_module(".info_intake", __name__)
needs_mapping = import_module(".needs_mapping", __name__)
scope_check = import_module(".scope_check", __name__)
system_description = import_module(".system_description", __name__)

__all__ = [
    "info_intake",
    "needs_mapping",
    "scope_check",
    "system_description",
]
