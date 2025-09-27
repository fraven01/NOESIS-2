"""Bootstrap registration for legacy graph runners."""

from __future__ import annotations

from ai_core.graphs import info_intake, needs_mapping, scope_check, system_description

from .adapters import module_runner
from .registry import register


def bootstrap() -> None:
    """Register the current set of graph runners with the registry.

    The bootstrap process is idempotent because the underlying registry overwrites
    existing entries on repeated registrations.
    """

    register("info_intake", module_runner(info_intake))
    register("scope_check", module_runner(scope_check))
    register("needs_mapping", module_runner(needs_mapping))
    register("system_description", module_runner(system_description))
