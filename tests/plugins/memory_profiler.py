"""Simple pytest plugin to profile per-test memory usage.

Enable with: pytest --mem-profile

It records:
- delta of `resource.getrusage(RUSAGE_SELF).ru_maxrss` around each test (process peak RSS increase)
- delta of tracemalloc peak during each test (Python allocations)

Notes:
- ru_maxrss behaviour is platform-dependent: on Linux it's in KB; we convert to MB.
- This plugin is intentionally lightweight and imports only stdlib modules (tracemalloc, resource).
"""

from __future__ import annotations

import time
import tracemalloc
import resource
from typing import Any

import pytest


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--mem-profile",
        action="store_true",
        help="Enable per-test memory profiling (tracemalloc + ru_maxrss)",
    )


def pytest_configure(config: Any) -> None:
    enabled = bool(config.getoption("--mem-profile"))
    config._mem_profile_enabled = enabled
    if enabled:
        config._mem_results = []  # type: ignore[attr-defined]
        # Start tracemalloc to track Python allocations
        tracemalloc.start()


@pytest.fixture(autouse=True)
def _mem_profile_fixture(request: Any, pytestconfig: Any):
    """Autouse fixture that records memory stats around each test when enabled."""
    if not getattr(pytestconfig, "_mem_profile_enabled", False):
        yield
        return

    nodeid = request.node.nodeid
    start_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_tracemem_curr, start_tracemem_peak = tracemalloc.get_traced_memory()
    t0 = time.time()

    yield

    duration = time.time() - t0
    end_ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    end_tracemem_curr, end_tracemem_peak = tracemalloc.get_traced_memory()

    # ru_maxrss is platform dependent (kilobytes on Linux); convert to MB
    delta_ru = max(0, end_ru - start_ru) / 1024.0
    delta_tracemem_mb = max(0, end_tracemem_peak - start_tracemem_peak) / (
        1024.0 * 1024.0
    )

    pytestconfig._mem_results.append(
        {
            "nodeid": nodeid,
            "rss_mb": delta_ru,
            "tracemalloc_mb": delta_tracemem_mb,
            "duration_s": duration,
        }
    )


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # noqa: ARG001
    config = session.config
    if not getattr(config, "_mem_profile_enabled", False):
        return

    results = sorted(config._mem_results, key=lambda r: r["rss_mb"], reverse=True)
    tr = config.pluginmanager.getplugin("terminalreporter")

    tr.section("Memory profile (top 25 by RSS delta)", sep="=")
    tr.line("nodeid | rss_mb | tracemalloc_mb | duration_s")
    for r in results[:25]:
        tr.line(
            f"{r['nodeid']} | {r['rss_mb']:.2f}MB | {r['tracemalloc_mb']:.2f}MB | {r['duration_s']:.2f}s"
        )

    results_t = sorted(
        config._mem_results, key=lambda r: r["tracemalloc_mb"], reverse=True
    )
    tr.section("Memory profile (top 25 by tracemalloc peak delta)", sep="=")
    tr.line("nodeid | tracemalloc_mb | rss_mb | duration_s")
    for r in results_t[:25]:
        tr.line(
            f"{r['nodeid']} | {r['tracemalloc_mb']:.2f}MB | {r['rss_mb']:.2f}MB | {r['duration_s']:.2f}s"
        )

    tracemalloc.stop()
