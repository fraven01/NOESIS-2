"""Base utilities for chaos test suites."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.chaos


class ChaosTestCase:
    """Base class for chaos test suites.

    Inherit from this class to automatically mark tests with ``@pytest.mark.chaos``
    and to document the expectation that the ``chaos_env`` fixture is used to
    manipulate fault-injection toggles.
    """

    pytestmark = pytest.mark.chaos
