"""Test-only Django app to prepare DB extensions for pytest.

This app is conditionally enabled only when running tests (TESTING=True)
so it never affects production or development environments.
"""
