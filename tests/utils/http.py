"""HTTP request helpers for integration and chaos tests.

The utilities encapsulate the tenant header contracts described in the
API reference so that tests remain aligned with production traffic.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Mapping, MutableMapping
from uuid import uuid4

from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
)

# Canonical example values mirrored from docs/api/reference.md.
DEFAULT_TENANT_SCHEMA = "acme_prod"
DEFAULT_TENANT_ID = "acme"
DEFAULT_CASE_ID = "crm-7421"


@lru_cache(maxsize=None)
def _cached_uuid(test_id: str) -> str:
    """Return a cached UUID v4 value for a given test identifier."""

    return str(uuid4())


def stable_idempotency_key(test_id: str) -> str:
    """Generate a stable UUID v4 idempotency key for the given test.

    Parameters
    ----------
    test_id:
        Unique identifier of the invoking test (``request.node.nodeid`` is a
        suitable value). Each identifier receives a lazily generated UUID v4
        that is cached for the duration of the test run, ensuring repeated
        calls remain idempotent.
    """

    if not test_id:
        raise ValueError("test_id must be a non-empty string for stable keys")
    return _cached_uuid(test_id)


def apply_idempotency_header(
    headers: MutableMapping[str, str], *, test_id: str, idempotency_key: str | None = None
) -> str:
    """Populate the idempotency header on the provided mapping.

    The helper fills ``Idempotency-Key`` with a stable UUID when ``idempotency_key``
    is not provided. The computed key is returned so tests can embed it into
    payload metadata or assertions.
    """

    key = idempotency_key or stable_idempotency_key(test_id)
    headers[IDEMPOTENCY_KEY_HEADER] = key
    return key


def build_request_headers(
    *,
    tenant_schema: str = DEFAULT_TENANT_SCHEMA,
    tenant_id: str = DEFAULT_TENANT_ID,
    case_id: str = DEFAULT_CASE_ID,
    test_id: str,
    idempotency_key: str | None = None,
    key_alias: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """Construct a fully populated tenant header dictionary for requests.

    Parameters mirror the header contracts outlined in ``docs/api/reference.md``.
    Tests should pass their ``request.node.nodeid`` as ``test_id`` to obtain a
    reproducible UUID v4 idempotency key across retries and repeated helper
    calls within the same scenario.
    """

    headers: Dict[str, str] = {
        X_TENANT_SCHEMA_HEADER: tenant_schema,
        X_TENANT_ID_HEADER: tenant_id,
        X_CASE_ID_HEADER: case_id,
    }
    apply_idempotency_header(headers, test_id=test_id, idempotency_key=idempotency_key)
    if key_alias:
        headers[X_KEY_ALIAS_HEADER] = key_alias
    if extra:
        headers.update(extra)
    return headers

