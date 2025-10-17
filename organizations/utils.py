import contextlib
import threading
from typing import Optional

from .models import Organization

_thread_locals = threading.local()


def current_organization(request: Optional[object] = None) -> Optional[Organization]:
    """Return the active :class:`~organizations.models.Organization`.

    The helper checks the given ``request`` for an ``organization`` attribute.
    Outside of request/response cycles the active organization can be set using
    :func:`set_current_organization` which stores the value in thread-local
    storage.

    Usage::

        from organizations.utils import set_current_organization
        with set_current_organization(my_org):
            some_service.fetch_for_current_org()  # automatically scoped to ``my_org``
    """
    if request is not None:
        return getattr(request, "organization", None)
    return getattr(_thread_locals, "organization", None)


@contextlib.contextmanager
def set_current_organization(org: Optional[Organization]):
    """Context manager to temporarily set the current organization."""
    previous = getattr(_thread_locals, "organization", None)
    _thread_locals.organization = org
    try:
        yield
    finally:
        if previous is None:
            if hasattr(_thread_locals, "organization"):
                delattr(_thread_locals, "organization")
        else:
            _thread_locals.organization = previous
