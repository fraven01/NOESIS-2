from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, HttpResponse
from django.utils.decorators import method_decorator

from .models import OrgMembership
from .utils import current_organization


ViewFunc = Callable[..., HttpResponse]


def require_org_member(view_func: ViewFunc) -> ViewFunc:
    """Ensure the user is a member of the active organization.

    This guard verifies that ``request.user`` has a membership for the
    organization returned by :func:`~organizations.utils.current_organization`.
    It can be applied to function-based views as a decorator or to
    class-based views using ``RequireOrgMember``::

        @require_org_member
        def my_view(request):
            ...

        class MyView(RequireOrgMember, View):
            ...

    The guard applies to *all* HTTP methods, including read-only requests.
    If no active organization exists or the user is not a member, a
    :class:`django.core.exceptions.PermissionDenied` exception is raised.
    """

    @wraps(view_func)
    def _wrapped_view(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        org = current_organization()
        if (
            org is None
            or not OrgMembership.objects.filter(
                organization=org, user=request.user
            ).exists()
        ):
            raise PermissionDenied("User is not a member of the active organization")
        return view_func(request, *args, **kwargs)

    return _wrapped_view


class RequireOrgMember:
    """Mixin applying :func:`require_org_member` to class-based views."""

    @method_decorator(require_org_member)
    def dispatch(self, *args: Any, **kwargs: Any) -> HttpResponse:
        return super().dispatch(*args, **kwargs)
