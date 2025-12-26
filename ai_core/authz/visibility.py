"""Visibility guards for Retrieval-Augmented Generation endpoints."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest

from ai_core.rag.visibility import Visibility, normalize_visibility
from organizations.models import OrgMembership
from organizations.utils import current_organization
from profiles.models import UserProfile

_P = ParamSpec("_P")
_R = TypeVar("_R")
_EXTENDED_VISIBILITY = {Visibility.ALL, Visibility.DELETED}


def _extract_internal_key(request: HttpRequest) -> str | None:
    if hasattr(request, "headers"):
        key = request.headers.get("X-Internal-Key")
        if key:
            return key
    return request.META.get("HTTP_X_INTERNAL_KEY")


def allow_extended_visibility(request: HttpRequest) -> bool:
    """Return whether the request may access soft-deleted content."""

    internal_key = _extract_internal_key(request)
    allowed_keys = getattr(settings, "RAG_INTERNAL_KEYS", ()) or ()
    if internal_key and internal_key in allowed_keys:
        return True

    user = getattr(request, "user", None)
    if user is None or not getattr(user, "is_authenticated", False):
        return False

    try:
        profile = user.userprofile
    except UserProfile.DoesNotExist:
        return False

    if not profile.is_active:
        return False

    if profile.role == UserProfile.Roles.TENANT_ADMIN:
        return True

    organization = current_organization(request)
    if organization is None:
        return False

    return OrgMembership.objects.filter(
        organization=organization,
        user=user,
        role=OrgMembership.Role.ADMIN,
    ).exists()


def resolve_effective_visibility(
    request: HttpRequest, visibility: object | None
) -> Visibility:
    """Normalise *visibility* for service usage respecting guard rules."""

    normalized, _ = normalize_visibility(visibility)
    if normalized in _EXTENDED_VISIBILITY and not allow_extended_visibility(request):
        return Visibility.ACTIVE
    return normalized


def enforce_visibility_permission(
    request: HttpRequest, visibility: object | None
) -> Visibility:
    """Ensure *request* may use *visibility* and raise otherwise."""

    normalized, _ = normalize_visibility(visibility)
    if normalized in _EXTENDED_VISIBILITY and not allow_extended_visibility(request):
        raise PermissionDenied("Extended visibility not permitted for this request")
    return normalized


def require_extended_visibility(
    func: Callable[_P, _R] | None = None,
    *,
    param: str = "visibility",
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator rejecting unauthorised extended visibility requests."""

    def _decorator(view_func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(view_func)
        def _wrapped(request: HttpRequest, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            requested = kwargs.get(param)
            if requested is None and hasattr(request, "GET"):
                requested = request.GET.get(param)
            enforce_visibility_permission(request, requested)
            return view_func(request, *args, **kwargs)

        return _wrapped

    if func is not None:
        return _decorator(func)
    return _decorator


__all__ = [
    "allow_extended_visibility",
    "enforce_visibility_permission",
    "require_extended_visibility",
    "resolve_effective_visibility",
]
