"""Middleware for profile-related checks.

ARCHITECTURE: Layered Account Policy Enforcement
================================================

This middleware provides EARLY-EXIT optimization for session-authenticated users.
However, it is NOT the authoritative enforcement point.

Enforcement Layers:
1. **Middleware (this file)**: Early-exit for session auth BEFORE view processing
   - Runs BEFORE DRF authentication
   - Only sees request.user if session-authenticated (cookies)
   - Provides fast redirect for expired accounts
   - Does NOT protect API endpoints using BasicAuth, OAuth, JWT, etc.

2. **Authentication Classes** (profiles.authentication): Authoritative enforcement
   - PolicyEnforcingSessionAuthentication
   - PolicyEnforcingBasicAuthentication
   - Runs DURING DRF authentication
   - Enforces policies for ALL auth methods
   - See: profiles.policies.AccountPolicyService

3. **Policy Service** (profiles.policies): Single source of truth
   - AccountPolicyService.enforce(user) -> bool
   - Centralized logic for expiry, active status, future policies
   - Used by authentication classes, middleware, signals

Why this architecture?
- Middleware: Fast path for session users (no view processing)
- Auth classes: Comprehensive coverage (API + session)
- Policy service: DRY, testable, extensible

Security Note:
Without authentication class enforcement, expired EXTERNAL accounts could bypass
expiry checks by using HTTP Basic Auth instead of session cookies.
"""

from __future__ import annotations

import logging

from django.contrib.auth import logout
from django.shortcuts import redirect
from django.utils import timezone

logger = logging.getLogger(__name__)


class ExternalAccountExpiryMiddleware:
    """Force logout for expired external accounts (session auth early-exit optimization).

    This middleware provides a fast path for session-authenticated users BEFORE
    view processing. However, authentication classes provide authoritative enforcement
    for ALL auth methods (Session, Basic, OAuth, JWT).

    See module docstring for full architecture explanation.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, "user", None)
        if user and user.is_authenticated:
            try:
                profile = user.userprofile
                if (
                    profile.account_type == profile.AccountType.EXTERNAL
                    and profile.expires_at
                    and timezone.now() >= profile.expires_at
                ):
                    logger.info(
                        "External account expired, forcing logout",
                        extra={
                            "user_id": user.id,
                            "username": user.username,
                            "expires_at": profile.expires_at.isoformat(),
                        },
                    )
                    logout(request)
                    return redirect("/accounts/login/?expired=1")
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception(
                    "Error checking account expiry: %s",
                    exc,
                    extra={"user_id": user.id if user else None},
                )
        return self.get_response(request)
