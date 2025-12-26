"""Policy-aware DRF authentication classes.

This module provides DRF authentication classes that enforce account policies
(expiry, active status, etc.) for ALL authentication methods (Session, Basic, future OAuth/JWT).

Architecture:
- PolicyEnforcingMixin: Wraps authenticate() to run policy checks after authentication
- Policy-enforcing variants of DRF authentication classes
- Raises AuthenticationFailed if ANY policy is violated

Usage:
    # In settings.py:
    REST_FRAMEWORK = {
        "DEFAULT_AUTHENTICATION_CLASSES": [
            "profiles.authentication.PolicyEnforcingSessionAuthentication",
            "profiles.authentication.PolicyEnforcingBasicAuthentication",
        ],
    }

This ensures that expired EXTERNAL accounts cannot bypass expiry checks via API auth.
"""

from __future__ import annotations

import structlog
from rest_framework.authentication import (
    BasicAuthentication,
    SessionAuthentication,
)
from rest_framework.exceptions import AuthenticationFailed

from .policies import account_policy

logger = structlog.get_logger(__name__)


class PolicyEnforcingMixin:
    """Mixin that enforces account policies after DRF authentication succeeds.

    This mixin wraps the authenticate() method to:
    1. Run the parent authentication logic (Session, Basic, etc.)
    2. Enforce ALL account policies via AccountPolicyService
    3. Raise AuthenticationFailed if ANY policy is violated

    The policy service provides detailed logging for debugging.
    """

    def authenticate(self, request):
        """Authenticate and enforce policies.

        Returns:
            (user, auth) tuple if authentication + policies succeed
            None if authentication failed (no credentials, invalid, etc.)

        Raises:
            AuthenticationFailed: If authentication succeeds but policies violated
        """
        # Call parent authentication (SessionAuthentication, BasicAuthentication, etc.)
        result = super().authenticate(request)

        # Parent returned None = no credentials or invalid credentials
        if result is None:
            return None

        user, auth = result

        # Enforce ALL account policies
        if not account_policy.enforce(user):
            logger.warning(
                "authentication_blocked_by_policy",
                user_id=user.id,
                username=user.username,
                auth_class=self.__class__.__name__,
            )
            raise AuthenticationFailed("Account policy violation")

        # All policies passed
        return result


class PolicyEnforcingSessionAuthentication(PolicyEnforcingMixin, SessionAuthentication):
    """Session authentication with account policy enforcement.

    This class:
    1. Authenticates via Django sessions (cookies)
    2. Enforces account policies (expiry, active status, etc.)
    3. Works in conjunction with ExternalAccountExpiryMiddleware for early-exit optimization

    Note: Middleware provides early-exit for session auth BEFORE DRF runs, but this
    authentication class is the authoritative enforcement point for ALL auth methods.
    """

    pass


class PolicyEnforcingBasicAuthentication(PolicyEnforcingMixin, BasicAuthentication):
    """HTTP Basic authentication with account policy enforcement.

    This class:
    1. Authenticates via HTTP Basic Auth (username:password in Authorization header)
    2. Enforces account policies (expiry, active status, etc.)

    CRITICAL: Without policy enforcement, expired EXTERNAL accounts could bypass
    expiry checks by using Basic Auth instead of session auth.
    """

    pass
