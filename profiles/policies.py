"""Centralized account policy enforcement.

This module provides a single source of truth for all account-level policies
(expiry, IP restrictions, concurrent sessions, etc.). All authentication methods
(Session, Basic, OAuth, JWT) should enforce policies through this service.

IMPORTANT: Account Policies vs. Authorization
=============================================
Account Policies (this module):
- USER-GLOBAL: Apply to the user across ALL tenants
- Examples: account expiry, active status, rate limits
- A user with expired account cannot access ANY tenant
- Enforced at authentication time (before tenant context)

Tenant-Specific Authorization (cases/authz.py):
- TENANT-SCOPED: Different rules per tenant
- Examples: role-based access, case permissions
- A user may have different roles in different tenants
- Enforced at view/object level (within tenant context)

Rationale:
- UserProfile is a shared model (not tenant-scoped)
- One User = One UserProfile = Global account status
- Tenant-specific access rules are handled separately

Architecture:
- PolicyViolation: Dataclass representing policy check results
- AccountPolicyService: Singleton service that runs all policy checks
- Used by: Authentication classes, middleware, signal handlers
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import structlog
from django.utils import timezone

if TYPE_CHECKING:
    from django.contrib.auth.models import User

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PolicyViolation:
    """Result of an account policy check.

    Attributes:
        violated: Whether the policy was violated
        reason: Human-readable explanation (empty if not violated)
        action: Enforcement action - "deny", "warn", "log", "allow"
        metadata: Additional context for logging/debugging
    """

    violated: bool
    reason: str
    action: str  # "deny", "warn", "log", "allow"
    metadata: dict


class AccountPolicyService:
    """Centralized service for all USER-GLOBAL account-level policies.

    IMPORTANT: All policies are USER-GLOBAL (not tenant-specific).
    - A user with expired account cannot access ANY tenant
    - Active/inactive status applies across ALL tenants
    - For tenant-specific authorization, see cases/authz.py

    This service provides a single point of enforcement for all account policies:
    - Expiry checks for EXTERNAL accounts (global expiry)
    - Active status checks (user.is_active, profile.is_active)
    - (Future) Concurrent session limits
    - (Future) IP allowlist/blocklist
    - (Future) Geographic restrictions
    - (Future) Rate limiting

    Usage:
        from profiles.policies import account_policy

        # In authentication class:
        if not account_policy.enforce(user):
            raise AuthenticationFailed('Account policy violation')

        # In view for detailed checks:
        result = account_policy.check_expiry(user)
        if result.violated:
            return Response({"error": result.reason}, status=403)
    """

    def check_expiry(
        self, user: User, current_time: datetime | None = None
    ) -> PolicyViolation:
        """Check if EXTERNAL account has expired.

        Args:
            user: User to check
            current_time: Override current time (for testing)

        Returns:
            PolicyViolation with violated=True if account expired
        """
        if current_time is None:
            current_time = timezone.now()

        try:
            profile = user.userprofile

            # Only EXTERNAL accounts have expiry
            if (
                profile.account_type == profile.AccountType.EXTERNAL
                and profile.expires_at
                and current_time >= profile.expires_at
            ):
                logger.info(
                    "account_expired",
                    user_id=user.id,
                    username=user.username,
                    expires_at=profile.expires_at.isoformat(),
                    account_type=profile.account_type,
                    check_time=current_time.isoformat(),
                )

                return PolicyViolation(
                    violated=True,
                    reason=f"Account expired at {profile.expires_at.isoformat()}",
                    action="deny",
                    metadata={
                        "expires_at": profile.expires_at.isoformat(),
                        "account_type": profile.account_type,
                        "check_time": current_time.isoformat(),
                    },
                )

        except Exception as exc:
            logger.exception(
                "policy_check_failed",
                error=str(exc),
                user_id=getattr(user, "id", None),
            )
            # On error, deny access (fail-closed)
            return PolicyViolation(
                violated=True,
                reason=f"Policy check failed: {exc}",
                action="deny",
                metadata={"error": str(exc)},
            )

        # No violation
        return PolicyViolation(violated=False, reason="", action="allow", metadata={})

    def check_is_active(self, user: User) -> PolicyViolation:
        """Check if user and profile are active.

        Args:
            user: User to check

        Returns:
            PolicyViolation with violated=True if inactive
        """
        try:
            if not user.is_active:
                logger.info(
                    "user_inactive",
                    user_id=user.id,
                    username=user.username,
                )
                return PolicyViolation(
                    violated=True,
                    reason="User account is inactive",
                    action="deny",
                    metadata={"user_is_active": False},
                )

            profile = user.userprofile
            if not profile.is_active:
                logger.info(
                    "profile_inactive",
                    user_id=user.id,
                    username=user.username,
                    account_type=profile.account_type,
                )
                return PolicyViolation(
                    violated=True,
                    reason="User profile is inactive",
                    action="deny",
                    metadata={
                        "profile_is_active": False,
                        "account_type": profile.account_type,
                    },
                )

        except Exception as exc:
            logger.exception(
                "policy_check_failed",
                error=str(exc),
                user_id=getattr(user, "id", None),
            )
            # On error, deny access (fail-closed)
            return PolicyViolation(
                violated=True,
                reason=f"Policy check failed: {exc}",
                action="deny",
                metadata={"error": str(exc)},
            )

        # No violation
        return PolicyViolation(violated=False, reason="", action="allow", metadata={})

    def enforce(self, user: User) -> bool:
        """Run all policy checks and return overall result.

        This is the main entry point for authentication classes.
        Returns False if ANY policy is violated.

        Args:
            user: User to check

        Returns:
            True if all policies pass, False if any violation
        """
        # Check 1: Is user/profile active?
        active_result = self.check_is_active(user)
        if active_result.violated:
            return False

        # Check 2: Expiry (EXTERNAL accounts only)
        expiry_result = self.check_expiry(user)
        if expiry_result.violated:
            return False

        # Future checks:
        # - concurrent session limits
        # - IP allowlist/blocklist
        # - geographic restrictions
        # - rate limiting
        # - time-based access windows

        return True


# Singleton instance - import this in authentication classes
account_policy = AccountPolicyService()
