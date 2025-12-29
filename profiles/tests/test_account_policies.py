"""Tests for USER-GLOBAL account policy enforcement across all authentication methods.

IMPORTANT: These tests validate USER-GLOBAL policies (not tenant-specific).
- Account expiry applies to ALL tenants
- Active status applies to ALL tenants
- Policies enforced at authentication time (before tenant context)

This test suite validates that account policies (expiry, active status) are
enforced consistently across:
- Session authentication (cookies)
- Basic authentication (HTTP Authorization header)
- Direct policy service calls

For tenant-specific authorization tests, see cases/tests/test_authz.py

Architecture:
- Policy Service: profiles.policies.AccountPolicyService (user-global)
- Authentication Classes: profiles.authentication.PolicyEnforcing* (all auth methods)
- Middleware: profiles.middleware.ExternalAccountExpiryMiddleware (session early-exit)
"""

from datetime import timedelta

import pytest
from django.test import Client
from django.utils import timezone
from rest_framework.test import APIClient

from profiles.models import UserProfile
from profiles.policies import account_policy
from users.tests.factories import UserFactory


# =============================================================================
# POLICY SERVICE TESTS (Direct)
# =============================================================================


@pytest.mark.django_db
class TestAccountPolicyService:
    """Test AccountPolicyService directly (unit tests)."""

    def test_check_expiry_external_expired(self):
        """Expired EXTERNAL account violates policy."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() - timedelta(days=1),
        )

        result = account_policy.check_expiry(user)

        assert result.violated is True
        assert "expired" in result.reason.lower()
        assert result.action == "deny"
        assert "expires_at" in result.metadata

    def test_check_expiry_external_not_expired(self):
        """Unexpired EXTERNAL account passes policy."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() + timedelta(days=7),
        )

        result = account_policy.check_expiry(user)

        assert result.violated is False
        assert result.action == "allow"

    def test_check_expiry_external_no_expiry(self):
        """EXTERNAL account without expires_at passes policy."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=None,
        )

        result = account_policy.check_expiry(user)

        assert result.violated is False
        assert result.action == "allow"

    def test_check_expiry_internal_account_ignores_expiry(self):
        """INTERNAL accounts ignore expires_at field."""
        user = UserFactory(
            account_type=UserProfile.AccountType.INTERNAL,
            expires_at=timezone.now() - timedelta(days=1),  # Past!
        )

        result = account_policy.check_expiry(user)

        # INTERNAL accounts don't check expiry
        assert result.violated is False
        assert result.action == "allow"

    def test_check_is_active_user_inactive(self):
        """Inactive user violates policy."""
        user = UserFactory(is_active=False)

        result = account_policy.check_is_active(user)

        assert result.violated is True
        assert "inactive" in result.reason.lower()
        assert result.action == "deny"

    def test_check_is_active_profile_inactive(self):
        """Inactive profile violates policy."""
        user = UserFactory(profile_is_active=False)

        result = account_policy.check_is_active(user)

        assert result.violated is True
        assert "inactive" in result.reason.lower()
        assert result.action == "deny"

    def test_check_is_active_both_active(self):
        """Active user and profile pass policy."""
        user = UserFactory(is_active=True, profile_is_active=True)

        result = account_policy.check_is_active(user)

        assert result.violated is False
        assert result.action == "allow"

    def test_enforce_all_policies_pass(self):
        """enforce() returns True when all policies pass."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() + timedelta(days=7),
            is_active=True,
            profile_is_active=True,
        )

        assert account_policy.enforce(user) is True

    def test_enforce_fails_on_expiry(self):
        """enforce() returns False when expiry policy violated."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() - timedelta(days=1),
            is_active=True,
            profile_is_active=True,
        )

        assert account_policy.enforce(user) is False

    def test_enforce_fails_on_inactive_user(self):
        """enforce() returns False when user inactive."""
        user = UserFactory(
            is_active=False,
            profile_is_active=True,
        )

        assert account_policy.enforce(user) is False

    def test_enforce_fails_on_inactive_profile(self):
        """enforce() returns False when profile inactive."""
        user = UserFactory(
            is_active=True,
            profile_is_active=False,
        )

        assert account_policy.enforce(user) is False


# =============================================================================
# SESSION AUTHENTICATION TESTS (Middleware + Auth Class)
# =============================================================================


@pytest.mark.django_db
class TestSessionAuthenticationPolicies:
    """Test session authentication with policy enforcement."""

    def test_session_blocks_expired_external(self, client: Client):
        """Session auth blocks expired EXTERNAL accounts via middleware."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() - timedelta(days=1),
        )
        client.force_login(user)

        # Access any protected view
        response = client.get("/cases/")

        # Middleware redirects to login with expired flag
        assert response.status_code == 302
        assert "/accounts/login/" in response.url
        assert "expired=1" in response.url

    def test_session_allows_unexpired_external(self, client: Client):
        """Session auth allows unexpired EXTERNAL accounts."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() + timedelta(days=7),
        )
        client.force_login(user)

        response = client.get("/cases/")

        # Should get to the view (200 or 403 depending on permissions)
        # NOT redirected to login
        assert response.status_code != 302 or "/accounts/login/" not in response.url

    def test_session_allows_internal_accounts(self, client: Client):
        """Session auth allows INTERNAL accounts (no expiry)."""
        user = UserFactory(account_type=UserProfile.AccountType.INTERNAL)
        client.force_login(user)

        response = client.get("/cases/")

        # Should get to the view
        assert response.status_code != 302 or "/accounts/login/" not in response.url

    def test_session_blocks_inactive_user(self, client: Client):
        """Session auth blocks inactive users."""
        user = UserFactory(is_active=False)
        client.force_login(user)

        # Django's authentication backend should deny inactive users
        response = client.get("/cases/")

        # Either redirected to login or 403 forbidden
        assert response.status_code in (302, 403)


# =============================================================================
# BASIC AUTHENTICATION TESTS (API with Authorization header)
# =============================================================================


@pytest.mark.django_db
class TestBasicAuthenticationPolicies:
    """Test Basic authentication with policy enforcement.

    CRITICAL: These tests verify the security fix for expired EXTERNAL accounts
    bypassing expiry checks when using HTTP Basic Auth instead of session auth.
    """

    def test_basic_blocks_expired_external(self):
        """Basic auth blocks expired EXTERNAL accounts (SECURITY FIX)."""
        user = UserFactory(
            username="test@example.com",
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() - timedelta(days=1),
        )
        user.set_password("test123")
        user.save()

        api_client = APIClient()
        api_client.credentials(
            HTTP_AUTHORIZATION=f"Basic {self._encode_credentials('test@example.com', 'test123')}"
        )

        # Try to access API endpoint
        response = api_client.get("/cases/")

        # Should be denied with 401 Unauthorized or 403 Forbidden
        assert response.status_code in (401, 403)

    def test_basic_allows_unexpired_external(self):
        """Basic auth allows unexpired EXTERNAL accounts."""
        user = UserFactory(
            username="test@example.com",
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() + timedelta(days=7),
        )
        user.set_password("test123")
        user.save()

        api_client = APIClient()
        api_client.credentials(
            HTTP_AUTHORIZATION=f"Basic {self._encode_credentials('test@example.com', 'test123')}"
        )

        response = api_client.get("/cases/")

        # Should be authenticated (200, 403 for permissions, 404, etc.)
        # NOT 401 (authentication failed)
        assert response.status_code != 401

    def test_basic_allows_internal_accounts(self):
        """Basic auth allows INTERNAL accounts (no expiry)."""
        user = UserFactory(
            username="test@example.com",
            account_type=UserProfile.AccountType.INTERNAL,
        )
        user.set_password("test123")
        user.save()

        api_client = APIClient()
        api_client.credentials(
            HTTP_AUTHORIZATION=f"Basic {self._encode_credentials('test@example.com', 'test123')}"
        )

        response = api_client.get("/cases/")

        # Should be authenticated
        assert response.status_code != 401

    def test_basic_blocks_inactive_user(self):
        """Basic auth blocks inactive users."""
        user = UserFactory(
            username="test@example.com",
            is_active=False,
        )
        user.set_password("test123")
        user.save()

        api_client = APIClient()
        api_client.credentials(
            HTTP_AUTHORIZATION=f"Basic {self._encode_credentials('test@example.com', 'test123')}"
        )

        response = api_client.get("/cases/")

        # Should be denied
        assert response.status_code in (401, 403)

    def test_basic_blocks_inactive_profile(self):
        """Basic auth blocks users with inactive profiles."""
        user = UserFactory(
            username="test@example.com",
            is_active=True,
            profile_is_active=False,
        )
        user.set_password("test123")
        user.save()

        api_client = APIClient()
        api_client.credentials(
            HTTP_AUTHORIZATION=f"Basic {self._encode_credentials('test@example.com', 'test123')}"
        )

        response = api_client.get("/cases/")

        # Should be denied
        assert response.status_code in (401, 403)

    @staticmethod
    def _encode_credentials(username: str, password: str) -> str:
        """Encode credentials for HTTP Basic Auth."""
        import base64

        credentials = f"{username}:{password}"
        return base64.b64encode(credentials.encode()).decode()


# =============================================================================
# EDGE CASES & ERROR HANDLING
# =============================================================================


@pytest.mark.django_db
class TestPolicyEdgeCases:
    """Test edge cases and error handling."""

    def test_check_expiry_with_custom_time(self):
        """check_expiry() accepts custom current_time for testing."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() + timedelta(days=7),
        )

        # Check with future time (should be expired)
        future_time = timezone.now() + timedelta(days=14)
        result = account_policy.check_expiry(user, current_time=future_time)

        assert result.violated is True

    def test_check_expiry_with_past_time(self):
        """check_expiry() with past time doesn't affect result."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
            expires_at=timezone.now() - timedelta(days=7),
        )

        # Check with past time (still expired)
        past_time = timezone.now() - timedelta(days=14)
        result = account_policy.check_expiry(user, current_time=past_time)

        assert result.violated is False  # Not expired at that past time

    def test_enforce_without_userprofile(self):
        """enforce() handles missing UserProfile gracefully."""
        from users.models import User

        # Create user and then delete profile to simulate edge case
        user = User.objects.create_user(username="noprofile", password="test123")

        # Delete the profile (if auto-created via signal)
        if hasattr(user, "userprofile"):
            user.userprofile.delete()
            # Clear cached property
            if hasattr(user, "_userprofile_cache"):
                delattr(user, "_userprofile_cache")

        # Refresh to ensure Django doesn't cache the relation
        user.refresh_from_db()

        # Should fail-closed (deny access)
        result = account_policy.enforce(user)

        assert result is False
