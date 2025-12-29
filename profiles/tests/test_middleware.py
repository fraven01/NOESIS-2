"""Tests for profiles middleware."""

from datetime import timedelta
from unittest.mock import Mock

import pytest
from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest, HttpResponse
from django.utils import timezone

from profiles.middleware import ExternalAccountExpiryMiddleware
from profiles.models import UserProfile
from users.tests.factories import UserFactory


@pytest.fixture
def get_response():
    """Mock get_response callable."""

    def _get_response(request):
        return HttpResponse("OK")

    return _get_response


@pytest.fixture
def middleware(get_response):
    """Create middleware instance."""
    return ExternalAccountExpiryMiddleware(get_response)


@pytest.fixture
def mock_request():
    """Create a mock request."""
    request = Mock(spec=HttpRequest)
    request.path = "/some/path/"
    request.META = {}
    return request


@pytest.mark.django_db
class TestExternalAccountExpiryMiddleware:
    """Tests for ExternalAccountExpiryMiddleware."""

    def test_allows_unauthenticated_users(self, middleware, mock_request):
        """Middleware allows unauthenticated users through."""
        mock_request.user = AnonymousUser()
        response = middleware(mock_request)
        assert response.status_code == 200
        assert response.content == b"OK"

    def test_allows_internal_accounts(self, middleware, mock_request):
        """Middleware allows INTERNAL accounts through."""
        user = UserFactory(account_type=UserProfile.AccountType.INTERNAL)
        mock_request.user = user
        response = middleware(mock_request)
        assert response.status_code == 200
        assert response.content == b"OK"

    def test_allows_external_without_expiry(self, middleware, mock_request):
        """Middleware allows EXTERNAL accounts without expires_at."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
        )
        user.userprofile.expires_at = None
        user.userprofile.save()
        mock_request.user = user
        response = middleware(mock_request)
        assert response.status_code == 200
        assert response.content == b"OK"

    def test_allows_external_not_yet_expired(self, middleware, mock_request):
        """Middleware allows EXTERNAL accounts not yet expired."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
        )
        user.userprofile.expires_at = timezone.now() + timedelta(days=7)
        user.userprofile.save()
        mock_request.user = user
        response = middleware(mock_request)
        assert response.status_code == 200
        assert response.content == b"OK"

    def test_redirects_expired_external_account(self, client):
        """Middleware redirects expired EXTERNAL accounts to login."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
        )
        # Set expiry to 1 day ago
        user.userprofile.expires_at = timezone.now() - timedelta(days=1)
        user.userprofile.save()

        # Use Django test client with session support
        client.force_login(user)
        # Use an endpoint that requires authentication (not AllowAny)
        response = client.get("/cases/", follow=False)

        # Should redirect to accounts login with expired parameter
        assert response.status_code == 302
        assert "/accounts/login/" in response.url
        assert "expired=1" in response.url

    def test_expiry_check_at_exact_expiry_time(self, client):
        """Test behavior at exact expiry time (should be expired)."""
        user = UserFactory(
            account_type=UserProfile.AccountType.EXTERNAL,
        )
        # Set expiry to 1 second ago (definitely expired)
        user.userprofile.expires_at = timezone.now() - timedelta(seconds=1)
        user.userprofile.save()

        # Use Django test client with session support
        client.force_login(user)
        response = client.get("/cases/", follow=False)

        # Should redirect (< check means now is expired)
        assert response.status_code == 302
        assert "/accounts/login/" in response.url
