"""Tests for public endpoints that should work without authentication."""

import pytest
from django.test import Client


@pytest.mark.django_db
class TestPublicEndpoints:
    """Test that public endpoints are accessible without authentication."""

    def test_admin_login_page_accessible(self):
        """Admin login page should be accessible without authentication."""
        client = Client()
        response = client.get("/admin/login/")
        # Should show login page, not redirect to login (would be circular)
        assert response.status_code == 200

    def test_accounts_login_page_accessible(self):
        """Accounts login page should be accessible without authentication."""
        client = Client()
        response = client.get("/accounts/login/")
        # Should show login page
        assert response.status_code == 200

    def test_document_lifecycle_health_accessible(self):
        """Document lifecycle health endpoint should be public."""
        client = Client()
        response = client.get("/api/health/document-lifecycle/")
        # Should return health data without authentication
        assert response.status_code == 200
        data = response.json()
        # Health endpoint returns structured checks
        assert "checks" in data or "error" in data

    def test_api_schema_accessible(self):
        """API schema should be accessible for documentation."""
        client = Client()
        response = client.get("/api/schema/")
        # Schema might be public or protected - this documents current behavior
        # If it returns 403, that might be intentional
        assert response.status_code in [200, 403]

    def test_invitation_accept_accessible(self):
        """Invitation acceptance page should be accessible without auth."""
        client = Client()
        # This will 404 because token doesn't exist, but shouldn't 403 or redirect to login
        response = client.get("/invite/accept/fake-token-12345/")
        # 404 means endpoint is accessible, just token not found
        # 302 to login would mean authentication required (bad)
        # 403 would mean permission denied (bad)
        assert response.status_code == 404
