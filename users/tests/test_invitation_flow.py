import pytest
from django.utils import timezone

from profiles.models import UserProfile
from users.tests.factories import InvitationFactory, UserFactory


@pytest.mark.django_db
def test_accept_invitation_updates_profile_and_marks_invitation(client):
    user = UserFactory()
    invitation = InvitationFactory(role=UserProfile.Roles.MANAGEMENT, email=user.email)
    client.force_login(user)
    response = client.get(f"/invite/accept/{invitation.token}/")
    assert response.status_code == 302
    invitation.refresh_from_db()
    assert invitation.accepted_at is not None
    profile = UserProfile.objects.get(user=user)
    assert profile.role == UserProfile.Roles.MANAGEMENT
    assert profile.is_active


@pytest.mark.django_db
def test_accept_invitation_expired_token_returns_403(client):
    """Expired invitations return 403 Forbidden (not 404)."""
    user = UserFactory()
    invitation = InvitationFactory(
        invitation_expires_at=timezone.now() - timezone.timedelta(days=1),
        email=user.email,
    )
    client.force_login(user)
    response = client.get(f"/invite/accept/{invitation.token}/")
    assert response.status_code == 403


@pytest.mark.django_db
def test_accept_invitation_cannot_be_used_twice(client):
    """Already accepted invitations return 403 Forbidden."""
    user = UserFactory()
    invitation = InvitationFactory(email=user.email)
    client.force_login(user)
    first = client.get(f"/invite/accept/{invitation.token}/")
    assert first.status_code == 302
    second = client.get(f"/invite/accept/{invitation.token}/")
    assert second.status_code == 403
