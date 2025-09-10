import pytest
from django.utils import timezone

from profiles.models import UserProfile
from users.tests.factories import InvitationFactory, UserFactory


@pytest.mark.django_db
def test_accept_invitation_updates_profile_and_marks_invitation(client):
    user = UserFactory()
    invitation = InvitationFactory(role=UserProfile.Roles.MANAGER)
    client.force_login(user)
    response = client.get(f"/invite/accept/{invitation.token}/")
    assert response.status_code == 302
    invitation.refresh_from_db()
    assert invitation.accepted_at is not None
    profile = UserProfile.objects.get(user=user)
    assert profile.role == UserProfile.Roles.MANAGER
    assert profile.is_active


@pytest.mark.django_db
def test_accept_invitation_expired_token_returns_404(client):
    user = UserFactory()
    invitation = InvitationFactory(
        expires_at=timezone.now() - timezone.timedelta(days=1)
    )
    client.force_login(user)
    response = client.get(f"/invite/accept/{invitation.token}/")
    assert response.status_code == 404


@pytest.mark.django_db
def test_accept_invitation_cannot_be_used_twice(client):
    user = UserFactory()
    invitation = InvitationFactory()
    client.force_login(user)
    first = client.get(f"/invite/accept/{invitation.token}/")
    assert first.status_code == 302
    second = client.get(f"/invite/accept/{invitation.token}/")
    assert second.status_code == 404
