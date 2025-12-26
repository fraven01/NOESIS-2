from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

from profiles.models import UserProfile


class User(AbstractUser):
    pass


class Invitation(models.Model):
    email = models.EmailField()
    role = models.CharField(max_length=20, choices=UserProfile.Roles.choices)
    account_type = models.CharField(
        max_length=20,
        choices=UserProfile.AccountType.choices,
        default=UserProfile.AccountType.INTERNAL,
    )
    token = models.CharField(max_length=64, unique=True, blank=True, null=True)
    invitation_expires_at = models.DateTimeField(
        blank=True,
        null=True,
        help_text="When this invitation expires (not user expiry)",
    )
    user_expires_at = models.DateTimeField(
        blank=True,
        null=True,
        help_text="When the created user account expires (for EXTERNAL accounts)",
    )
    accepted_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def generate_token(self):
        """Generate a token and expiry for this invitation."""
        import secrets

        self.token = secrets.token_urlsafe(16)
        self.invitation_expires_at = timezone.now() + timezone.timedelta(days=7)
        return self.token
