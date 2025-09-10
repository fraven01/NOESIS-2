from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone

from profiles.models import UserProfile


class User(AbstractUser):
    pass


class Invitation(models.Model):
    email = models.EmailField()
    role = models.CharField(max_length=20, choices=UserProfile.Roles.choices)
    token = models.CharField(max_length=64, unique=True, blank=True, null=True)
    expires_at = models.DateTimeField(blank=True, null=True)
    accepted_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def generate_token(self):
        """Generate a token and expiry for this invitation."""
        import secrets

        self.token = secrets.token_urlsafe(16)
        self.expires_at = timezone.now() + timezone.timedelta(days=7)
        return self.token
