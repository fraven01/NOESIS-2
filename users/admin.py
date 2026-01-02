import logging
import secrets

from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils import timezone

from .models import Invitation, User

logger = logging.getLogger(__name__)


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """User admin with autocomplete support."""

    search_fields = ["username", "email", "first_name", "last_name"]


@admin.register(Invitation)
class InvitationAdmin(admin.ModelAdmin):
    list_display = (
        "email",
        "role",
        "account_type",
        "token",
        "invitation_expires_at",
        "user_expires_at",
        "accepted_at",
        "created_at",
    )
    list_filter = ("role", "account_type", "accepted_at")
    actions = ["invite_guest"]

    @admin.action(description="Gast einladen")
    def invite_guest(self, request, queryset):
        for invitation in queryset:
            if invitation.accepted_at:
                self.message_user(
                    request,
                    f"{invitation.email} bereits angenommen",
                    level=messages.WARNING,
                )
                continue
            if not invitation.token:
                invitation.token = secrets.token_urlsafe(16)
            if not invitation.invitation_expires_at:
                invitation.invitation_expires_at = timezone.now() + timezone.timedelta(
                    days=7
                )
            invitation.save()
            logger.info(
                "invitation.stub_mail",
                extra={
                    "email": invitation.email,
                    "token": invitation.token,
                    "path": f"/invite/accept/{invitation.token}/",
                },
            )
            self.message_user(
                request,
                f"/invite/accept/{invitation.token}/",
                level=messages.INFO,
            )
