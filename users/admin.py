import secrets

from django.contrib import admin, messages
from django.utils import timezone

from .models import Invitation


@admin.register(Invitation)
class InvitationAdmin(admin.ModelAdmin):
    list_display = ("email", "role", "token", "expires_at", "accepted_at", "created_at")
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
            if not invitation.expires_at:
                invitation.expires_at = timezone.now() + timezone.timedelta(days=7)
            invitation.save()
            print(
                f"Stub mail to {invitation.email}: /invite/accept/{invitation.token}/"
            )
            self.message_user(
                request,
                f"/invite/accept/{invitation.token}/",
                level=messages.INFO,
            )
