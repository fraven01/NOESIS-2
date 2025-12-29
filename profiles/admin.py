from django.contrib import admin

from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "role",
        "account_type",
        "is_active",
        "expires_at",
        "created_at",
    )
    list_filter = ("role", "account_type", "is_active")
    list_select_related = ("user",)
    readonly_fields = ("created_at",)
