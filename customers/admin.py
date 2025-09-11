from django.contrib import admin
from django.core.management import call_command
from django.db import connection
from django_tenants.utils import get_public_schema_name

from .models import Domain, Tenant


def is_public_schema() -> bool:
    try:
        return connection.schema_name == get_public_schema_name()
    except Exception:
        return False


@admin.action(description="Migrate selected tenants")
def migrate_selected_tenants(modeladmin, request, queryset):
    for tenant in queryset:
        call_command("migrate_schemas", schema=tenant.schema_name)


if is_public_schema():

    @admin.register(Tenant)
    class TenantAdmin(admin.ModelAdmin):
        list_display = ("schema_name", "name", "paid_until", "on_trial", "created_on")
        actions = [migrate_selected_tenants]

    @admin.register(Domain)
    class DomainAdmin(admin.ModelAdmin):
        list_display = ("domain", "tenant", "is_primary")
        list_select_related = ("tenant",)
