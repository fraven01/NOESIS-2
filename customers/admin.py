from django.contrib import admin
from django.core.management import call_command

from .models import Domain, Tenant


@admin.action(description="Migrate selected tenants")
def migrate_selected_tenants(modeladmin, request, queryset):
    for tenant in queryset:
        call_command("migrate_schemas", schema=tenant.schema_name)


@admin.register(Tenant)
class TenantAdmin(admin.ModelAdmin):
    list_display = ("schema_name", "name", "paid_until", "on_trial", "created_on")
    actions = [migrate_selected_tenants]


@admin.register(Domain)
class DomainAdmin(admin.ModelAdmin):
    list_display = ("domain", "tenant", "is_primary")
    list_select_related = ("tenant",)
