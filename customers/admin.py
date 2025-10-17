from functools import update_wrapper

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


class TenantAwareAdminSite(admin.AdminSite):
    """Admin site that registers tenant models only on the public schema."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamic_models = []

    def register_dynamic(self, model, admin_class=None, **options):
        """Register models that should only appear on the public schema."""

        self._dynamic_models.append((model, admin_class, options))
        if is_public_schema():
            super().register(model, admin_class, **options)

    def _ensure_dynamic_models(self):
        should_register = is_public_schema()
        for model, admin_class, options in self._dynamic_models:
            is_registered = model in self._registry
            if should_register and not is_registered:
                super().register(model, admin_class, **options)
            elif not should_register and is_registered:
                super().unregister(model)

    def each_context(self, request):
        self._ensure_dynamic_models()
        return super().each_context(request)

    def get_app_list(self, request, app_label=None):
        self._ensure_dynamic_models()
        return super().get_app_list(request, app_label)

    def admin_view(self, view, cacheable=False):
        def wrapper(request, *args, **kwargs):
            self._ensure_dynamic_models()
            return super(TenantAwareAdminSite, self).admin_view(view, cacheable=cacheable)(
                request, *args, **kwargs
            )

        return update_wrapper(wrapper, view)


tenant_admin_site = TenantAwareAdminSite()
admin.site = tenant_admin_site
admin.sites.site = tenant_admin_site


@admin.action(description="Migrate selected tenants")
def migrate_selected_tenants(modeladmin, request, queryset):
    for tenant in queryset:
        call_command("migrate_schemas", schema=tenant.schema_name)


class TenantAdmin(admin.ModelAdmin):
    list_display = ("schema_name", "name", "paid_until", "on_trial", "created_on")
    actions = [migrate_selected_tenants]


class DomainAdmin(admin.ModelAdmin):
    list_display = ("domain", "tenant", "is_primary")
    list_select_related = ("tenant",)


tenant_admin_site.register_dynamic(Tenant, TenantAdmin)
tenant_admin_site.register_dynamic(Domain, DomainAdmin)
