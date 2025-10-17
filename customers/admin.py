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


def enable_tenant_aware_admin(site: admin.AdminSite) -> admin.AdminSite:
    """Mutate the existing admin site to be tenant aware."""

    if getattr(site, "_tenant_aware_enabled", False):
        return site

    dynamic_models = []

    original_register = site.register
    original_unregister = site.unregister
    original_each_context = site.each_context
    original_get_app_list = site.get_app_list
    original_admin_view = site.admin_view

    def register_dynamic(model, admin_class=None, **options):
        """Register models that should only appear on the public schema."""

        dynamic_models.append((model, admin_class, options))
        if is_public_schema():
            original_register(model, admin_class, **options)

    def ensure_dynamic_models():
        should_register = is_public_schema()
        for model, admin_class, options in dynamic_models:
            is_registered = model in site._registry
            if should_register and not is_registered:
                original_register(model, admin_class, **options)
            elif not should_register and is_registered:
                original_unregister(model)

    def each_context(request):
        ensure_dynamic_models()
        return original_each_context(request)

    def get_app_list(request, app_label=None):
        ensure_dynamic_models()
        return original_get_app_list(request, app_label)

    def admin_view(view, cacheable=False):
        view_func = original_admin_view(view, cacheable=cacheable)

        def wrapper(request, *args, **kwargs):
            ensure_dynamic_models()
            return view_func(request, *args, **kwargs)

        return update_wrapper(wrapper, view_func)

    site.register_dynamic = register_dynamic
    site._ensure_dynamic_models = ensure_dynamic_models
    site.each_context = each_context
    site.get_app_list = get_app_list
    site.admin_view = admin_view
    site._tenant_aware_dynamic_models = dynamic_models
    site._tenant_aware_enabled = True

    return site


tenant_admin_site = enable_tenant_aware_admin(admin.site)
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
