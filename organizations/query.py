from django.db import models

from .utils import current_organization


class OrganizationScopedQuerySet(models.QuerySet):
    """QuerySet that limits results to the active organization."""

    def __init__(self, *args, organization_field: str = "organization", **kwargs):
        super().__init__(*args, **kwargs)
        self.organization_field = organization_field

    def for_current(self, require: bool = True):
        """Filter the queryset to the current organization.

        - If ``require`` is True (default) and no current organization is set,
          return an empty queryset.
        - If ``require`` is False and no organization is set, leave unscoped.
        """
        org = current_organization()
        if org is None:
            return self if not require else self.none()
        return self.filter(**{self.organization_field: org})


class OrganizationManager(models.Manager.from_queryset(OrganizationScopedQuerySet)):
    """Manager applying organization scoping to all queries."""

    def __init__(
        self,
        *args,
        organization_field: str = "organization",
        require_org: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.organization_field = organization_field
        self.require_org = require_org

    def get_queryset(self):
        qs = OrganizationScopedQuerySet(
            self.model, using=self._db, organization_field=self.organization_field
        )
        require = getattr(self.model, "ORGANIZATION_REQUIRED", self.require_org)
        return qs.for_current(require=require)
