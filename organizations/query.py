from django.db import models

from .utils import current_organization


class OrganizationScopedQuerySet(models.QuerySet):
    """QuerySet that limits results to the active organization."""

    def __init__(self, *args, organization_field: str = "organization", **kwargs):
        super().__init__(*args, **kwargs)
        self.organization_field = organization_field

    def for_current(self):
        """Filter the queryset to the current organization."""
        org = current_organization()
        if org is None:
            return self.none()
        return self.filter(**{self.organization_field: org})


class OrganizationManager(models.Manager.from_queryset(OrganizationScopedQuerySet)):
    """Manager applying organization scoping to all queries."""

    def __init__(self, *args, organization_field: str = "organization", **kwargs):
        super().__init__(*args, **kwargs)
        self.organization_field = organization_field

    def get_queryset(self):
        qs = OrganizationScopedQuerySet(
            self.model, using=self._db, organization_field=self.organization_field
        )
        return qs.for_current()
