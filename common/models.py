from django.db import models


class TimestampedModel(models.Model):
    """Abstract base model with self-updating ``created_at`` and ``updated_at``.

    ``created_at`` is set when the object is created and ``updated_at`` is
    refreshed on each save.
    """

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class DomainPolicyOverride(TimestampedModel):
    """Tenant-specific overrides for domain preference and block lists."""

    tenant_id = models.CharField(max_length=255, unique=True)
    preferred_hosts = models.JSONField(default=list, blank=True)
    blocked_hosts = models.JSONField(default=list, blank=True)

    class Meta:
        indexes = [
            models.Index(
                fields=("tenant_id",), name="domain_policy_override_tenant_idx"
            )
        ]


class DomainPolicy(TimestampedModel):
    """Tenant-specific policy rules for boosting or rejecting domains."""

    class Action(models.TextChoices):
        BOOST = "boost", "Boost"
        REJECT = "reject", "Reject"

    tenant_id = models.CharField(max_length=255)
    domain = models.CharField(max_length=512)
    action = models.CharField(
        max_length=16, choices=Action.choices, default=Action.BOOST
    )
    priority = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("tenant_id", "domain"), name="domain_policy_unique_rule"
            )
        ]
        indexes = [
            models.Index(
                fields=("tenant_id", "action"), name="domain_policy_action_idx"
            )
        ]
