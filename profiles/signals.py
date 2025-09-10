from django.conf import settings
from django.db import connection
from django.db.models.signals import post_save
from django.dispatch import receiver

from .services import ensure_user_profile


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_user_profile(sender, instance, created, **kwargs):
    if not created or connection.schema_name == settings.PUBLIC_SCHEMA_NAME:
        return
    ensure_user_profile(instance)
