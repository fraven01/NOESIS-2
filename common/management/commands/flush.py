"""Override Django's flush command to allow cascading by default."""

from django.core.management.commands.flush import Command as DjangoFlushCommand


class Command(DjangoFlushCommand):
    """Flush database tables while cascading FKs (required for shared tenants)."""

    def handle(self, *args, **options):
        options = dict(options)
        options.setdefault("allow_cascade", True)
        return super().handle(*args, **options)
