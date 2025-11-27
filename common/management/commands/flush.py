"""Override Django's flush command to allow cascading by default."""

import logging

from django.core.management.commands.flush import Command as DjangoFlushCommand


LOGGER = logging.getLogger(__name__)


class Command(DjangoFlushCommand):
    """Flush database tables while cascading FKs (required for shared tenants)."""

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--no-cascade-log",
            action="store_true",
            help="Skip emitting the cascade-enabled log entry during flush.",
        )

    def handle(self, *args, **options):
        options = dict(options)
        log_cascade = not options.pop("no_cascade_log", False)
        options["allow_cascade"] = True
        allow_cascade = options["allow_cascade"]
        if log_cascade:
            LOGGER.info("flushing database (allow_cascade=%s)", allow_cascade)
        return super().handle(*args, **options)
