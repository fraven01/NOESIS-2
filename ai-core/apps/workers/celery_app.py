"""Celery application with predefined queues."""

from __future__ import annotations

import os

from celery import Celery
from kombu import Queue


app = Celery(
    "ai_core",
    broker=os.getenv("CELERY_BROKER_URL", "memory://"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "rpc://"),
)

app.conf.task_queues = [
    Queue("ingest_raw"),
    Queue("extract_text"),
    Queue("pii_mask"),
    Queue("chunk"),
    Queue("embed"),
    Queue("upsert"),
]

# Execute tasks synchronously for this placeholder setup.
app.conf.task_always_eager = True


# Import tasks so Celery registers them.
from .tasks import (  # noqa: E402
    ingest_raw,  # noqa: F401
    extract_text,  # noqa: F401
    pii_mask,  # noqa: F401
    chunk,  # noqa: F401
    embed,  # noqa: F401
    upsert,  # noqa: F401
)
