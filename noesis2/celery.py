import os
from celery import Celery


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")


app = Celery("noesis2")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
