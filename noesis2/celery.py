import os

from celery import Celery

from common.celery import ContextTask


settings_module = os.getenv("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)


app = Celery("noesis2")
app.Task = ContextTask
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
