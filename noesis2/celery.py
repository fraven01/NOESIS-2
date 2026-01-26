import os
from collections.abc import Mapping

from celery import Celery
from celery.signals import setup_logging, task_received
from structlog.contextvars import bind_contextvars, clear_contextvars

from common.celery import ScopedTask
from common.constants import X_TRACE_ID_HEADER
from common.logging import (
    bind_log_context,
    clear_log_context,
    configure_logging,
    get_logger,
)


settings_module = os.getenv("DJANGO_SETTINGS_MODULE", "noesis2.settings.development")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)


app = Celery("noesis2")
app.Task = ScopedTask
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

logger = get_logger(__name__)


@setup_logging.connect
def _configure_celery_logging(*_args, **_kwargs) -> None:
    configure_logging()


@task_received.connect
def _log_task_received(request=None, **_kwargs) -> None:
    if request is None:
        return

    def _candidate_headers(value: object) -> list[Mapping[str, object]]:
        candidates: list[Mapping[str, object]] = []
        if isinstance(value, Mapping):
            candidates.append(value)
        return candidates

    def _extract_headers(obj: object) -> list[Mapping[str, object]]:
        if obj is None:
            return []
        headers_list: list[Mapping[str, object]] = []

        direct = getattr(obj, "headers", None)
        headers_list.extend(_candidate_headers(direct))

        if isinstance(obj, Mapping):
            headers_list.extend(_candidate_headers(obj.get("headers")))

        request_dict = getattr(obj, "request_dict", None)
        if request_dict is None and isinstance(obj, Mapping):
            request_dict = obj.get("request_dict") or obj.get("_request_dict")
        if isinstance(request_dict, Mapping):
            headers_list.extend(_candidate_headers(request_dict.get("headers")))
            headers_list.extend(_candidate_headers(request_dict.get("properties")))

        message = getattr(obj, "_message", None)
        if message is None and isinstance(obj, Mapping):
            message = obj.get("_message")
        if message is not None:
            headers_list.extend(_candidate_headers(getattr(message, "headers", None)))
            headers_list.extend(
                _candidate_headers(getattr(message, "properties", None))
            )
            headers_list.extend(
                _candidate_headers(getattr(message, "delivery_info", None))
            )

        properties = getattr(obj, "properties", None)
        if properties is None and isinstance(obj, Mapping):
            properties = obj.get("properties")
        if isinstance(properties, Mapping):
            headers_list.extend(_candidate_headers(properties.get("headers")))
            headers_list.extend(
                _candidate_headers(properties.get("application_headers"))
            )

        delivery_info = getattr(obj, "delivery_info", None)
        if delivery_info is None and isinstance(obj, Mapping):
            delivery_info = obj.get("delivery_info")
        if isinstance(delivery_info, Mapping):
            headers_list.extend(_candidate_headers(delivery_info.get("headers")))

        return headers_list

    def _extract_trace_id(obj: object) -> str | None:
        for headers in _extract_headers(obj):
            lowered = {str(key).lower(): value for key, value in headers.items()}
            trace_id = lowered.get(X_TRACE_ID_HEADER.lower()) or lowered.get(
                "x-trace-id"
            )
            if trace_id is None:
                trace_id = lowered.get("trace_id")
            if trace_id:
                return str(trace_id)
        return None

    trace_id = _extract_trace_id(request)
    if trace_id is None:
        trace_id = _extract_trace_id(_kwargs.get("message"))
    if trace_id is None:
        trace_id = _extract_trace_id(_kwargs.get("headers"))
    debug_sources = [
        type(entry).__name__ for entry in _extract_headers(request) if entry
    ]
    try:
        if trace_id:
            bind_contextvars(trace_id=trace_id)
            bind_log_context(trace_id=trace_id)
        logger.debug(
            "celery.task.received.debug",
            task_id=getattr(request, "id", None),
            task_name=getattr(request, "name", None),
            trace_id=trace_id,
            header_sources=debug_sources,
        )
        logger.info(
            "celery.task.received",
            task_id=getattr(request, "id", None),
            task_name=getattr(request, "name", None),
        )
    finally:
        clear_log_context()
        clear_contextvars()
