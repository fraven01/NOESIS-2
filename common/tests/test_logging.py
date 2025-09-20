import logging

import pytest

from common import logging as common_logging


@pytest.fixture(autouse=True)
def _clear_log_context():
    common_logging.clear_log_context()
    yield
    common_logging.clear_log_context()


def _build_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=0, msg="hello", args=(), exc_info=None
    )


def test_filter_masks_values_when_masking_enabled(settings):
    settings.LOGGING_ALLOW_UNMASKED_CONTEXT = False
    filt = common_logging.RequestTaskContextFilter()

    with common_logging.log_context(
        trace_id="trace-abcdef", case_id="case-12345", tenant="tenant-42", key_alias="alias-secret"
    ):
        record = _build_record()
        assert filt.filter(record) is True

    assert record.trace_id.startswith("tr")
    assert record.trace_id.endswith("ef")
    assert "***" in record.trace_id
    assert record.case_id != "case-12345"
    assert record.tenant != "tenant-42"
    assert record.key_alias != "alias-secret"


def test_filter_allows_unmasked_values_when_opted_in(settings):
    settings.LOGGING_ALLOW_UNMASKED_CONTEXT = True
    filt = common_logging.RequestTaskContextFilter()

    with common_logging.log_context(
        trace_id="trace-xyz", case_id="case-xyz", tenant="tenant-xyz", key_alias="alias-xyz"
    ):
        record = _build_record()
        filt.filter(record)

    assert record.trace_id == "trace-xyz"
    assert record.case_id == "case-xyz"
    assert record.tenant == "tenant-xyz"
    assert record.key_alias == "alias-xyz"


def test_filter_sets_placeholders_when_context_missing(settings):
    settings.LOGGING_ALLOW_UNMASKED_CONTEXT = False
    record = _build_record()
    filt = common_logging.RequestTaskContextFilter()

    filt.filter(record)

    assert record.trace_id == "-"
    assert record.case_id == "-"
    assert record.tenant == "-"
    assert record.key_alias == "-"


def test_logging_configuration_includes_context_fields(settings):
    verbose_fmt = settings.LOGGING["formatters"]["verbose"]["format"]
    json_fmt = settings.LOGGING["formatters"]["json"]["fmt"]

    for field in ("trace_id", "case_id", "tenant", "key_alias"):
        placeholder = f"%({field})s"
        assert placeholder in verbose_fmt
        assert placeholder in json_fmt


def test_handlers_apply_request_context_filter(settings):
    handler_filters = settings.LOGGING["handlers"]["console"]["filters"]
    json_filters = settings.LOGGING["handlers"]["json_console"]["filters"]

    assert "request_task_context" in handler_filters
    assert "request_task_context" in json_filters
