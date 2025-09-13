"""Ensure worker logs contain masked content only."""

import logging
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from apps.workers.tasks.extract_text import extract_text


def test_extract_text_logs_are_masked(caplog):
    payload = {"data": "John Doe"}
    with caplog.at_level(logging.INFO):
        extract_text(payload)
    assert "John" not in caplog.text
    assert "Doe" not in caplog.text
    assert "[PII]" in caplog.text
