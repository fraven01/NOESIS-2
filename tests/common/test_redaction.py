from common.redaction import MASK, Redactor


def test_timestamp_fields_are_not_redacted() -> None:
    redactor = Redactor()
    event = {
        "timestamp": "2025-09-25T21:53:40Z",
        "@timestamp": "2025-09-25T21:53:40Z",
    }

    result = redactor(None, "", event)

    assert result["timestamp"] == "2025-09-25T21:53:40Z"
    assert result["@timestamp"] == "2025-09-25T21:53:40Z"


def test_german_phone_numbers_are_masked() -> None:
    redactor = Redactor()
    event = {
        "contact_international": "+49 30 1234567",
        "contact_national": "030-1234567",
    }

    result = redactor(None, "", event)

    assert result["contact_international"] == MASK
    assert result["contact_national"] == MASK


def test_iso_datetime_values_are_not_masked() -> None:
    redactor = Redactor()
    event = {
        "message": "2025-09-25T21:53:40Z",
    }

    result = redactor(None, "", event)

    assert result["message"] == "2025-09-25T21:53:40Z"


def test_iso_datetime_embedded_in_text_is_not_masked() -> None:
    redactor = Redactor()
    event = {
        "message": "Termin am 2025-09-25T21:53:40Z",
    }

    result = redactor(None, "", event)

    assert result["message"] == "Termin am 2025-09-25T21:53:40Z"


def test_german_phone_embedded_in_text_is_masked() -> None:
    redactor = Redactor()
    event = {
        "message": "Ruf mich an: +49 30 1234567 oder 030-1234567",
    }

    result = redactor(None, "", event)

    assert result["message"] == f"Ruf mich an: {MASK} oder {MASK}"
