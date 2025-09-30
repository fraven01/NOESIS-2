import pytest

from ai_core.rag.normalization import normalise_text


@pytest.mark.parametrize("value", [None, "", "   "])
def test_normalise_text_handles_empty_values(value: str | None) -> None:
    assert normalise_text(value) == ""


def test_normalise_text_normalises_whitespace_and_case() -> None:
    text = "  Kunden   ÜberBlick  "

    assert normalise_text(text) == "kund überblick"


def test_normalise_text_handles_umlauts_and_eszett() -> None:
    text = "Äpfel Straße"

    assert normalise_text(text) == "äpfel straß"


def test_normalise_text_plural_heuristic_variants() -> None:
    text = "Kunden Kunde Kundenen"

    normalised = normalise_text(text)

    assert normalised.split(" ") == ["kund", "kund", "kunden"]
