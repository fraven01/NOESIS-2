"""Tests for chunking utils."""

from ai_core.rag.chunking.utils import split_sentences


def test_split_sentences_handles_abbreviations():
    text = "Dr. Smith went home. He slept."
    sentences = split_sentences(text)
    assert sentences == ["Dr. Smith went home.", "He slept."]


def test_split_sentences_handles_decimals():
    text = "Pi is 3.14. That is approximate."
    sentences = split_sentences(text)
    assert sentences == ["Pi is 3.14.", "That is approximate."]


def test_split_sentences_handles_urls():
    text = "Visit https://example.com/test.html for more. Next sentence."
    sentences = split_sentences(text)
    assert sentences == [
        "Visit https://example.com/test.html for more.",
        "Next sentence.",
    ]
