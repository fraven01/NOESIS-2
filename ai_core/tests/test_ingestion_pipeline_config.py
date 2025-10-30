import pytest

from ai_core.ingestion import _build_document_pipeline_config
from ai_core.tools import InputError


def test_pipeline_config_applies_meta_overrides(settings):
    settings.DOCUMENT_PIPELINE_ENABLE_OCR = False
    settings.DOCUMENT_PIPELINE_CAPTION_MIN_CONFIDENCE_DEFAULT = 0.4

    meta = {
        "pipeline_config": {
            "enable_ocr": True,
            "caption_min_confidence_default": 0.9,
        }
    }

    config = _build_document_pipeline_config(meta=meta)

    assert config.enable_ocr is True
    assert config.caption_min_confidence_default == pytest.approx(0.9)


def test_pipeline_config_prefers_meta_over_state(settings):
    settings.DOCUMENT_PIPELINE_ENABLE_OCR = False

    state = {"pipeline_config": {"enable_ocr": False}}
    meta = {"pipeline_config": {"enable_ocr": True}}

    config = _build_document_pipeline_config(state=state, meta=meta)

    assert config.enable_ocr is True


def test_pipeline_config_rejects_unknown_fields():
    meta = {"pipeline_config": {"unknown": True}}

    with pytest.raises(InputError) as exc_info:
        _build_document_pipeline_config(meta=meta)

    assert "invalid_pipeline_config_override" in str(exc_info.value)
