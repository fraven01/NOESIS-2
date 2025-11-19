from __future__ import annotations

import re
from typing import Callable

from django.conf import settings

from ai_core.infra import pii
from ai_core.infra.pii_flags import get_pii_config


class PIIMasker:
    """Apply ingestion PII masking with a numeric fallback."""

    _DIGIT_PATTERN = re.compile(r"\d")

    def __init__(
        self,
        enabled: bool | None = None,
        mask_func: Callable[[str], str] = pii.mask,
        config_loader: Callable[[], dict] = get_pii_config,
    ) -> None:
        if enabled is None:
            try:
                enabled = bool(getattr(settings, "INGESTION_PII_MASK_ENABLED", True))
            except Exception:
                enabled = True
        self._enabled = bool(enabled)
        self._mask_func = mask_func
        self._config_loader = config_loader

    def mask(self, value: str) -> str:
        if not self._enabled:
            return value
        masked_value = self._mask_func(value)
        if masked_value == value:
            config = self._config_loader() or {}
            mode = str(config.get("mode", "")).lower()
            policy = str(config.get("policy", "")).lower()
            if mode != "off" and policy != "off":
                masked_value = self._DIGIT_PATTERN.sub("X", value)
        return masked_value
