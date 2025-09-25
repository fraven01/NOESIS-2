"""Authentication helpers for NOESIS 2 APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from django.conf import settings
from rest_framework import exceptions
from rest_framework.authentication import BaseAuthentication, get_authorization_header


@dataclass
class LiteLLMAdminUser:
    """Lightweight user object representing an authenticated LiteLLM admin."""

    master_key: str

    @property
    def is_authenticated(self) -> bool:  # pragma: no cover - simple property
        return True

    @property
    def is_anonymous(self) -> bool:  # pragma: no cover - simple property
        return False

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return "LiteLLMAdminUser"


class LiteLLMMasterKeyAuthentication(BaseAuthentication):
    """Authenticate requests using the configured LiteLLM master key."""

    keyword = b"bearer"

    def authenticate(self, request) -> Optional[Tuple[LiteLLMAdminUser, str]]:
        auth = get_authorization_header(request).split()
        if not auth:
            return None

        if auth[0].lower() != self.keyword:
            return None

        if len(auth) == 1:
            raise exceptions.AuthenticationFailed(
                "Authorization header missing bearer token."
            )

        if len(auth) > 2:
            raise exceptions.AuthenticationFailed(
                "Authorization header must not contain spaces in token."
            )

        try:
            provided_token = auth[1].decode()
        except UnicodeError as exc:  # pragma: no cover - defensive guard
            raise exceptions.AuthenticationFailed(
                "Invalid bearer token encoding."
            ) from exc

        master_key = getattr(settings, "LITELLM_MASTER_KEY", "")
        if not master_key:
            raise exceptions.AuthenticationFailed(
                "LiteLLM master key is not configured."
            )

        if provided_token != master_key:
            raise exceptions.AuthenticationFailed("Invalid LiteLLM master key.")

        user = LiteLLMAdminUser(master_key=master_key)
        return (user, master_key)

    def authenticate_header(
        self, request
    ) -> str:  # pragma: no cover - mirrors DRF default behaviour
        return "Bearer"
