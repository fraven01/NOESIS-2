from rest_framework.exceptions import AuthenticationFailed
from rest_framework.request import Request
from rest_framework.test import APIRequestFactory

import pytest

from django.test import override_settings

from noesis2.api.authentication import LiteLLMMasterKeyAuthentication


factory = APIRequestFactory()


def _request_with_auth(header: str | None = None) -> Request:
    kwargs = {}
    if header is not None:
        kwargs["HTTP_AUTHORIZATION"] = header
    return Request(factory.get("/admin/litellm/", **kwargs))


def test_no_authorization_header_returns_none():
    auth = LiteLLMMasterKeyAuthentication()
    request = _request_with_auth()
    assert auth.authenticate(request) is None


def test_non_bearer_scheme_is_ignored():
    auth = LiteLLMMasterKeyAuthentication()
    request = _request_with_auth("Basic abc123")
    assert auth.authenticate(request) is None


def test_missing_token_raises_error():
    auth = LiteLLMMasterKeyAuthentication()
    request = _request_with_auth("Bearer")
    with pytest.raises(AuthenticationFailed):
        auth.authenticate(request)


@override_settings(LITELLM_MASTER_KEY="")
def test_missing_master_key_configuration_raises():
    auth = LiteLLMMasterKeyAuthentication()
    request = _request_with_auth("Bearer dev-token")
    with pytest.raises(AuthenticationFailed):
        auth.authenticate(request)


@override_settings(LITELLM_MASTER_KEY="super-secret")
def test_invalid_token_rejected():
    auth = LiteLLMMasterKeyAuthentication()
    request = _request_with_auth("Bearer other-token")
    with pytest.raises(AuthenticationFailed):
        auth.authenticate(request)


@override_settings(LITELLM_MASTER_KEY="super-secret")
def test_valid_token_authenticates():
    auth = LiteLLMMasterKeyAuthentication()
    request = _request_with_auth("Bearer super-secret")
    user, token = auth.authenticate(request)
    assert token == "super-secret"
    assert user.is_authenticated
    assert not user.is_anonymous
