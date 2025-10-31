"""Helpers to determine concrete media types for image assets."""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse

import httpx

from documents.contract_utils import normalize_media_type

_IMAGE_EXTENSION_MAP = {
    ".apng": "image/apng",
    ".avif": "image/avif",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}

_HTTP_HEADERS = {
    "Accept": "image/avif,image/webp,image/apng,image/*;q=0.8",
    "User-Agent": "Noesis2Parser/1.0 (+https://noesis.ai)",
}
_HTTP_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=5.0)
_HTTP_TRANSPORT_OVERRIDE: Optional[httpx.BaseTransport] = None


def _normalise_media_type_candidate(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    primary = value.split(";", 1)[0].strip()
    if not primary:
        return None
    try:
        return normalize_media_type(primary)
    except ValueError:
        return None


def _media_type_from_data_uri(file_uri: str) -> Optional[str]:
    try:
        prefix = file_uri.split(";", 1)[0]
        scheme = prefix.split(":", 1)[1]
    except (IndexError, ValueError):
        return None
    return _normalise_media_type_candidate(scheme)


async def _fetch_remote_media_type_async(url: str) -> Optional[str]:
    client_kwargs = {"follow_redirects": True, "timeout": _HTTP_TIMEOUT}
    if _HTTP_TRANSPORT_OVERRIDE is not None:
        client_kwargs["transport"] = _HTTP_TRANSPORT_OVERRIDE
    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            head_response = await client.head(url, headers=_HTTP_HEADERS)
        except httpx.RequestError:
            return None
        media_type = _normalise_media_type_candidate(
            head_response.headers.get("Content-Type")
        )
        if media_type:
            return media_type
        try:
            async with client.stream("GET", url, headers=_HTTP_HEADERS) as get_response:
                media_type = _normalise_media_type_candidate(
                    get_response.headers.get("Content-Type")
                )
                if media_type:
                    return media_type
        except httpx.RequestError:
            return None
    return None


@lru_cache(maxsize=256)
def _resolve_remote_media_type(url: str) -> Optional[str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop_running = False
    else:
        loop_running = True
    if loop_running:
        return None
    try:
        return asyncio.run(_fetch_remote_media_type_async(url))
    except Exception:  # pragma: no cover - defensive catch to keep parsing resilient
        return None


def resolve_image_media_type(
    file_uri: str, *, declared_type: Optional[str] = None
) -> str:
    declared = _normalise_media_type_candidate(declared_type)
    if declared:
        return declared

    if file_uri.startswith("data:"):
        inline_type = _media_type_from_data_uri(file_uri)
        if inline_type:
            return inline_type

    parsed = urlparse(file_uri)
    path = parsed.path or file_uri
    _, extension = os.path.splitext((path or "").lower())
    if extension and extension in _IMAGE_EXTENSION_MAP:
        return _IMAGE_EXTENSION_MAP[extension]

    if parsed.scheme in {"http", "https"}:
        remote_type = _resolve_remote_media_type(file_uri)
        if remote_type:
            return remote_type

    return "image/unspecified"


__all__ = ["resolve_image_media_type"]
