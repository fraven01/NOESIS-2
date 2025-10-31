from __future__ import annotations

import gzip
import zlib
from typing import Iterable, Optional

try:
    import brotli  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    brotli = None  # type: ignore


_GZIP_MAGIC = b"\x1f\x8b"
_ZLIB_MAGIC_FIRST_BYTE = 0x78
_ZLIB_SECOND_BYTE_CANDIDATES = {0x01, 0x5E, 0x9C, 0xDA}


def _normalise_encoding_hints(encoding: Optional[str]) -> Iterable[str]:
    if not encoding:
        return ()
    for token in encoding.split(","):
        normalized = token.strip().lower()
        if normalized:
            yield normalized


def _guess_encoding(payload: bytes) -> Iterable[str]:
    if payload.startswith(_GZIP_MAGIC):
        yield "gzip"
    if (
        len(payload) >= 2
        and payload[0] == _ZLIB_MAGIC_FIRST_BYTE
        and payload[1] in _ZLIB_SECOND_BYTE_CANDIDATES
    ):
        yield "deflate"
    if brotli is not None:
        yield "br"


def _try_decompress_gzip(payload: bytes) -> Optional[bytes]:
    try:
        return gzip.decompress(payload)
    except Exception:
        return None


def _try_decompress_deflate(payload: bytes) -> Optional[bytes]:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(payload, wbits)
        except Exception:
            continue
    return None


def _try_decompress_brotli(payload: bytes) -> Optional[bytes]:
    if brotli is None:
        return None
    try:
        return brotli.decompress(payload)  # type: ignore[no-any-return]
    except Exception:
        return None


def decompress_payload(payload: bytes, encoding: Optional[str] = None) -> bytes:
    """
    Return bytes with optional content-encoding decoded.

    This helper is shared between crawler fetchers and document normalization
    to ensure consistent handling of gzip/deflate/brotli payloads.
    """

    for hint in list(_normalise_encoding_hints(encoding)) + list(
        _guess_encoding(payload)
    ):
        if hint in {"identity", "utf-8", "utf8"}:
            return payload
        if hint in {"gzip", "x-gzip"}:
            decompressed = _try_decompress_gzip(payload)
            if decompressed is not None:
                return decompressed
        if hint in {"deflate"}:
            decompressed = _try_decompress_deflate(payload)
            if decompressed is not None:
                return decompressed
        if hint in {"br", "brotli"}:
            decompressed = _try_decompress_brotli(payload)
            if decompressed is not None:
                return decompressed

    for candidate in (
        _try_decompress_gzip,
        _try_decompress_deflate,
        _try_decompress_brotli,
    ):
        decompressed = candidate(payload)
        if decompressed is not None:
            return decompressed
    return payload


__all__ = ["decompress_payload"]
