from __future__ import annotations

import base64
import gzip
import zlib
from typing import Any, Iterable, Mapping, Optional

try:
    import brotli  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    brotli = None  # type: ignore


_GZIP_MAGIC = b"\x1f\x8b"
_ZLIB_MAGIC_FIRST_BYTE = 0x78
_ZLIB_SECOND_BYTE_CANDIDATES = {0x01, 0x5E, 0x9C, 0xDA}


def _coerce_inline_payload(blob: Any) -> Optional[bytes]:
    if blob is None:
        return None
        
    # Check for LocalFileBlob (deferred import to avoid circularity if possible, 
    # but strictly checking type name or attributes is safer if we can't import).
    # However, standard practice is to handle the type.
    # contracts.py does NOT import payloads.py, so we can import.
    from documents.contracts import LocalFileBlob
    
    if isinstance(blob, LocalFileBlob):
        try:
            with open(blob.path, "rb") as f:
                return f.read()
        except Exception:
            return None

    if hasattr(blob, "decoded_payload"):
        payload = blob.decoded_payload()
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
    if hasattr(blob, "resolve_payload_bytes"):
        payload = blob.resolve_payload_bytes()
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
    content = _extract_mapping_candidate(blob, "content")
    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(blob, Mapping):
        base64_value = blob.get("base64")
        if isinstance(base64_value, (bytes, bytearray)):
            return bytes(base64_value)
        if isinstance(base64_value, str):
            try:
                return base64.b64decode(base64_value)
            except Exception:  # pragma: no cover - malformed payloads
                return None
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    return None


def _extract_mapping_candidate(obj: Any, key: str) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, Mapping):
        return obj.get(key)
    return None


def _normalise_encoding_hints(encoding: Optional[str]) -> Iterable[str]:
    if not encoding:
        return ()
    for token in encoding.split(","):
        normalised = token.strip().lower()
        if normalised:
            yield normalised


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


def _decompress_payload(payload: bytes, encoding: Optional[str]) -> bytes:
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


def extract_payload(
    blob: Any, *, content_encoding: Optional[str] = None
) -> Optional[bytes]:
    """
    Return raw bytes for a blob, applying optional content-encoding decoding.

    This helper centralises payload extraction for crawler-parsed documents
    so parsers and ingestion code do not have to reimplement the logic.
    """

    payload = _coerce_inline_payload(blob)
    if not payload:
        return payload
    return _decompress_payload(payload, content_encoding)


__all__ = ["extract_payload"]
