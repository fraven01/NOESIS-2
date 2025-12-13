"""Blob writer adapters backed by shared storage implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from common.assets import BlobIO, sha256_bytes

from . import object_store


def _normalize_segments(segments: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for raw in segments:
        candidate = str(raw).strip()
        if not candidate:
            continue
        normalized.append(object_store.sanitize_identifier(candidate))
    return normalized


@dataclass
class ObjectStoreBlobWriter(BlobIO):
    """``BlobWriter`` implementation backed by ``ai_core.infra.object_store``."""

    prefix: Sequence[str] = ()

    def _path(self, checksum: str) -> str:
        segments = _normalize_segments(self.prefix)
        segments.append(f"{checksum}.bin")
        return "/".join(segments)

    def put(self, payload: bytes) -> tuple[str, str, int]:
        checksum = sha256_bytes(payload)
        path = self._path(checksum)
        object_store.put_bytes(path, payload)
        return f"objectstore://{path}", checksum, len(payload)

    def get(self, uri: str) -> bytes:
        path = uri
        if uri.startswith("objectstore://"):
            path = uri.split("objectstore://", 1)[1]
        return object_store.read_bytes(path)


class S3BlobWriter(BlobIO):
    """Blob writer using an S3-compatible object storage backend."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: Sequence[str] | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = tuple(prefix or ())
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client = None

    def _client_factory(self):  # pragma: no cover - exercised in integration
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region_name,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
        )

    @property
    def _s3(self):  # pragma: no cover - exercised in integration
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _key(self, checksum: str) -> str:
        segments = _normalize_segments(self._prefix)
        segments.append(f"{checksum}.bin")
        return "/".join(segments)

    def put(self, payload: bytes) -> tuple[str, str, int]:  # pragma: no cover - network
        checksum = sha256_bytes(payload)
        key = self._key(checksum)
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=payload)
        return f"s3://{self._bucket}/{key}", checksum, len(payload)

    def get(self, uri: str) -> bytes:  # pragma: no cover - network
        if uri.startswith("s3://"):
            _, remainder = uri.split("s3://", 1)
            bucket, key = remainder.split("/", 1)
        else:
            bucket = self._bucket
            key = uri
        response = self._s3.get_object(Bucket=bucket, Key=key)
        body = response.get("Body")
        if body is None:
            raise ValueError("s3_empty_body")
        payload = body.read()
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("s3_body_invalid")
        return bytes(payload)


__all__ = ["ObjectStoreBlobWriter", "S3BlobWriter"]
