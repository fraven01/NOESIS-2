from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Dict, List

from celery import shared_task
from django.conf import settings

from .infra import object_store, pii
from .rag.schemas import Chunk
from .rag.vector_client import EMBEDDING_DIM, PgVectorClient, get_default_client

logger = logging.getLogger(__name__)

# Factory returning the default pgvector client (can be patched in tests)
VECTOR_CLIENT_FACTORY: Callable[[], PgVectorClient] = get_default_client


def _build_path(meta: Dict[str, str], *parts: str) -> str:
    tenant = meta["tenant"]
    case = meta["case"]
    return "/".join([tenant, case, *parts])


@shared_task
def ingest_raw(meta: Dict[str, str], name: str, data: bytes) -> Dict[str, str]:
    """Persist raw document bytes."""
    path = _build_path(meta, "raw", name)
    object_store.put_bytes(path, data)
    return {"path": path}


@shared_task
def extract_text(meta: Dict[str, str], raw_path: str) -> Dict[str, str]:
    """Decode bytes to text and store."""
    full = object_store.BASE_PATH / raw_path
    text = full.read_bytes().decode("utf-8")
    out_path = _build_path(meta, "text", f"{Path(raw_path).stem}.txt")
    object_store.put_bytes(out_path, text.encode("utf-8"))
    return {"path": out_path}


@shared_task
def pii_mask(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
    """Mask PII in text."""
    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    masked = pii.mask(text)
    out_path = _build_path(meta, "text", f"{Path(text_path).stem}.masked.txt")
    object_store.put_bytes(out_path, masked.encode("utf-8"))
    return {"path": out_path}


@shared_task
def chunk(meta: Dict[str, str], text_path: str) -> Dict[str, str]:
    """Split text into chunks; stubbed as a single chunk."""
    full = object_store.BASE_PATH / text_path
    text = full.read_text(encoding="utf-8")
    hash_val = hashlib.sha1(text.encode("utf-8")).hexdigest()
    chunks: List[Dict[str, object]] = [
        {
            "content": text,
            "meta": {
                "tenant": meta["tenant"],
                "case": meta["case"],
                "source": text_path,
                "hash": hash_val,
            },
        }
    ]
    out_path = _build_path(meta, "embeddings", "chunks.json")
    object_store.write_json(out_path, chunks)
    return {"path": out_path}


@shared_task
def embed(meta: Dict[str, str], chunks_path: str) -> Dict[str, str]:
    """Attach dummy embedding vectors to chunks."""
    chunks = object_store.read_json(chunks_path)
    embeddings = [{**ch, "embedding": [0.0] * EMBEDDING_DIM} for ch in chunks]
    out_path = _build_path(meta, "embeddings", "vectors.json")
    object_store.write_json(out_path, embeddings)
    return {"path": out_path}


@shared_task
def upsert(meta: Dict[str, str], embeddings_path: str) -> int:
    """Upsert embedded chunks into the vector client."""
    if not settings.RAG_ENABLED:
        logger.info(
            "Skipping vector upsert because RAG is disabled (tenant=%s, case=%s)",
            meta.get("tenant"),
            meta.get("case"),
        )
        return 0

    data = object_store.read_json(embeddings_path)
    chunk_objs = []
    for ch in data:
        vector = ch.get("embedding")
        embedding = [float(v) for v in vector] if vector is not None else None
        chunk_objs.append(Chunk(content=ch["content"], meta=ch["meta"], embedding=embedding))

    client = VECTOR_CLIENT_FACTORY()
    written = client.upsert_chunks(chunk_objs)
    return written
