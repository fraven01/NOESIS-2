from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from celery import shared_task

from .infra import object_store, pii
from .rag.vector_client import InMemoryVectorClient
from .rag.schemas import Chunk

# Global in-memory vector client used by the upsert task
VECTOR_CLIENT = InMemoryVectorClient()


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
    embeddings = [{**ch, "vector": [0.0]} for ch in chunks]
    out_path = _build_path(meta, "embeddings", "vectors.json")
    object_store.write_json(out_path, embeddings)
    return {"path": out_path}


@shared_task
def upsert(meta: Dict[str, str], embeddings_path: str) -> int:
    """Upsert embedded chunks into the vector client."""
    data = object_store.read_json(embeddings_path)
    chunk_objs = [Chunk(content=ch["content"], meta=ch["meta"]) for ch in data]
    VECTOR_CLIENT.upsert_chunks(chunk_objs)
    return len(chunk_objs)
