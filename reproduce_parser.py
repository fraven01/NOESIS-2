import hashlib
import base64
from uuid import uuid4
from datetime import datetime, timezone
from documents.contracts import (
    NormalizedDocument,
    DocumentRef,
    DocumentMeta,
    InlineBlob,
)
from documents.parsers_html import HtmlDocumentParser
from documents.pipeline import DocumentPipelineConfig


def test_parser():
    file_path = r"f:\NOESIS-2\NOESIS-2\.ai_core_store\documents\uploads\83467711d2dfeaee4b430995642692d05048893c22c4be54198c69d65af30e32.bin"

    print(f"Reading file {file_path}...")
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except Exception as e:
        print(f"Read failed: {e}")
        return

    print(f"Size: {len(content)}")
    checksum = hashlib.sha256(content).hexdigest()
    content_b64 = base64.b64encode(content).decode("ascii")

    doc = NormalizedDocument(
        ref=DocumentRef(
            tenant_id="dev",
            document_id=uuid4(),
            workflow_id="test",
            collection_id=uuid4(),
        ),
        meta=DocumentMeta(
            origin_uri="https://de.wikipedia.org/wiki/Bamberg",
            tenant_id="dev",
            workflow_id="test",
        ),
        blob=InlineBlob(
            type="inline",
            base64=content_b64,
            size=len(content),
            media_type="text/html",
            sha256=checksum,
        ),
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        source="crawler",
    )

    parser = HtmlDocumentParser()
    print("Parsing...")
    try:
        result = parser.parse(doc, DocumentPipelineConfig())
        print(f"Blocks: {len(result.text_blocks)}")
        print(f"Assets: {len(result.assets)}")

        text_len = sum(len(b.text) for b in result.text_blocks)
        print(f"Total Text Length: {text_len}")

        if result.text_blocks:
            print("First block:", result.text_blocks[0].text[:100])
        else:
            print("No blocks found!")

    except Exception as e:
        print(f"Parser failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_parser()
