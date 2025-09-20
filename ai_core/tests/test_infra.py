import json

from django.http import HttpResponse

from ai_core.infra import object_store, pii
from ai_core.infra.config import get_config
from ai_core.infra.resp import apply_std_headers
from ai_core.infra.tracing import trace


def test_get_config_reads_env(monkeypatch):
    get_config.cache_clear()
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.local")
    monkeypatch.setenv("LITELLM_API_KEY", "secret")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sec")

    cfg = get_config()
    assert cfg.litellm_base_url == "http://litellm.local"
    assert cfg.litellm_api_key == "secret"
    assert cfg.redis_url == "redis://localhost:6379/0"
    assert cfg.langfuse_public_key == "pub"
    assert cfg.langfuse_secret_key == "sec"


def test_apply_std_headers_sets_metadata_headers_for_success():
    resp = HttpResponse("ok", status=200)
    meta = {
        "trace_id": "abc123",
        "case": "case-1",
        "tenant": "tenant-1",
        "key_alias": "alias-1",
    }

    result = apply_std_headers(resp, meta)

    assert result["X-Trace-ID"] == "abc123"
    assert result["X-Case-ID"] == "case-1"
    assert result["X-Tenant-ID"] == "tenant-1"
    assert result["X-Key-Alias"] == "alias-1"


def test_apply_std_headers_skips_missing_optional_headers():
    resp = HttpResponse("ok", status=200)
    meta = {"trace_id": "abc123", "case": "case-1", "tenant": "tenant-1"}

    result = apply_std_headers(resp, meta)

    assert "X-Key-Alias" not in result
    assert result["X-Trace-ID"] == "abc123"


def test_apply_std_headers_ignores_non_success_responses():
    meta = {"trace_id": "abc123", "case": "case-1", "tenant": "tenant-1"}

    for status in (400, 500):
        resp = HttpResponse("error", status=status)
        result = apply_std_headers(resp, meta)
        assert "X-Trace-ID" not in result
        assert "X-Case-ID" not in result
        assert "X-Tenant-ID" not in result
        assert "X-Key-Alias" not in result


def test_pii_mask_replaces_digits():
    assert pii.mask("User 123") == "User XXX"


def test_object_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    object_store.write_json("tenant/case/state.json", {"ok": True})
    assert object_store.read_json("tenant/case/state.json") == {"ok": True}
    object_store.put_bytes("tenant/case/raw/data.bin", b"hi")
    stored = tmp_path / ".ai_core_store/tenant/case/raw/data.bin"
    assert stored.read_bytes() == b"hi"


def test_trace_logs_locally_when_no_keys(monkeypatch, capsys):
    @trace("node1")
    def sample(state, meta):
        return "ok"

    sample(
        {}, {"tenant": "t1", "case": "c1", "trace_id": "tid", "prompt_version": "v1"}
    )
    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert len(lines) == 2
    start, end = map(json.loads, lines)
    assert start["event"] == "node.start"
    assert end["event"] == "node.end"
    assert end["duration_ms"] >= 0


def test_trace_sends_to_langfuse(monkeypatch):
    dispatched = []

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pub")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sec")
    monkeypatch.setattr(
        "ai_core.infra.tracing._dispatch_langfuse",
        lambda trace_id, node_name, metadata: dispatched.append(
            {"traceId": trace_id, "name": node_name, "metadata": metadata}
        ),
    )

    @trace("node2")
    def sample(state, meta):
        return "ok"

    sample(
        {}, {"tenant": "t1", "case": "c1", "trace_id": "tid", "prompt_version": "v1"}
    )
    assert dispatched[0]["traceId"] == "tid"
    assert dispatched[0]["name"] == "node2"
    assert dispatched[0]["metadata"]["tenant"] == "t1"
