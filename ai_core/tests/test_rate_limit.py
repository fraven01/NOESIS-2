import logging

from redis.exceptions import RedisError

from ai_core.infra import rate_limit


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.expire_at = {}
        self.now = 0

    def incr(self, key):
        self._expire(key)
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    def expire(self, key, ttl):
        self.expire_at[key] = self.now + ttl

    def _expire(self, key):
        if key in self.expire_at and self.now >= self.expire_at[key]:
            self.store.pop(key, None)
            self.expire_at.pop(key, None)


def test_check_respects_quota_and_window(monkeypatch):
    fake = FakeRedis()
    rate_limit._get_redis.cache_clear()
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: fake)
    monkeypatch.setattr(rate_limit, "get_quota", lambda: 2)

    fake.now = 0
    assert rate_limit.check("t1", now=0)
    fake.now = 1
    assert rate_limit.check("t1", now=1)
    fake.now = 2
    assert not rate_limit.check("t1", now=2)

    fake.now = 61
    assert rate_limit.check("t1", now=61)


def test_get_quota_env_override(monkeypatch):
    monkeypatch.setenv("AI_CORE_RATE_LIMIT_QUOTA", "5")
    assert rate_limit.get_quota() == 5


def test_fail_open_on_redis_error(monkeypatch, caplog):
    rate_limit._get_redis.cache_clear()

    def boom():
        raise RedisError("boom")

    monkeypatch.setattr(rate_limit, "_get_redis", boom)
    with caplog.at_level(logging.WARNING):
        assert rate_limit.check("t1")
    assert "fail-open" in caplog.text
