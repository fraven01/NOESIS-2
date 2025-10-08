from ai_core.rag.filters import strict_match


def test_strict_match_positive():
    meta = {"tenant_id": "t1", "case_id": "c1", "source": "s1", "hash": "h1"}
    assert strict_match(meta, "t1", "c1") is True


def test_strict_match_negative():
    meta = {"tenant_id": "t1", "case_id": "c1", "source": "s1", "hash": "h1"}
    assert strict_match(meta, "t1", "c2") is False


def test_strict_match_negative_tenant():
    meta = {"tenant_id": "t1", "case_id": "c1", "source": "s1", "hash": "h1"}
    assert strict_match(meta, "t2", "c1") is False
