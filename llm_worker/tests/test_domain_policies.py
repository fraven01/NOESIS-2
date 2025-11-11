import pytest

from llm_worker import domain_policies
from llm_worker.domain_policies import DomainPolicyAction


def test_yaml_defaults_provide_whitelist_and_blacklist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(domain_policies, "_POLICY_CACHE", {})
    policy = domain_policies.get_domain_policy(None)

    good = policy.evaluate("good.example.com")
    assert good is not None
    assert good.action is DomainPolicyAction.BOOST

    bad = policy.evaluate("bad.example.com")
    assert bad is not None
    assert bad.action is DomainPolicyAction.REJECT


def test_yaml_rules_apply_regex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(domain_policies, "_POLICY_CACHE", {})
    policy = domain_policies.get_domain_policy(None)

    gov = policy.evaluate("justice.gov")
    assert gov is not None
    assert gov.action is DomainPolicyAction.BOOST
