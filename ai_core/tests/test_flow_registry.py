from __future__ import annotations

from ai_core.agent.flows.registry import get_flow_contract


def test_flow_registry_resolves_dummy_flow():
    contract = get_flow_contract("dummy_flow")
    assert contract.flow_name == "dummy_flow"
    assert contract.flow_version == "0.1.0"


def test_flow_contract_is_id_free():
    contract = get_flow_contract("dummy_flow")
    forbidden = {"tenant_id", "user_id", "case_id", "workflow_id"}
    input_fields = set(contract.InputModel.model_fields.keys())
    output_fields = set(contract.OutputModel.model_fields.keys())

    assert forbidden.isdisjoint(input_fields)
    assert forbidden.isdisjoint(output_fields)
