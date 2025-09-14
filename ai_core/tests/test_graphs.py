from ai_core.graphs import info_intake, scope_check, needs_mapping, system_description

META = {"tenant": "t1", "case": "c1", "trace_id": "tr"}


def test_info_intake_adds_meta():
    state, result = info_intake.run({}, META)
    assert state["meta"] == META
    assert result["tenant"] == META["tenant"]


def test_scope_check_never_creates_draft():
    state, result = scope_check.run({}, META)
    assert "draft" not in state
    assert "draft" not in result
    assert result["missing"] == ["scope"]


def test_needs_mapping_breaks_on_missing():
    initial = {"missing": ["scope"]}
    state, result = needs_mapping.run(initial, META)
    assert result["missing"] == ["scope"]
    assert "needs" not in state


def test_needs_mapping_success():
    initial = {"missing": [], "needs_input": ["a", "b"]}
    state, result = needs_mapping.run(initial, META)
    assert state["needs"] == ["a", "b"]
    assert result["mapped"] is True


def test_system_description_only_when_no_missing():
    state, result = system_description.run({"missing": ["scope"]}, META)
    assert result["skipped"] is True
    assert "description" not in state

    state2, result2 = system_description.run({"missing": []}, META)
    assert "description" in state2
    assert result2["description"] == state2["description"]
