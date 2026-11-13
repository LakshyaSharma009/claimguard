import pandas as pd
import pytest

from claimguard.rules import evaluate, hit_rate_table, load_rules

# rule id -> fields that should make it fire, starting from the quiet claim
TRIGGERS = {
    "R01_EARLY_ACCIDENT": {"Days_Policy_Accident": "1 to 7"},
    "R02_NO_EVIDENCE": {"PoliceReportFiled": "No", "WitnessPresent": "No"},
    "R03_RECENT_ADDRESS_CHANGE": {"AddressChange_Claim": "under 6 months"},
    "R04_FREQUENT_CLAIMANT": {"PastNumberOfClaims": "more than 4"},
    "R05_MANY_SUPPLEMENTS": {"NumberOfSuppliments": "more than 5"},
    "R06_COVERAGE_MISMATCH": {"BasePolicy": "All Perils", "AgeOfVehicle": "more than 7"},
    "R07_EARLY_CLAIM": {"Days_Policy_Claim": "8 to 15"},
}
ALL_RULES = load_rules(include_disabled=True)


def test_every_rule_has_a_trigger_case():
    assert {r.id for r in ALL_RULES} == set(TRIGGERS)


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.id)
def test_rule_fires_on_trigger_claim(rule, quiet_claim):
    assert rule.matches({**quiet_claim, **TRIGGERS[rule.id]})


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.id)
def test_rule_quiet_on_quiet_claim(rule, quiet_claim):
    assert not rule.matches(quiet_claim)


def test_coverage_mismatch_needs_all_perils(quiet_claim):
    rule = next(r for r in ALL_RULES if r.id == "R06_COVERAGE_MISMATCH")
    # old vehicle on a liability policy: no mismatch
    assert not rule.matches({**quiet_claim, "AgeOfVehicle": "more than 7"})
    # all perils with a cheap car: mismatch through the second "any" group
    assert rule.matches({**quiet_claim, "BasePolicy": "All Perils", "VehiclePrice": "less than 20000"})


def test_rule_score_is_capped(quiet_claim):
    claim = dict(quiet_claim)
    for fields in TRIGGERS.values():
        claim.update(fields)
    score, fired = evaluate(claim, ALL_RULES)
    assert len(fired) == len(ALL_RULES)
    assert score == 1.0


def test_disabled_rules_are_not_loaded_by_default():
    active = load_rules()
    assert all(r.enabled for r in active)
    assert len(active) < len(ALL_RULES)


@pytest.mark.parametrize("bad, message", [
    ("when: { NotAColumn: x }", "unknown field"),
    ("when: { Fault: Nobody }", "has no value"),
])
def test_bad_yaml_fails_loudly(tmp_path, bad, message):
    path = tmp_path / "rules.yaml"
    path.write_text(f"rules:\n  - id: X\n    {bad}\n    weight: 0.1\n    reason: r\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_rules(path)


def test_hit_rate_table():
    df = pd.DataFrame([
        {"Days_Policy_Accident": "1 to 7", "FraudFound_P": 1},
        {"Days_Policy_Accident": "more than 30", "FraudFound_P": 0},
    ])
    rule = next(r for r in ALL_RULES if r.id == "R01_EARLY_ACCIDENT")
    table = hit_rate_table(df, [rule])
    assert table.loc[0, "fires"] == 1
    assert table.loc[0, "fraud_share"] == 1.0
    assert table.loc[0, "lift"] == 2.0


def test_config_hash_ignores_line_endings(tmp_path):
    from claimguard.rules import ROOT, config_hash
    for name in ("rules.yaml", "reasons.yaml", "thresholds.yaml"):
        text = (ROOT / "config" / name).read_bytes().replace(b"\r\n", b"\n")
        (tmp_path / name).write_bytes(text.replace(b"\n", b"\r\n"))
    assert config_hash(tmp_path) == config_hash()
