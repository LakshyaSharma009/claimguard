import random

import numpy as np
import pytest

from claimguard.schema import CATEGORIES, NUMERIC
from claimguard.scoring import Band, assign_band, capacity_threshold, combine, cost_threshold


def test_combine_stays_in_unit_interval():
    rng = np.random.default_rng(0)
    for w in (0.0, 0.3, 0.7, 1.0):
        s = combine(rng.random(1000), rng.random(1000), w)
        assert s.min() >= 0 and s.max() <= 1


def test_combine_rejects_bad_weight():
    with pytest.raises(ValueError):
        combine(0.5, 0.5, 1.5)


@pytest.mark.parametrize("score, expected",
                         [(0.0, Band.LOW), (0.1, Band.MEDIUM), (0.5, Band.HIGH), (1.0, Band.HIGH)])
def test_assign_band(score, expected):
    assert assign_band(score, medium=0.1, high=0.3) is expected


def test_strong_rule_lifts_low_to_medium_and_never_lowers():
    assert assign_band(0.01, 0.1, 0.3, strong_rule_hit=True) is Band.MEDIUM
    assert assign_band(0.9, 0.1, 0.3, strong_rule_hit=True) is Band.HIGH


def test_assign_band_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        assign_band(0.5, medium=0.4, high=0.2)


def test_cost_threshold_on_toy_data():
    # Frauds score high, honest claims low: the cut sits between them at zero cost.
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.05])
    t, cost = cost_threshold(y, s, cost_ratio=10)
    assert 0.2 < t <= 0.8
    assert cost == 0


def test_cost_threshold_reviews_nothing_when_fraud_is_cheap():
    y = np.array([1, 0, 0, 0])
    s = np.array([0.1, 0.9, 0.8, 0.7])
    t, _ = cost_threshold(y, s, cost_ratio=0.5)
    assert t == 1.0


def test_capacity_threshold():
    assert capacity_threshold(np.linspace(0, 1, 101), 0.10) == pytest.approx(0.9)


def random_claim(rng: random.Random) -> dict:
    claim = {c: rng.choice(v) for c, v in CATEGORIES.items()}
    claim.update({c: rng.randint(lo, min(hi, 120)) for c, (lo, hi) in NUMERIC.items()})
    claim["Deductible"] = rng.choice([300, 400, 500, 700])
    return claim


def test_scorer_returns_valid_score_and_band_for_random_claims(scorer):
    rng = random.Random(7)
    for _ in range(200):
        out = scorer.score(random_claim(rng), explain=False)
        assert 0 <= out["score"] <= 1
        assert out["band"] in {"LOW", "MEDIUM", "HIGH"}


def test_scorer_reasons_are_readable(scorer, quiet_claim):
    claim = {**quiet_claim, "Fault": "Policy Holder", "BasePolicy": "All Perils",
             "PolicyType": "Sedan - All Perils"}
    out = scorer.score(claim)
    assert 1 <= len(out["model_reasons"]) <= 3
    for reason in out["model_reasons"]:
        assert reason["impact"] > 0
        assert "_" not in reason["text"]  # sentences, not raw feature names


def test_strong_rule_claim_is_at_least_medium(scorer, quiet_claim):
    out = scorer.score({**quiet_claim, "Days_Policy_Claim": "8 to 15"}, explain=False)
    assert out["band"] in {"MEDIUM", "HIGH"}
    assert any(h["id"] == "R07_EARLY_CLAIM" for h in out["rule_hits"])
