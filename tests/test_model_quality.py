"""Regression guard: a retrain that loses quality must fail CI.

The fixture is a fixed, stratified sample of the 1996 test year (half of its fraud
cases, a quarter of its honest ones), so its base rate is ~9.9%. The model shipped
with this repo scores ~0.215 PR-AUC on it; a model with no skill scores ~0.099.
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from claimguard.schema import TARGET

FIXTURE = Path(__file__).parent / "fixtures" / "holdout_sample.csv"
MIN_PR_AUC = 0.17
MIN_ROC_AUC = 0.70


@pytest.fixture(scope="module")
def holdout():
    return pd.read_csv(FIXTURE)


def test_fixture_is_the_expected_sample(holdout):
    assert len(holdout) == 1074
    assert holdout[TARGET].sum() == 106


def test_model_quality_floor(scorer, holdout):
    p = scorer.model_probability(holdout.to_dict("records"))
    pr_auc = average_precision_score(holdout[TARGET], p)
    assert pr_auc >= MIN_PR_AUC, f"PR-AUC {pr_auc:.3f} fell below {MIN_PR_AUC}"
    assert pr_auc >= 1.5 * holdout[TARGET].mean()
    assert roc_auc_score(holdout[TARGET], p) >= MIN_ROC_AUC


def test_probabilities_are_calibrated_roughly(scorer, holdout):
    # Calibrated on 1994, where fraud was 6.7%: the mean score should be near real-world rates,
    # not near 0.5 as an uncalibrated class-weighted model would give.
    p = scorer.model_probability(holdout.to_dict("records"))
    assert 0.02 < p.mean() < 0.15
