"""Combine the rules and model layers, and turn a score into a review band."""
from __future__ import annotations

from enum import Enum

import numpy as np


class Band(str, Enum):
    LOW = "LOW"        # auto-approve
    MEDIUM = "MEDIUM"  # quick review
    HIGH = "HIGH"      # send to the investigation team


_ORDER = {Band.LOW: 0, Band.MEDIUM: 1, Band.HIGH: 2}


def combine(model_prob, rule_score, weight: float):
    """Final score = weight * model probability + (1 - weight) * rule score, kept in [0, 1]."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    return np.clip(weight * np.asarray(model_prob, dtype=float)
                   + (1 - weight) * np.asarray(rule_score, dtype=float), 0.0, 1.0)


def assign_band(score: float, medium: float, high: float, strong_rule_hit: bool = False,
                strong_min_band: Band = Band.MEDIUM) -> Band:
    if not 0 <= medium <= high <= 1:
        raise ValueError("thresholds must satisfy 0 <= medium <= high <= 1")
    if score >= high:
        band = Band.HIGH
    elif score >= medium:
        band = Band.MEDIUM
    else:
        band = Band.LOW
    if strong_rule_hit and _ORDER[band] < _ORDER[strong_min_band]:
        band = strong_min_band
    return band


def cost_threshold(y, scores, cost_ratio: float) -> tuple[float, float]:
    """Cut-off for review (MEDIUM or above) that minimises total cost.

    Cost = cost_ratio * missed frauds + 1 * honest claims sent to review.
    Returns (threshold, cost per claim).
    """
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = y.sum() - tp
    cost = cost_ratio * fn + fp
    # Only cut between distinct scores, so every claim with the same score lands in one band.
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    candidates = np.where(last_of_tie)[0]
    flag_none = cost_ratio * y.sum()
    best = candidates[np.argmin(cost[candidates])]
    if flag_none <= cost[best]:
        return 1.0, float(flag_none / len(y))
    return float(s_sorted[best]), float(cost[best] / len(y))


def capacity_threshold(scores, capacity: float) -> float:
    """Score above which roughly `capacity` of claims fall (the HIGH volume the team can handle)."""
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - capacity))


class ClaimScorer:
    """Everything needed to score one claim: model bundle, rules, reason texts."""

    def __init__(self, bundle: dict, rules: list, reasons: dict):
        from .explain import Explainer
        from .model import base_pipeline

        self.bundle = bundle
        self.rules = rules
        self.reasons = reasons
        self.columns = bundle["columns"]
        self.model = bundle["calibrated"]
        self.weight = float(bundle["combine_weight"])
        self.medium = float(bundle["thresholds"]["medium"])
        self.high = float(bundle["thresholds"]["high"])
        self.strong_min = Band(bundle.get("strong_rule_min_band", "MEDIUM"))
        self.model_version = bundle["model_version"]
        self.explainer = Explainer(base_pipeline(self.model), self.columns, background=bundle.get("background"))

    @classmethod
    def from_paths(cls, model_path=None, rules_path=None, reasons_path=None) -> "ClaimScorer":
        from .explain import DEFAULT_REASONS, load_reasons
        from .model import DEFAULT_MODEL, load_bundle
        from .rules import DEFAULT_RULES, load_rules

        return cls(load_bundle(model_path or DEFAULT_MODEL),
                   load_rules(rules_path or DEFAULT_RULES),
                   load_reasons(reasons_path or DEFAULT_REASONS))

    def model_probability(self, claims: list[dict]) -> np.ndarray:
        from .data import claims_frame

        return self.model.predict_proba(claims_frame(claims, self.columns))[:, 1]

    def score(self, claim: dict, explain: bool = True) -> dict:
        from .rules import evaluate

        prob = float(self.model_probability([claim])[0])
        rule_score, fired = evaluate(claim, self.rules)
        final = float(combine(prob, rule_score, self.weight))
        band = assign_band(final, self.medium, self.high,
                           strong_rule_hit=any(r.strong for r in fired), strong_min_band=self.strong_min)
        return {
            "score": round(final, 4),
            "band": band.value,
            "model_probability": round(prob, 4),
            "rule_score": round(rule_score, 4),
            "rule_hits": [{"id": r.id, "reason": r.reason, "weight": r.weight, "strong": r.strong}
                          for r in fired],
            "model_reasons": self.explainer.reasons(claim, self.reasons) if explain else [],
            "model_version": self.model_version,
        }
