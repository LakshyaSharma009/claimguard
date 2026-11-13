"""YAML rules engine: hand-written red flags that run before the model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import hashlib

import pandas as pd
import yaml

from .schema import CATEGORIES, TARGET

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES = ROOT / "config" / "rules.yaml"


@dataclass(frozen=True)
class Rule:
    id: str
    when: dict[str, list[str]]
    weight: float
    reason: str
    any: list[dict[str, list[str]]] = field(default_factory=list)
    strong: bool = False
    enabled: bool = True

    def matches(self, claim: Mapping[str, Any]) -> bool:
        if not _group_matches(self.when, claim):
            return False
        return not self.any or any(_group_matches(g, claim) for g in self.any)

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Vectorised version of matches() for whole datasets."""
        hit = _group_mask(self.when, df)
        if self.any:
            any_hit = pd.Series(False, index=df.index)
            for group in self.any:
                any_hit |= _group_mask(group, df)
            hit &= any_hit
        return hit


def _as_list(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def _group_matches(group: dict[str, list[str]], claim: Mapping[str, Any]) -> bool:
    return all(str(claim.get(col)) in allowed for col, allowed in group.items())


def _group_mask(group: dict[str, list[str]], df: pd.DataFrame) -> pd.Series:
    hit = pd.Series(True, index=df.index)
    for col, allowed in group.items():
        hit &= df[col].astype(str).isin(allowed)
    return hit


def _parse_group(raw: Mapping[str, Any], rule_id: str) -> dict[str, list[str]]:
    group = {}
    for col, value in raw.items():
        if col not in CATEGORIES:
            raise ValueError(f"{rule_id}: unknown field {col!r}")
        allowed = _as_list(value)
        unknown = set(allowed) - set(CATEGORIES[col])
        if unknown:
            raise ValueError(f"{rule_id}: {col} has no value(s) {sorted(unknown)}")
        group[col] = allowed
    return group


def load_rules(path: str | Path = DEFAULT_RULES, include_disabled: bool = False) -> list[Rule]:
    """Load and validate rules, so a typo in the YAML fails loudly at startup."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rules, seen = [], set()
    for item in raw["rules"]:
        rid = item["id"]
        if rid in seen:
            raise ValueError(f"duplicate rule id {rid}")
        seen.add(rid)
        weight = float(item["weight"])
        if not 0 <= weight <= 1:
            raise ValueError(f"{rid}: weight must be between 0 and 1")
        rule = Rule(
            id=rid,
            when=_parse_group(item.get("when", {}), rid),
            any=[_parse_group(g, rid) for g in item.get("any", [])],
            weight=weight,
            reason=str(item["reason"]),
            strong=bool(item.get("strong", False)),
            enabled=bool(item.get("enabled", True)),
        )
        if rule.enabled or include_disabled:
            rules.append(rule)
    return rules


def evaluate(claim: Mapping[str, Any], rules: list[Rule]) -> tuple[float, list[Rule]]:
    """Return (rule score capped at 1.0, rules that fired)."""
    fired = [r for r in rules if r.matches(claim)]
    return min(1.0, sum(r.weight for r in fired)), fired


def rule_scores(df: pd.DataFrame, rules: list[Rule]) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for rule in rules:
        total += rule.mask(df) * rule.weight
    return total.clip(upper=1.0)


def strong_hits(df: pd.DataFrame, rules: list[Rule]) -> pd.Series:
    hit = pd.Series(False, index=df.index)
    for rule in rules:
        if rule.strong:
            hit |= rule.mask(df)
    return hit


def hit_rate_table(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    """How often each rule fires, and what share of those claims were fraud."""
    base = df[TARGET].mean()
    rows = []
    for rule in rules:
        m = rule.mask(df)
        fires = int(m.sum())
        precision = float(df.loc[m, TARGET].mean()) if fires else 0.0
        rows.append({
            "rule": rule.id,
            "fires": fires,
            "fire_rate": fires / len(df),
            "fraud_share": precision,
            "lift": precision / base if base else 0.0,
            "fraud_caught": int(df.loc[m, TARGET].sum()),
            "enabled": rule.enabled,
        })
    return pd.DataFrame(rows)


def config_hash(config_dir: str | Path = ROOT / "config") -> str:
    """Short fingerprint of the three config files, stored with every decision."""
    h = hashlib.sha256()
    for name in ("rules.yaml", "reasons.yaml", "thresholds.yaml"):
        # Ignore line endings, so a Windows checkout (CRLF) hashes the same as Linux (LF).
        h.update(Path(config_dir, name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]
