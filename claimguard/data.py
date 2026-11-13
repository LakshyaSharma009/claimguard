"""Load, clean and split fraud_oracle.csv by time."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CATEGORIES, DROP_COLUMNS, TARGET, TIME_COLUMN

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "raw" / "fraud_oracle.csv"

# Train on the earliest year, tune thresholds on the next, test on the latest.
SPLIT_YEARS = {"train": [1994], "valid": [1995], "test": [1996]}


def load(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    # One raw row has "0" for the claimed day and month: a data-entry error.
    bad = pd.Series(False, index=df.index)
    for col, allowed in CATEGORIES.items():
        if col in df.columns:
            bad |= ~df[col].isin(allowed)
    df = df.loc[~bad].copy()
    # Age 0 means the age was not recorded.
    df["Age"] = df["Age"].replace(0, np.nan).astype(float)
    return df.reset_index(drop=True)


def claims_frame(claims: list[dict], columns: list[str]) -> pd.DataFrame:
    """Build a model-ready frame from claim dicts (e.g. API payloads)."""
    df = pd.DataFrame([{c: claim.get(c) for c in columns} for claim in claims], columns=columns)
    if "Age" in df:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").replace(0, np.nan).astype(float)
    return df


def time_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: df[df[TIME_COLUMN].isin(years)].reset_index(drop=True)
            for name, years in SPLIT_YEARS.items()}


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, part in splits.items():
        rows.append({
            "split": name,
            "years": ", ".join(str(y) for y in SPLIT_YEARS[name]),
            "claims": len(part),
            "fraud": int(part[TARGET].sum()),
            "fraud_rate": round(float(part[TARGET].mean()), 4),
        })
    return pd.DataFrame(rows)
