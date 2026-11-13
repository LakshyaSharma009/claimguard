"""SHAP reason codes: turn model contributions into sentences an investigator can read."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import claims_frame
from .features import source_feature

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REASONS = ROOT / "config" / "reasons.yaml"


def load_reasons(path: str | Path = DEFAULT_REASONS) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["features"]


def reason_text(feature: str, value: Any, reasons: Mapping[str, Any]) -> str:
    entry = reasons.get(feature, {})
    values = entry.get("values", {})
    if value is not None and str(value) in values:
        return values[str(value)]
    label = entry.get("label", feature)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{label}: not recorded"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return entry.get("template", "{label}: {value}").format(label=label, value=value)


class Explainer:
    """Wraps a fitted preprocess+classifier pipeline and explains single claims.

    Contributions are in log-odds of the uncalibrated model. Calibration is a
    monotone map of that score, so the ranking of reasons is unchanged.
    """

    def __init__(self, pipeline: Pipeline, columns: list[str], background: pd.DataFrame | None = None):
        self.prep = pipeline.named_steps["prep"]
        self.clf = pipeline.named_steps["clf"]
        self.columns = columns
        self.encoded_names = list(self.prep.get_feature_names_out())
        self.groups = [source_feature(n, columns) for n in self.encoded_names]
        if isinstance(self.clf, LogisticRegression):
            if background is None:
                raise ValueError("a linear model needs background data")
            self.explainer = shap.LinearExplainer(self.clf, self.prep.transform(background[columns]))
        else:
            self.explainer = shap.TreeExplainer(self.clf)

    def encoded_shap(self, X: pd.DataFrame) -> np.ndarray:
        values = self.explainer.shap_values(self.prep.transform(X[self.columns]))
        if isinstance(values, list):          # older SHAP: one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:                  # (rows, features, classes)
            values = values[..., 1]
        return values

    def grouped_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """One SHAP value per raw column: one-hot pieces are summed back together."""
        enc = pd.DataFrame(self.encoded_shap(X), columns=self.encoded_names)
        return enc.T.groupby(self.groups, sort=False).sum().T[self.columns]

    def reasons(self, claim: Mapping[str, Any], reasons: Mapping[str, Any], top: int = 3) -> list[dict]:
        return self.reasons_many([claim], reasons, top)[0]

    def reasons_many(self, claims: list[Mapping[str, Any]], reasons: Mapping[str, Any],
                     top: int = 3) -> list[list[dict]]:
        """Top features pushing each claim towards fraud. One SHAP call for the whole list."""
        frame = claims_frame([dict(c) for c in claims], self.columns)
        grouped = self.grouped_shap(frame)
        out = []
        for i in range(len(frame)):
            contrib = grouped.iloc[i].sort_values(ascending=False)
            row = []
            for feature, impact in contrib.items():
                if impact <= 0 or len(row) >= top:
                    break
                value = frame.at[i, feature]
                # Plain Python types, so the audit log stores 500, not the string "500".
                value = None if pd.isna(value) else (value.item() if isinstance(value, np.generic) else value)
                row.append({"feature": feature,
                            "value": value,
                            "text": reason_text(feature, value, reasons),
                            "impact": round(float(impact), 4)})
            out.append(row)
        return out
