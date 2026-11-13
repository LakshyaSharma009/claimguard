"""Model candidates, imbalanced-data metrics, calibration and the saved model bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from .features import build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "claimguard_model.joblib"
SEED = 42


def make_classifier(kind: str, **params: Any):
    if kind == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=2000, C=params.get("C", 1.0))
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        )
    if kind == "hgb":
        defaults = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                        min_samples_leaf=40, l2_regularization=1.0)
        defaults.update(params)
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **defaults)
    raise ValueError(f"unknown model kind {kind!r}")


def make_pipeline(kind: str, columns: list[str], **params: Any) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(columns)), ("clf", make_classifier(kind, **params))])


HGB_SEARCH_SPACE = {
    "clf__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "clf__max_iter": [100, 200, 300, 500],
    "clf__max_leaf_nodes": [7, 15, 31, 63],
    "clf__max_depth": [None, 3, 4, 6],
    "clf__min_samples_leaf": [10, 20, 40, 80],
    "clf__l2_regularization": [0.0, 0.1, 1.0, 5.0],
}


def calibrate(pipeline: Pipeline, X, y) -> CalibratedClassifierCV:
    """Platt scaling fitted on out-of-fold predictions.

    ensemble=False keeps one base model trained on all rows, which SHAP can explain;
    sigmoid is used over isotonic because there are only ~400 fraud cases to fit on.
    """
    cal = CalibratedClassifierCV(pipeline, method="sigmoid", cv=5, ensemble=False)
    return cal.fit(X, y)


def base_pipeline(calibrated: CalibratedClassifierCV) -> Pipeline:
    return calibrated.calibrated_classifiers_[0].estimator


# ---------- metrics ----------

def recall_at_top(y, scores, fraction: float = 0.10) -> float:
    y = np.asarray(y)
    k = max(1, int(np.ceil(fraction * len(y))))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def evaluate(y, scores, threshold: float) -> dict[str, Any]:
    y = np.asarray(y)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
        "recall_top10": recall_at_top(y, scores, 0.10),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------- persistence ----------

def save_bundle(bundle: dict[str, Any], path: str | Path = DEFAULT_MODEL) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    return joblib.load(path)
