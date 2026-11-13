"""Train ClaimGuard end to end and write every results table.

    python -m claimguard.train [--csv data/raw/fraud_oracle.csv] [--quick]

Order of work (each step only looks at the data it is allowed to see):
  1. 1994 (train): compare models with 5-fold stratified CV, tune boosting, SMOTE test,
     rule hit rates.
  2. 1995 (valid): fairness choice, combine weight, cost and capacity thresholds.
  3. 1996 (test): one final, untouched evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import yaml  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,  # noqa: E402
                                     cross_validate)

from . import data, rules as rules_mod  # noqa: E402
from .rules import config_hash  # noqa: E402
from .explain import Explainer  # noqa: E402
from .features import build_preprocessor  # noqa: E402
from .model import (DEFAULT_MODEL, HGB_SEARCH_SPACE, SEED, base_pipeline, calibrate,  # noqa: E402
                    calibrated_model, evaluate, make_classifier, make_pipeline, recall_at_top, save_bundle)
from .schema import PROTECTED_COLUMNS, TARGET, TIME_COLUMN, feature_columns  # noqa: E402
from .scoring import Band, assign_band, capacity_threshold, combine, cost_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"
FIXTURE = ROOT / "tests" / "fixtures" / "holdout_sample.csv"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cv_summary(name: str, estimator, X, y) -> dict:
    res = cross_validate(estimator, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    out = {"model": name,
           "cv_pr_auc": res["test_pr_auc"].mean(), "cv_pr_auc_std": res["test_pr_auc"].std(),
           "cv_roc_auc": res["test_roc_auc"].mean()}
    print(f"  {name:<34} PR-AUC {out['cv_pr_auc']:.3f} ± {out['cv_pr_auc_std']:.3f}")
    return out


def group_table(df: pd.DataFrame, scores, fraction: float = 0.10) -> pd.DataFrame:
    """Flag rate per protected group when the top `fraction` of claims is reviewed."""
    cut = np.quantile(scores, 1 - fraction)
    flagged = np.asarray(scores) >= cut
    age_band = pd.cut(df["Age"], [0, 25, 40, 65, 200], labels=["<=25", "26-40", "41-65", "65+"])
    rows = []
    for col, groups in (("Sex", df["Sex"]), ("Age", age_band.cat.add_categories("unknown").fillna("unknown").astype(str))):
        for g in sorted(groups.unique()):
            m = (groups == g).to_numpy()
            fraud = df.loc[m, TARGET].to_numpy()
            rows.append({"attribute": col, "group": g, "claims": int(m.sum()),
                         "fraud_rate": float(fraud.mean()),
                         "flag_rate": float(flagged[m].mean()),
                         "recall": float(flagged[m][fraud == 1].mean()) if fraud.sum() else float("nan")})
    return pd.DataFrame(rows)


def fit_with_oof_thresholds(frame: pd.DataFrame, columns: list[str], kind: str, params: dict,
                            weight: float, rules: list, cost_ratio: float, capacity: float) -> dict:
    """Fit a calibrated model on `frame` and set its band thresholds without a separate year.

    The thresholds come from out-of-fold calibrated scores (5-fold CV over `frame`), so every
    score used to pick them comes from a model that did not see that claim.
    """
    X, y = frame[columns], frame[TARGET].to_numpy()
    oof = cross_val_predict(calibrated_model(make_pipeline(kind, columns, **params)), X, y,
                            cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    s = combine(oof, rules_mod.rule_scores(frame, rules).to_numpy(), weight)
    t_medium, _ = cost_threshold(y, s, cost_ratio)
    t_high = max(t_medium, capacity_threshold(s, capacity))
    model = calibrate(make_pipeline(kind, columns, **params), X, y)
    return {"model": model, "medium": t_medium, "high": t_high,
            "oof_pr_auc": float(average_precision_score(y, s))}


def main(csv: str | None = None, quick: bool = False) -> dict:
    REPORTS.mkdir(exist_ok=True)
    th_cfg = yaml.safe_load((CONFIG / "thresholds.yaml").read_text(encoding="utf-8"))
    cost_ratio, capacity = float(th_cfg["cost_ratio"]), float(th_cfg["high_capacity"])
    csv_path = Path(csv) if csv else data.DEFAULT_CSV
    df = data.load(csv_path)
    splits = data.time_split(df)
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    summary = data.split_summary(splits)
    print(summary.to_string(index=False))

    ytr = train[TARGET].to_numpy()

    # ---------- 0. calendar columns: in or out? (default boosting, decided on valid) ----------
    calendar_rows = []
    for label, include in (("with calendar columns", True), ("without calendar columns", False)):
        cols = feature_columns(include_protected=True, include_calendar=include)
        pipe = make_pipeline("hgb", cols).fit(train[cols], ytr)
        cv = cross_validate(make_pipeline("hgb", cols), train[cols], ytr, cv=CV,
                            scoring="average_precision", n_jobs=-1)["test_score"].mean()
        calendar_rows.append({"features": label, "cv_pr_auc_1994": cv,
                              "valid_pr_auc_1995": average_precision_score(
                                  valid[TARGET], pipe.predict_proba(valid[cols])[:, 1])})
    calendar_table = pd.DataFrame(calendar_rows)
    include_calendar = bool(calendar_table.valid_pr_auc_1995[0] > calendar_table.valid_pr_auc_1995[1])
    print(calendar_table.round(3).to_string(index=False))
    print(f"Calendar columns: {'kept' if include_calendar else 'dropped'}")

    all_cols = feature_columns(include_protected=True, include_calendar=include_calendar)
    Xtr = train[all_cols]

    # ---------- rules: hit rates on train (all rules, including disabled ones) ----------
    every_rule = rules_mod.load_rules(include_disabled=True)
    active_rules = [r for r in every_rule if r.enabled]
    hits = rules_mod.hit_rate_table(train, every_rule)
    hits["break_even_ok"] = hits["fraud_share"] >= 1 / (1 + cost_ratio)
    hits["decision"] = np.where(hits["enabled"], "keep", "drop")
    hits_valid = rules_mod.hit_rate_table(valid, every_rule)[["rule", "fires", "fraud_share", "lift"]]

    # ---------- 1. model comparison on train with 5-fold CV ----------
    print("Model comparison (5-fold stratified CV on 1994):")
    comparison = [
        cv_summary("Logistic Regression (balanced)", make_pipeline("logreg", all_cols), Xtr, ytr),
        cv_summary("Random Forest (balanced_subsample)", make_pipeline("rf", all_cols), Xtr, ytr),
        cv_summary("HistGradientBoosting (defaults)", make_pipeline("hgb", all_cols), Xtr, ytr),
    ]
    search = RandomizedSearchCV(make_pipeline("hgb", all_cols), HGB_SEARCH_SPACE,
                                n_iter=8 if quick else 40, scoring="average_precision",
                                cv=CV, random_state=SEED, n_jobs=-1)
    search.fit(Xtr, ytr)
    hgb_params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    i = search.best_index_
    comparison.append({"model": "HistGradientBoosting (tuned)",
                       "cv_pr_auc": search.cv_results_["mean_test_score"][i],
                       "cv_pr_auc_std": search.cv_results_["std_test_score"][i],
                       "cv_roc_auc": float("nan")})
    print(f"  {'HistGradientBoosting (tuned)':<34} PR-AUC {comparison[-1]['cv_pr_auc']:.3f}  {hgb_params}")
    comparison = pd.DataFrame(comparison)

    candidates = {"logreg": ({}, comparison.cv_pr_auc[0]), "rf": ({}, comparison.cv_pr_auc[1]),
                  "hgb": (hgb_params, max(comparison.cv_pr_auc[2], comparison.cv_pr_auc[3]))}
    if comparison.cv_pr_auc[2] > comparison.cv_pr_auc[3]:
        candidates["hgb"] = ({}, comparison.cv_pr_auc[2])
    kind = max(candidates, key=lambda k: candidates[k][1])
    params = candidates[kind][0]
    print(f"Best model: {kind} {params}")

    # ---------- SMOTE experiment on the best model, inside each fold ----------
    smote_clf = make_classifier(kind, **params)
    smote_clf.set_params(class_weight=None)
    smote_pipe = ImbPipeline([("prep", build_preprocessor(all_cols)),
                              ("smote", SMOTE(random_state=SEED)), ("clf", smote_clf)])
    smote = cv_summary(f"{kind} + SMOTE (no class weight)", smote_pipe, Xtr, ytr)
    weighted = cv_summary(f"{kind} + class weights", make_pipeline(kind, all_cols, **params), Xtr, ytr)
    use_smote = smote["cv_pr_auc"] > weighted["cv_pr_auc"]
    smote_table = pd.DataFrame([weighted, smote])
    if use_smote:
        print("SMOTE beat class weights, but it is reported only: the shipped model keeps class weights "
              "so SHAP explains a model trained on real claims.")

    # ---------- 2a. fairness: with vs without protected attributes (decided on valid) ----------
    fair_cols = feature_columns(include_protected=False, include_calendar=include_calendar)
    fairness = {}
    for label, cols in (("with Sex/Age", all_cols), ("without Sex/Age", fair_cols)):
        pipe = make_pipeline(kind, cols, **params).fit(train[cols], ytr)
        pv = pipe.predict_proba(valid[cols])[:, 1]
        pt = pipe.predict_proba(test[cols])[:, 1]
        fairness[label] = {
            "cols": cols,
            "valid_pr_auc": average_precision_score(valid[TARGET], pv),
            "test_pr_auc": average_precision_score(test[TARGET], pt),
            "groups_valid": group_table(valid, pv),
            "groups_test": group_table(test, pt),
        }
    gap = fairness["with Sex/Age"]["valid_pr_auc"] - fairness["without Sex/Age"]["valid_pr_auc"]
    # Policy from config: protected attributes are excluded unless explicitly allowed.
    keep_protected = bool(th_cfg.get("use_protected_attributes", False))
    columns = all_cols if keep_protected else fair_cols
    print(f"Fairness: PR-AUC gap with-vs-without = {gap:+.4f}; "
          f"{'keeping' if keep_protected else 'dropping'} {PROTECTED_COLUMNS}")

    # ---------- calibrate the chosen model on train ----------
    uncal = make_pipeline(kind, columns, **params).fit(train[columns], ytr)
    calibrated = calibrate(make_pipeline(kind, columns, **params), train[columns], ytr)
    p_valid = calibrated.predict_proba(valid[columns])[:, 1]
    p_test = calibrated.predict_proba(test[columns])[:, 1]
    calib_table = pd.DataFrame([
        {"model": "uncalibrated (class-weighted)", "mean_score_valid": uncal.predict_proba(valid[columns])[:, 1].mean(),
         "brier_valid": evaluate(valid[TARGET], uncal.predict_proba(valid[columns])[:, 1], 0.5)["brier"]},
        {"model": "calibrated (sigmoid)", "mean_score_valid": p_valid.mean(),
         "brier_valid": evaluate(valid[TARGET], p_valid, 0.5)["brier"]},
    ])
    calib_table["actual_fraud_rate_valid"] = valid[TARGET].mean()

    # ---------- 2b. combine weight + thresholds on valid ----------
    r_valid = rules_mod.rule_scores(valid, active_rules).to_numpy()
    r_test = rules_mod.rule_scores(test, active_rules).to_numpy()
    weight_rows = [{"weight": w, "valid_pr_auc": average_precision_score(valid[TARGET], combine(p_valid, r_valid, w))}
                   for w in th_cfg["combine_weight_grid"]]
    weights = pd.DataFrame(weight_rows)
    weight = float(weights.loc[weights.valid_pr_auc.idxmax(), "weight"])
    s_valid, s_test = combine(p_valid, r_valid, weight), combine(p_test, r_test, weight)

    t_medium, cost_valid = cost_threshold(valid[TARGET], s_valid, cost_ratio)
    t_high = max(t_medium, capacity_threshold(s_valid, capacity))
    print(f"Combine weight {weight}; thresholds MEDIUM >= {t_medium:.4f}, HIGH >= {t_high:.4f}")

    strong_min = Band(th_cfg.get("strong_rule_min_band", "MEDIUM"))

    def bands_for(frame, scores):
        strong = rules_mod.strong_hits(frame, active_rules).to_numpy()
        return np.array([assign_band(s, t_medium, t_high, bool(h), strong_min).value
                         for s, h in zip(scores, strong)])

    def band_table(frame, scores):
        b = bands_for(frame, scores)
        y = frame[TARGET].to_numpy()
        rows = []
        for band in ("LOW", "MEDIUM", "HIGH"):
            m = b == band
            rows.append({"band": band, "claims": int(m.sum()), "share": m.mean(),
                         "fraud": int(y[m].sum()), "fraud_rate": y[m].mean() if m.any() else 0.0,
                         "share_of_all_fraud": y[m].sum() / y.sum()})
        return pd.DataFrame(rows), b

    def policy_cost(frame, b):
        y = frame[TARGET].to_numpy()
        reviewed = b != "LOW"
        return float((cost_ratio * ((~reviewed) & (y == 1)).sum() + (reviewed & (y == 0)).sum()) / len(y))

    bands_valid, b_valid = band_table(valid, s_valid)
    bands_test, b_test = band_table(test, s_test)

    # ---------- 3. final test evaluation ----------
    t_model_only, _ = cost_threshold(valid[TARGET], p_valid, cost_ratio)
    test_rows = []
    for name, sv, st, thr in (
        ("Rules only", r_valid, r_test, cost_threshold(valid[TARGET], r_valid, cost_ratio)[0]),
        ("Model only (calibrated)", p_valid, p_test, t_model_only),
        (f"Combined ({weight:.1f} model + {1 - weight:.1f} rules)", s_valid, s_test, t_medium),
    ):
        m = evaluate(test[TARGET], st, thr)
        test_rows.append({"scorer": name, "valid_pr_auc": average_precision_score(valid[TARGET], sv),
                          "test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"],
                          "recall_top10": m["recall_top10"], "precision@review": m["precision"],
                          "recall@review": m["recall"], "review_rate": m["flag_rate"]})
    test_table = pd.DataFrame(test_rows)
    final_metrics = evaluate(test[TARGET], s_test, t_medium)
    final_metrics["recall_top10_model_only"] = recall_at_top(test[TARGET], p_test, 0.10)
    yt = test[TARGET].to_numpy()
    reviewed = b_test != "LOW"
    final_metrics["band_policy"] = {
        "precision": float(yt[reviewed].mean()), "recall": float(yt[reviewed].sum() / yt.sum()),
        "review_rate": float(reviewed.mean()),
        "confusion": {"tn": int(((~reviewed) & (yt == 0)).sum()), "fp": int((reviewed & (yt == 0)).sum()),
                      "fn": int(((~reviewed) & (yt == 1)).sum()), "tp": int((reviewed & (yt == 1)).sum())},
    }
    cost_table = pd.DataFrame([
        {"policy": "Review nothing", "cost_per_claim_test": cost_ratio * yt.mean()},
        {"policy": "Review everything", "cost_per_claim_test": 1 - yt.mean()},
        {"policy": "ClaimGuard bands (MEDIUM+HIGH reviewed)", "cost_per_claim_test": policy_cost(test, b_test)},
    ])

    # ---------- fixed holdout sample for CI (stratified, test year; never trained on) ----------
    fraud = test[test[TARGET] == 1].sample(frac=0.5, random_state=SEED)
    honest = test[test[TARGET] == 0].sample(frac=0.25, random_state=SEED)
    holdout = pd.concat([fraud, honest]).sample(frac=1, random_state=SEED)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    # Raw fraud_oracle.csv format (unrecorded Age is 0, not blank), so it can be posted to /score/batch.
    holdout.assign(Age=holdout["Age"].fillna(0).astype(int)).to_csv(FIXTURE, index=False, lineterminator="\n")

    # ---------- 4. does more data help? every row is scored on all of 1996 ----------
    # Same model settings as above. With 1995 used for training there is no spare year for
    # the thresholds, so they come from out-of-fold scores inside the training years.
    def policy_row(name: str, years: str, n_train: int, scores, medium: float, high: float) -> dict:
        b = bands_for_thresholds(test, scores, medium, high)
        reviewed = b != "LOW"
        return {"model": name, "trained_on": years, "train_claims": n_train,
                "test_pr_auc": average_precision_score(yt, scores),
                "recall_top10": recall_at_top(yt, scores, 0.10),
                "review_rate": reviewed.mean(), "recall@review": yt[reviewed].sum() / yt.sum(),
                "cost_per_claim": policy_cost(test, b), "medium": medium, "high": high}

    def bands_for_thresholds(frame, scores, medium, high):
        strong = rules_mod.strong_hits(frame, active_rules).to_numpy()
        return np.array([assign_band(s, medium, high, bool(h), strong_min).value
                         for s, h in zip(scores, strong)])

    def test_scores(model):
        return combine(model.predict_proba(test[columns])[:, 1], r_test, weight)

    oof_1994 = fit_with_oof_thresholds(train, columns, kind, params, weight, active_rules, cost_ratio, capacity)
    train_9495 = pd.concat([train, valid], ignore_index=True)
    oof_9495 = fit_with_oof_thresholds(train_9495, columns, kind, params, weight, active_rules,
                                       cost_ratio, capacity)
    forward_table = pd.DataFrame([
        policy_row("A: thresholds from 1995 (method above)", "1994", len(train), s_test, t_medium, t_high),
        policy_row("B: out-of-fold thresholds", "1994", len(train), test_scores(oof_1994["model"]),
                   oof_1994["medium"], oof_1994["high"]),
        policy_row("C: out-of-fold thresholds", "1994-1995", len(train_9495),
                   test_scores(oof_9495["model"]), oof_9495["medium"], oof_9495["high"]),
    ])
    print(forward_table.round(4).to_string(index=False))

    # Is C's PR-AUC gain over A more than noise? Paired bootstrap over the same 1996 claims.
    pa, pc = s_test, test_scores(oof_9495["model"])
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].any():
            diffs.append(average_precision_score(yt[i], pc[i]) - average_precision_score(yt[i], pa[i]))
    diffs = np.array(diffs)
    boot = {"mean": float(diffs.mean()), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "share_c_better": float((diffs > 0).mean())}
    print(f"C - A PR-AUC on 1996: {boot['mean']:+.4f}, 95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")

    # ---------- 5. the shipped model ----------
    final_fit = th_cfg.get("final_fit", "all_years")
    if final_fit == "all_years":
        final_frame = pd.concat([train, valid, test.drop(index=holdout.index)], ignore_index=True)
        final = fit_with_oof_thresholds(final_frame, columns, kind, params, weight, active_rules,
                                        cost_ratio, capacity)
        deployed, d_medium, d_high = final["model"], final["medium"], final["high"]
        expected = forward_table.iloc[2]
    elif final_fit == "train_year":
        final_frame, deployed, d_medium, d_high = train, calibrated, t_medium, t_high
        expected = forward_table.iloc[0]
    else:
        raise ValueError(f"final_fit must be all_years or train_year, not {final_fit!r}")
    p_hold = deployed.predict_proba(holdout[columns])[:, 1]
    holdout_metrics = {"claims": len(holdout), "fraud": int(holdout[TARGET].sum()),
                       "pr_auc": float(average_precision_score(holdout[TARGET], p_hold)),
                       "roc_auc": float(roc_auc_score(holdout[TARGET], p_hold))}
    final_years = sorted(final_frame[TIME_COLUMN].unique().tolist())
    print(f"Shipped model: {final_fit}, {len(final_frame)} claims ({int(final_frame[TARGET].sum())} fraud), "
          f"MEDIUM >= {d_medium:.4f}, HIGH >= {d_high:.4f}, holdout PR-AUC {holdout_metrics['pr_auc']:.3f}")

    # ---------- SHAP global view of the shipped model, on the holdout it never saw ----------
    base = base_pipeline(deployed)
    explainer = Explainer(base, columns, background=final_frame)
    grouped = explainer.grouped_shap(holdout)
    importance = grouped.abs().mean().sort_values(ascending=False)
    plt.figure()
    shap.summary_plot(explainer.encoded_shap(holdout), base.named_steps["prep"].transform(holdout[columns]),
                      feature_names=explainer.encoded_names, max_display=15, show=False)
    plt.title("SHAP summary, shipped model on the CI holdout (encoded features)")
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=130)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.head(15)[::-1].plot.barh(ax=ax, color="#3b6ea5")
    ax.set_xlabel("mean |SHAP| (log-odds), CI holdout")
    ax.set_title("Which raw features drive the fraud score")
    fig.tight_layout()
    fig.savefig(REPORTS / "shap_importance.png", dpi=130)
    plt.close(fig)

    # ---------- save model bundle ----------
    version = f"{kind}-v2"
    bundle = {
        "model_version": version,
        "kind": kind,
        "params": params,
        "columns": columns,
        "calibrated": deployed,
        "background": final_frame[columns].sample(200, random_state=SEED),
        "combine_weight": weight,
        "thresholds": {"medium": d_medium, "high": d_high},
        "final_fit": final_fit,
        "training_years": final_years,
        "training_claims": len(final_frame),
        "strong_rule_min_band": strong_min.value,
        "cost_ratio": cost_ratio,
        "high_capacity": capacity,
        "config_hash": config_hash(),
        "data_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Best estimate for unseen claims: the matching forward check (scored on all of 1996).
        "test_metrics": {"pr_auc": float(expected["test_pr_auc"]), "recall_top10": float(expected["recall_top10"])},
        "holdout_metrics": holdout_metrics,
        # Ties the model to the exact CI holdout it was not trained on (line endings ignored).
        "holdout_sha256": hashlib.sha256(FIXTURE.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    }
    save_bundle(bundle, DEFAULT_MODEL)

    # ---------- reports ----------
    metrics = {
        "model_version": version, "kind": kind, "params": params, "keep_protected": keep_protected,
        "include_calendar": include_calendar, "use_smote": bool(use_smote), "combine_weight": weight,
        "thresholds_1994_model": {"medium": t_medium, "high": t_high}, "cost_ratio": cost_ratio,
        "high_capacity": capacity, "valid_cost_per_claim": cost_valid, "test": final_metrics,
        "forward_check": forward_table.to_dict("records"), "forward_bootstrap_c_minus_a": boot,
        "shipped": {"final_fit": final_fit, "training_years": final_years,
                    "training_claims": len(final_frame), "training_fraud": int(final_frame[TARGET].sum()),
                    "thresholds": {"medium": d_medium, "high": d_high}, "holdout": holdout_metrics},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    hits_out = hits.copy()
    hits_out["valid_fraud_share"] = hits_valid["fraud_share"].to_numpy()
    hits_out["valid_lift"] = hits_valid["lift"].to_numpy()
    (REPORTS / "rule_hit_rates.md").write_text(
        "# Rule hit rates (1994 training year; validation 1995 for stability)\n\n"
        f"Base fraud rate in training: {train[TARGET].mean():.3f}. "
        f"Cost break-even precision 1/(1+{cost_ratio:g}) = {1 / (1 + cost_ratio):.3f}.\n\n"
        + md_table(hits_out[["rule", "fires", "fire_rate", "fraud_share", "lift", "fraud_caught",
                             "valid_fraud_share", "valid_lift", "break_even_ok", "decision"]]) + "\n",
        encoding="utf-8")

    fair_md = ["# Fairness check: with vs without Sex and Age\n",
               f"Same model ({kind}, same parameters). Protected columns: {', '.join(PROTECTED_COLUMNS)}.\n",
               md_table(pd.DataFrame([{"variant": k, "valid_pr_auc": v["valid_pr_auc"],
                                       "test_pr_auc": v["test_pr_auc"]} for k, v in fairness.items()])),
               f"\nPR-AUC gap on validation (with minus without) = {gap:+.4f}. Shipped model: "
               f"**{'with' if keep_protected else 'without'} Sex/Age**, set by `use_protected_attributes` "
               "in config/thresholds.yaml.\n",
               "\nHow the decision was made: the first version of this check used a PR-AUC-only rule "
               "(keep the columns if they add more than 0.01) and kept them. The per-group table below "
               "then showed that the model with Sex/Age flags women at about half the rate of men, "
               "and catches far less of their fraud. Many regulators also forbid sex-based insurance "
               "decisions. The columns are now excluded as a policy, and this report shows what that costs.\n",
               "\nFlag rate per group when the top 10% of 1996 claims are reviewed:\n"]
    for k, v in fairness.items():
        fair_md += [f"\n## {k}\n", md_table(v["groups_test"])]
    (REPORTS / "fairness.md").write_text("\n".join(fair_md) + "\n", encoding="utf-8")

    results = [
        "# ClaimGuard results\n",
        f"Generated {bundle['trained_at']} · model `{version}` · config `{bundle['config_hash']}`\n",
        "## Data and time split\n", md_table(summary), "",
        "## Calendar columns (default boosting; trained 1994, judged on 1995)\n", md_table(calendar_table), "",
        f"Decision: calendar columns **{'kept' if include_calendar else 'dropped'}**. Cross-validation "
        "inside one year mixes claims from the same weeks, so these columns look useful there; "
        "the validation year shows whether they carry over.\n",
        "## Model comparison (5-fold stratified CV on 1994)\n", md_table(comparison), "",
        "## SMOTE vs class weights (SMOTE applied inside each fold)\n", md_table(smote_table), "",
        f"Kept: **{'SMOTE (reported only, see note)' if use_smote else 'class weights'}**.\n",
        "## Calibration (validation 1995)\n", md_table(calib_table, ".4f"), "",
        "## Combine weight search (validation 1995)\n", md_table(weights), "",
        f"Chosen weight: **{weight}**. Thresholds: MEDIUM >= {t_medium:.4f} (cost-minimising at "
        f"{cost_ratio:g}:1), HIGH >= {t_high:.4f} (top {capacity:.0%} of validation scores).\n",
        "## Test year 1996 (untouched until now)\n", md_table(test_table), "",
        "## Bands on the test year\n", md_table(bands_test), "",
        "## Cost on the test year (assumed cost ratio)\n", md_table(cost_table, ".4f"), "",
        "## Confusion matrix, test year (reviewed = MEDIUM or HIGH)\n",
        "| | reviewed | auto-approved |\n|---|---|---|\n"
        f"| fraud | {final_metrics['band_policy']['confusion']['tp']} | {final_metrics['band_policy']['confusion']['fn']} |\n"
        f"| honest | {final_metrics['band_policy']['confusion']['fp']} | {final_metrics['band_policy']['confusion']['tn']} |\n",
        "## More training years (same settings; every row scored on all of 1996)\n",
        md_table(forward_table, ".4f"), "",
        "A is the method above. B changes only how thresholds are set (out-of-fold scores inside the "
        "training year instead of a separate year), so A vs B shows whether that change is safe. "
        "C adds 1995 to the training data.\n",
        f"Paired bootstrap of C minus A PR-AUC over the same 1996 claims (2000 resamples): "
        f"{boot['mean']:+.4f}, 95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]; "
        f"C was better in {boot['share_c_better']:.1%} of resamples.\n",
        "## Shipped model\n",
        f"Fit: **{final_fit}** (`final_fit` in config/thresholds.yaml) on {len(final_frame)} claims from "
        f"{', '.join(map(str, final_years))} ({int(final_frame[TARGET].sum())} fraud). The "
        f"{holdout_metrics['claims']}-claim CI holdout from 1996 is left out of training. "
        f"Thresholds from out-of-fold scores: MEDIUM >= {d_medium:.4f}, HIGH >= {d_high:.4f}. "
        f"Holdout PR-AUC {holdout_metrics['pr_auc']:.3f}, ROC-AUC {holdout_metrics['roc_auc']:.3f} "
        f"(holdout fraud rate {holdout_metrics['fraud'] / holdout_metrics['claims']:.3f}). "
        "There is no later year left to test it on, so its expected performance on new claims is the "
        f"matching row of the table above ({expected['model']}, trained on {expected['trained_on']}).\n",
        "## Global feature importance (mean |SHAP|, shipped model on the CI holdout)\n",
        md_table(importance.head(12).rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}), ".4f"),
    ]
    (REPORTS / "results.md").write_text("\n".join(results) + "\n", encoding="utf-8")
    print(test_table.round(3).to_string(index=False))
    print(bands_test.round(3).to_string(index=False))
    print(f"Saved {DEFAULT_MODEL} and reports in {REPORTS}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--quick", action="store_true", help="smaller hyper-parameter search")
    args = ap.parse_args()
    main(args.csv, args.quick)
