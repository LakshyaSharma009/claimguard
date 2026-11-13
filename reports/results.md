# ClaimGuard results

Generated 2026-11-13T03:13:33+00:00 · model `hgb-v2` · config `733c657a2e73d77b`

## Data and time split

| split | years | claims | fraud | fraud_rate |
|---|---|---|---|---|
| train | 1994 | 6141 | 409 | 0.067 |
| valid | 1995 | 5195 | 301 | 0.058 |
| test | 1996 | 4083 | 213 | 0.052 |

## Calendar columns (default boosting; trained 1994, judged on 1995)

| features | cv_pr_auc_1994 | valid_pr_auc_1995 |
|---|---|---|
| with calendar columns | 0.274 | 0.117 |
| without calendar columns | 0.245 | 0.186 |

Decision: calendar columns **dropped**. Cross-validation inside one year mixes claims from the same weeks, so these columns look useful there; the validation year shows whether they carry over.

## Model comparison (5-fold stratified CV on 1994)

| model | cv_pr_auc | cv_pr_auc_std | cv_roc_auc |
|---|---|---|---|
| Logistic Regression (balanced) | 0.208 | 0.026 | 0.827 |
| Random Forest (balanced_subsample) | 0.240 | 0.028 | 0.831 |
| HistGradientBoosting (defaults) | 0.245 | 0.009 | 0.824 |
| HistGradientBoosting (tuned) | 0.287 | 0.024 | nan |

## SMOTE vs class weights (SMOTE applied inside each fold)

| model | cv_pr_auc | cv_pr_auc_std | cv_roc_auc |
|---|---|---|---|
| hgb + class weights | 0.287 | 0.024 | 0.844 |
| hgb + SMOTE (no class weight) | 0.253 | 0.034 | 0.834 |

Kept: **class weights**.

## Calibration (validation 1995)

| model | mean_score_valid | brier_valid | actual_fraud_rate_valid |
|---|---|---|---|
| uncalibrated (class-weighted) | 0.3098 | 0.1666 | 0.0579 |
| calibrated (sigmoid) | 0.0621 | 0.0501 | 0.0579 |

## Combine weight search (validation 1995)

| weight | valid_pr_auc |
|---|---|
| 0.500 | 0.164 |
| 0.600 | 0.174 |
| 0.700 | 0.183 |
| 0.800 | 0.188 |
| 0.900 | 0.190 |
| 1.000 | 0.192 |

Chosen weight: **1.0**. Thresholds: MEDIUM >= 0.0944 (cost-minimising at 10:1), HIGH >= 0.2089 (top 5% of validation scores).

## Test year 1996 (untouched until now)

| scorer | valid_pr_auc | test_pr_auc | test_roc_auc | recall_top10 | precision@review | recall@review | review_rate |
|---|---|---|---|---|---|---|---|
| Rules only | 0.075 | 0.062 | 0.532 | 0.150 | 0.143 | 0.028 | 0.010 |
| Model only (calibrated) | 0.192 | 0.136 | 0.778 | 0.263 | 0.110 | 0.549 | 0.261 |
| Combined (1.0 model + 0.0 rules) | 0.192 | 0.136 | 0.778 | 0.263 | 0.110 | 0.549 | 0.261 |

## Bands on the test year

| band | claims | share | fraud | fraud_rate | share_of_all_fraud |
|---|---|---|---|---|---|
| LOW | 2902 | 0.711 | 92 | 0.032 | 0.432 |
| MEDIUM | 938 | 0.230 | 90 | 0.096 | 0.423 |
| HIGH | 243 | 0.060 | 31 | 0.128 | 0.146 |

## Cost on the test year (assumed cost ratio)

| policy | cost_per_claim_test |
|---|---|
| Review nothing | 0.5217 |
| Review everything | 0.9478 |
| ClaimGuard bands (MEDIUM+HIGH reviewed) | 0.4849 |

## Confusion matrix, test year (reviewed = MEDIUM or HIGH)

| | reviewed | auto-approved |
|---|---|---|
| fraud | 121 | 92 |
| honest | 1060 | 2810 |

## More training years (same settings; every row scored on all of 1996)

| model | trained_on | train_claims | test_pr_auc | recall_top10 | review_rate | recall@review | cost_per_claim | medium | high |
|---|---|---|---|---|---|---|---|---|---|
| A: thresholds from 1995 (method above) | 1994 | 6141 | 0.1358 | 0.2629 | 0.2892 | 0.5681 | 0.4849 | 0.0944 | 0.2089 |
| B: out-of-fold thresholds | 1994 | 6141 | 0.1358 | 0.2629 | 0.3456 | 0.6854 | 0.4739 | 0.0838 | 0.2254 |
| C: out-of-fold thresholds | 1994-1995 | 11336 | 0.1598 | 0.2676 | 0.2295 | 0.4789 | 0.4764 | 0.1202 | 0.1947 |

A is the method above. B changes only how thresholds are set (out-of-fold scores inside the training year instead of a separate year), so A vs B shows whether that change is safe. C adds 1995 to the training data.

Paired bootstrap of C minus A PR-AUC over the same 1996 claims (2000 resamples): +0.0231, 95% CI [+0.0048, +0.0429]; C was better in 99.2% of resamples.

## Shipped model

Fit: **all_years** (`final_fit` in config/thresholds.yaml) on 14345 claims from 1994, 1995, 1996 (817 fraud). The 1074-claim CI holdout from 1996 is left out of training. Thresholds from out-of-fold scores: MEDIUM >= 0.0977, HIGH >= 0.1743. Holdout PR-AUC 0.224, ROC-AUC 0.778 (holdout fraud rate 0.099). There is no later year left to test it on, so its expected performance on new claims is the matching row of the table above (C: out-of-fold thresholds, trained on 1994-1995).

## Global feature importance (mean |SHAP|, shipped model on the CI holdout)

| feature | mean_abs_shap |
|---|---|
| Fault | 1.3504 |
| BasePolicy | 0.7576 |
| PolicyType | 0.1344 |
| AddressChange_Claim | 0.0700 |
| Make | 0.0639 |
| AgeOfVehicle | 0.0347 |
| VehiclePrice | 0.0248 |
| NumberOfSuppliments | 0.0167 |
| DriverRating | 0.0124 |
| AccidentArea | 0.0113 |
| PoliceReportFiled | 0.0079 |
| AgentType | 0.0070 |
