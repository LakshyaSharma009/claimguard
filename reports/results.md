# ClaimGuard results

Generated 2026-11-12T18:32:48+00:00 · model `hgb-v1` · config `b2ef4040de836fc5`

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

## Global feature importance (mean |SHAP|, test year)

| feature | mean_abs_shap |
|---|---|
| Fault | 1.6256 |
| BasePolicy | 0.7657 |
| PolicyType | 0.1819 |
| NumberOfSuppliments | 0.0950 |
| VehiclePrice | 0.0890 |
| Make | 0.0835 |
| AddressChange_Claim | 0.0611 |
| PastNumberOfClaims | 0.0590 |
| AgeOfVehicle | 0.0577 |
| AgentType | 0.0435 |
| AccidentArea | 0.0370 |
| DriverRating | 0.0286 |
