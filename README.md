# ClaimGuard

Fraud triage for vehicle insurance claims. Each claim gets a fraud score, a review band
(LOW / MEDIUM / HIGH), the red-flag rules it hit, the top model reasons in plain English,
and a hash-chained audit record. Investigators record their findings against the decision,
and `/metrics` reports how often each band turns out to be real fraud.

ClaimGuard sends claims to people for review. It never rejects a claim by itself.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements-dev.txt
pytest -q                       # 61 tests, uses the committed model
uvicorn claimguard.api:app --reload
```

Open http://localhost:8000/docs for the interactive API.

### Docker

```bash
docker build -t claimguard .
docker run -p 8000:8000 -v claimguard-data:/data claimguard
```

The audit database lives in the `/data` volume.

### Retraining

Put the Kaggle *Vehicle Insurance Claim Fraud Detection* file at `data/raw/fraud_oracle.csv`, then:

```bash
python -m claimguard.train            # full search
python -m claimguard.train --quick    # smaller hyper-parameter search
```

This rewrites `models/claimguard_model.joblib`, every file in `reports/` and the CI holdout
fixture in `tests/fixtures/`.

## API

| method | path | purpose |
|---|---|---|
| POST | `/score` | score one claim (JSON). An unknown category or out-of-range number returns 422, never a wrong score |
| POST | `/score/batch` | score a CSV with the `fraud_oracle.csv` columns (max 5000 rows). Bad rows are listed in `errors` and the rest are still scored |
| GET | `/decisions/{id}` | stored decision plus any reviews |
| POST | `/decisions/{id}/review` | record `CONFIRMED_FRAUD` or `CLEARED`, with the reviewer and notes |
| GET | `/metrics` | decisions, reviews and confirmed-fraud rate for each band |
| GET | `/health` | model version, thresholds, loaded rules, and whether the config matches what the model was trained with |

`Age` is optional: blank, missing or 0 all mean "not recorded". An optional `claim_ref`
(your own claim number) is stored with the decision and echoed back in the response.

Environment variables: `CLAIMGUARD_DB` (SQLite audit log, default `data/claimguard.db`),
`CLAIMGUARD_MODEL` (model bundle path).

Check that the audit log has not been tampered with:

```bash
python -m claimguard.audit verify
```

## How a claim is scored

1. **Model.** Gradient boosting (scikit-learn `HistGradientBoostingClassifier`) with class weights
   and sigmoid calibration, so the probability is close to the real fraud rate.
2. **Rules.** Red flags from `config/rules.yaml`. A rule stays on only if, in the 1994 training
   year, the claims it flags are at least 1.5x as likely to be fraud as the average claim.
   4 of the 7 rules pass (see `reports/rule_hit_rates.md`). A rule marked `strong` sends a
   claim to at least MEDIUM.
3. **Combined score** = w × model + (1 − w) × rule score. Training searched w on the 1995 year and
   chose 1.0: on this data, adding the rules to the score did not improve ranking. The rules still
   set the minimum band and are shown to investigators as reasons.
4. **Bands.** MEDIUM is the score where reviewing costs less than missing fraud, given the cost
   ratio in `config/thresholds.yaml`. HIGH is the top 5% of scores, sized to the
   investigation team's capacity.
5. **Reasons.** SHAP values per claim, written out as text using `config/reasons.yaml`.

## Evaluation

The data is split by year so the model is always judged on claims that came later than the
ones it learned from: train on 1994, choose settings on 1995, and test once on 1996.
Full tables are in `reports/results.md`.

| test year 1996 | PR-AUC | ROC-AUC | fraud caught in top 10% |
|---|---|---|---|
| rules only | 0.062 | 0.532 | 15% |
| ClaimGuard model | 0.136 | 0.778 | 26% |

The fraud base rate is 5.2%. Reviewing MEDIUM + HIGH (26% of claims) catches 55% of the fraud.
At the assumed 10:1 cost ratio this costs 0.485 per claim, against 0.522 for reviewing nothing.

Decisions worth knowing:

- **Calendar columns dropped.** In cross-validation within 1994 they looked useful, but they
  hurt on 1995.
- **Class weights instead of SMOTE.** SMOTE was applied inside each fold and still scored lower.
- **Sex and Age are not used.** With them, the model flagged women at about half the rate of men.
  Leaving them out costs about 0.026 validation PR-AUC. See `reports/fairness.md`.

## Guardrails

- CI (`.github/workflows/ci.yml`) runs the tests against the committed model and a fixed holdout
  sample, so a retrain that drops below PR-AUC 0.17 / ROC-AUC 0.70 on that sample fails.
  CI then builds the Docker image and checks it with a request to `/health`.
- The model bundle stores a hash of the config files. `/health` reports
  `config_matches_training: false` if the rules or thresholds were edited without retraining.
- Every decision and review is appended to a hash chain in SQLite, so editing or deleting a
  record is detectable.

## Limitations

- One public dataset, from 1994–1996, with no claim amounts. Cost figures use an assumed ratio,
  not money.
- PR-AUC of 0.136 is a modest result. The model ranks claims for review and is not good enough
  to deny any claim on its own.
- The shipped model learns only from the 6,141 claims in 1994, because 1995 was used to set the
  thresholds and 1996 was held back for testing. Refitting on all three years would likely help,
  but the thresholds would then need a fresh validation period.
- Requirements are pinned because the model is a pickle and must load with the same scikit-learn.

## Layout

```
claimguard/   api, scoring, rules, model, features, explain, audit, train
config/       rules.yaml, thresholds.yaml, reasons.yaml (all business assumptions live here)
models/       trained model bundle
reports/      generated results, fairness, rule hit rates, SHAP plots
tests/        unit, API, audit and model-quality tests
```
