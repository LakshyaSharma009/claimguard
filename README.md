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
pytest -q                       # 68 tests, uses the committed model
uvicorn claimguard.api:app --reload
```

Open http://localhost:8000 for the web app, or http://localhost:8000/docs for the interactive API.

### Web app

A single page served by the API itself (plain HTML/CSS/JS in `claimguard/static/`, no build step):

- **Score a claim**: a form built from `/schema`, so it only offers values the API accepts.
  Fields the model doesn't use (calendar columns, Sex/Age) are greyed out. The result shows the
  band, the score against the thresholds, the rules that fired and the top model reasons.
- **Batch upload**: score a CSV, see band counts and rejected rows, download the results.
- **Review queue**: unreviewed MEDIUM/HIGH decisions, highest score first. Open one to see why
  it was flagged and record *Confirmed fraud* or *Cleared*.
- **Monitoring**: confirmed-fraud rate per band from investigator outcomes, plus model details.

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
| GET | `/decisions` | review queue: filter by `band` (repeatable) and `reviewed`, `sort=score` or `recent`, `limit` up to 500 |
| GET | `/decisions/{id}` | stored decision plus any reviews |
| POST | `/decisions/{id}/review` | record `CONFIRMED_FRAUD` or `CLEARED`, with the reviewer and notes |
| GET | `/metrics` | decisions, reviews and confirmed-fraud rate for each band |
| GET | `/health` | model version, training data, thresholds, loaded rules, and whether the config matches what the model was trained with |
| GET | `/schema` | form fields, allowed values, labels, and whether the model uses each field (drives the web form) |

Batches are scored with one model call and one SHAP call for all rows and logged in one audit
transaction. The 1,074-claim holdout scores in about 1 second, against 33 seconds one claim at a time.

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
   investigation team's capacity. For the shipped model both thresholds are set from
   out-of-fold scores (see below): MEDIUM >= 0.098, HIGH >= 0.174.
5. **Reasons.** SHAP values per claim, written out as text using `config/reasons.yaml`.

## Evaluation

The data is split by year so the model is always judged on claims that came later than the
ones it learned from: train on 1994, choose settings on 1995, and test once on 1996.
Full tables are in `reports/results.md`.

| test year 1996 | PR-AUC | ROC-AUC | fraud caught in top 10% |
|---|---|---|---|
| rules only | 0.062 | 0.532 | 15% |
| ClaimGuard model | 0.136 | 0.778 | 26% |

The fraud base rate is 5.2%. With the 1994 model, reviewing MEDIUM + HIGH (29% of claims) catches
57% of the fraud. At the assumed 10:1 cost ratio this costs 0.485 per claim, against 0.522 for
reviewing nothing.

### The shipped model uses all three years

Every choice above is made on the 1994/1995/1996 split. The shipped model is then refit with the
same settings on all three years: 14,345 claims with 817 fraud cases, twice the 409 cases in 1994.
The CI holdout (1,074 claims from 1996) is left out. With 1995 now training data, there is no spare
year for the thresholds, so they come from out-of-fold scores: 5-fold cross-validation over the
training claims, so each score comes from a model that did not see that claim.

There is no later year to test the shipped model on, so training checks the method on 1996,
using models that never saw 1996:

| on all of 1996 | trained on | PR-AUC | review rate | fraud caught | cost per claim |
|---|---|---|---|---|---|
| A: thresholds from 1995 | 1994 | 0.136 | 29% | 57% | 0.485 |
| B: out-of-fold thresholds | 1994 | 0.136 | 35% | 69% | 0.474 |
| C: out-of-fold thresholds | 1994 + 1995 | 0.160 | 23% | 48% | 0.476 |

- A vs B: out-of-fold thresholds are as good as a separate year, at a slightly lower cost.
- A vs C: more training years rank fraud better. A paired bootstrap on the same 1996 claims puts
  the PR-AUC gain at +0.023, 95% CI [+0.005, +0.043]; the 1994+1995 model was better in 99.2% of
  resamples (computed by `train.py`, reported in `reports/results.md`).
- C reviews fewer claims than A because its thresholds came out higher. The cost per claim is
  about the same.

Set `final_fit: train_year` in `config/thresholds.yaml` and retrain to ship the 1994 model instead.

Decisions worth knowing:

- **Calendar columns dropped.** In cross-validation within 1994 they looked useful, but they
  hurt on 1995.
- **Class weights instead of SMOTE.** SMOTE was applied inside each fold and still scored lower.
- **Sex and Age are not used.** With them, the model flagged women at about half the rate of men.
  Leaving them out costs about 0.026 validation PR-AUC. See `reports/fairness.md`.

## Guardrails

- CI (`.github/workflows/ci.yml`) runs the tests against the committed model and a fixed holdout
  sample, so a retrain that drops below PR-AUC 0.17 / ROC-AUC 0.70 on that sample fails
  (the shipped model scores 0.224 / 0.778; the sample is 9.9% fraud). The model stores a hash
  of the holdout file, and a test checks it, so the model can't silently be trained on those claims.
  CI then builds the Docker image and checks it with a request to `/health`.
- The model bundle stores a hash of the config files. `/health` reports
  `config_matches_training: false` if the rules or thresholds were edited without retraining.
- Every decision and review is appended to a hash chain in SQLite, so editing or deleting a
  record is detectable.

## Limitations

- One public dataset, from 1994–1996, with no claim amounts. Cost figures use an assumed ratio,
  not money.
- A PR-AUC of 0.14–0.16 on an unseen year is a modest result. The model ranks claims for review
  and is not good enough to deny any claim on its own.
- The shipped model's own performance on a later year is unknown, because no later year exists.
  Row C above is the best estimate. Once new claims are reviewed, `/metrics` shows the real
  confirmed-fraud rate per band.
- The CI holdout comes from the same year as part of the training data, so its PR-AUC (0.224)
  is somewhat optimistic. It guards against regressions and is not a forecast.
- Requirements are pinned because the model is a pickle and must load with the same scikit-learn.

## Layout

```
claimguard/   api, scoring, rules, model, features, explain, audit, train; static/ holds the web app
config/       rules.yaml, thresholds.yaml, reasons.yaml (all business assumptions live here)
models/       trained model bundle
reports/      generated results, fairness, rule hit rates, SHAP plots
tests/        unit, API, audit and model-quality tests
```
