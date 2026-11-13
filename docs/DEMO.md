# ClaimGuard demo guide

A 10–12 minute walkthrough for anyone: a manager, an interviewer, or a technical audience.
Every number below is what you should see on screen. If you see something different, check
[Troubleshooting](#troubleshooting).

## Before the demo (once, about 5 minutes)

You need Python 3.13 and git.

```bash
git clone https://github.com/LakshyaSharma009/claimguard.git
cd claimguard
python -m venv .venv
.venv\Scripts\activate             # Windows (PowerShell or cmd)
# source .venv/bin/activate        # macOS / Linux
pip install -r requirements-dev.txt
pytest -q                          # expect: 68 passed
```

## Right before the audience arrives (1 minute)

Start with an empty audit log so the numbers on screen are clean.

**Windows PowerShell**
```powershell
Remove-Item data\demo.db -ErrorAction SilentlyContinue
$env:CLAIMGUARD_DB = "data\demo.db"
uvicorn claimguard.api:app --port 8000
```

**macOS / Linux**
```bash
rm -f data/demo.db
CLAIMGUARD_DB=data/demo.db uvicorn claimguard.api:app --port 8000
```

Open **http://localhost:8000** in a browser. The top right should read
*Model hgb-v2 · trained on 1994–1996 (14,345 claims)*. Keep a second terminal open (with the venv
activated) for step 6.

---

## The demo

### 1. The problem (1 minute, no clicking)

Say:

> Insurers get thousands of claims and a few percent are fraudulent. Reviewing all of them is
> expensive; reviewing none lets fraud through. ClaimGuard scores each claim, puts it in a band:
> **LOW** (approve), **MEDIUM** (quick check), **HIGH** (investigate), and explains why.
> It never rejects anyone on its own: a person always decides.

### 2. Score one claim (2 minutes): tab *Score a claim*

1. Click **Fill: low-risk example**, then **Score claim** at the bottom.
   - Expect **LOW**, score about **0.003**. "The other driver was at fault, the police report and witness
     exist, a liability-only policy. Nothing stands out."
2. Scroll up, type `DEMO-001` into **Your claim reference**, click **Fill: high-risk example**, then
   **Score claim**.
   - Expect **HIGH**, score **0.180**.
   - **Red-flag rules**: *accident within 15 days of the policy starting* and *recent address change*.
     These are hand-written business rules in a YAML file.
   - **Top model reasons**: *policy holder was at fault*, *all-perils policy*. "These come from SHAP: for
     every claim, the model shows which facts pushed the score up."
   - The bar under the score shows where 0.180 sits relative to the MEDIUM and HIGH thresholds.
3. Point at the **greyed-out fields** (sex, age, accident month).
   > The model is not allowed to use these. Sex and age made it flag women at half the rate of men,
   > so they are excluded as a policy. The calendar fields looked useful in one year but failed the next.

### 3. Score a whole batch (1 minute): tab *Batch upload*

1. Click **Choose File** and pick `tests/fixtures/holdout_sample.csv` from the project folder.
2. Click **Upload and score**.
   - Expect **1,074 claims scored in about 1 second**: **HIGH 50 · MEDIUM 232 · LOW 792 · 0 rejected**.
   > These are real 1996 claims the model never saw during training. About a quarter go to review;
   > three quarters are approved straight away.
3. Optional: **Download results CSV** to show the output a claims system would import.

### 4. The investigator's view (2 minutes): tab *Review queue*

1. The queue shows unreviewed HIGH and MEDIUM claims, highest score first.
2. Click the top row. The right side shows why it was flagged, and **Full claim** expands every field.
3. Choose **Confirmed fraud**, type a reviewer name (e.g. `demo-investigator`), add a note, and click
   **Record outcome**.
   - "Outcome recorded" appears, and the claim leaves the *Not yet reviewed* list.
   - Switch the filter to **Reviewed** to show it there.
   > Outcomes are never edited, only added. That is how we learn whether the model was right.

### 5. Is it working? (1 minute): tab *Monitoring*

- The table shows decisions per band and the **confirmed-fraud rate** from investigator outcomes.
  > In production you watch this: if HIGH claims stop turning out to be fraud more often than LOW
  > claims, the model needs retraining.
- The **Model** panel shows the version, training data, thresholds, active rules, and
  *config hash (matches training)*: if someone edits the rules without retraining, it says so.

### 6. Can you trust the record? (2 minutes): second terminal

```bash
python -m claimguard.audit verify --db data/demo.db
```
Expect `"ok": true` and the number of decisions you created.

```bash
python scripts/tamper_demo.py --db data/demo.db
```
This works on a **copy** of the log and shows, step by step:
1. The chain verifies.
2. A normal edit to downgrade the top HIGH claim to LOW is **blocked by the database**.
3. Someone removes the protection and edits it anyway.
4. Verification now reports **DETECTED: … row contents changed after it was written**.

> Every decision is chained with SHA-256 hashes, like a ledger. You can't quietly change a
> decision after the fact, so disputes and audits can rely on it.

### 7. The evidence (1–2 minutes)

- **Swagger API docs**: footer link *API docs* (http://localhost:8000/docs). Every feature in the web app
  is also an API call another system can make.
- **GitHub Actions**: https://github.com/LakshyaSharma009/claimguard/actions. Every push runs 68 tests,
  including a model-quality floor, then builds and checks the Docker image.
- **Project report**: `docs/ClaimGuard_Project_Report.pdf`, 7 pages on why every decision was made.
- **Headline result** (from the report):
  > Judged on a year it never saw, the model ranks fraud about 3x better than random (PR-AUC 0.160 vs a
  > 5.2% base rate). Training on more years improved that by +0.023, and a bootstrap test showed
  > the gain is real (95% CI +0.005 to +0.043).

---

## Likely questions, short answers

| Question | Answer |
|---|---|
| Why not just report accuracy? | Fraud is ~6% of claims. Saying "honest" to everything is 94% accurate and useless. PR-AUC measures how well fraud is ranked to the top. |
| A PR-AUC of 0.16 sounds low. | The base rate is 0.05, so it's ~3x better than chance on a year it never saw, with no claim amounts or text in the data. That's why it prioritises reviews and never decides alone. |
| How do you know it isn't overfitted? | Data is split by year: decisions on 1994/1995, one final test on 1996. The shipped model is also checked on a holdout it never trained on, in CI. |
| Why leave out sex and age? | With them, women were flagged at half the rate of men. Many regulators forbid it. The cost of leaving them out (0.026 PR-AUC) is measured in `reports/fairness.md`. |
| Why do the rules have no weight in the score? | Tested on 1995: the model alone ranked best. Rules still force a claim to at least MEDIUM and are shown to investigators as reasons. |
| Where do the thresholds come from? | MEDIUM: where reviewing costs less than missing fraud, at an assumed 10:1 cost. HIGH: the top 5%, sized to the investigation team. Both are assumptions in `config/thresholds.yaml`. |
| What if someone changes the rules? | `/health` and the Monitoring tab show "config CHANGED since training" until it's retrained, and every decision stores the config fingerprint it used. |
| Can it run in production? | It ships as a Docker image with a health check. For several instances you'd move the audit log from SQLite to PostgreSQL and add authentication. |

## After the demo

Press `Ctrl+C` in the server terminal. To reset for the next demo, delete `data/demo.db` (it is
git-ignored).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `address already in use` / error 10048 | Another server is on that port. Use `--port 8001` and open http://localhost:8001. |
| Page says "API unreachable" | The server isn't running, or you opened a different port. |
| Model fails to load / pickle error | Install exactly the pinned versions: `pip install -r requirements.txt` in a fresh venv. |
| Batch numbers differ from 50 / 232 / 792 | You uploaded a different file. Use `tests/fixtures/holdout_sample.csv`. |
| No Python on the demo machine | With Docker: `docker build -t claimguard .` then `docker run -p 8000:8000 claimguard`. Step 6 then runs inside the container: `docker exec -it <container> python -m claimguard.audit verify --db /data/claimguard.db`. |
