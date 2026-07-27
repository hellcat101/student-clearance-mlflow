# RUNBOOK — Student Academic Clearance MLflow Project

Do the steps in order. Each step says **what to run**, **what you should see**, and
**what to screenshot** for submission. There are two tracks — do **Track A (local)** to
get everything working fast, then **Track B (Databricks)** for the cloud requirement.
You can submit with either; doing both is strongest.

---

## Step 0 — One-time setup (5 min)

1. Install Python 3.11+ and Git.
2. In the project folder:
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Confirm the dataset is at `data/global_student_academic_status.csv`.

✅ **Done when:** `pip list` shows `mlflow`, `scikit-learn`, `pandas`.

---

## Step 1 — Confirm the dataset with your professor (already approved)

You've done this — the professor approved the Kaggle dataset over Teams.
Keep a screenshot of that approval message for your submission.

📸 **Screenshot:** the Teams message where the dataset was approved.

---

## Step 2 — Prove the objective & data understanding

Run the signal check so you understand (and can defend) the data:
```bash
python analysis/signal_check.py
```
You'll see chi-square p-values, mutual information ~0, and ROC-AUC ≈ 0.51.
This is your evidence for the "why the F1 is low" justification.

📸 **Screenshot:** the full `signal_check.py` output.

---

## TRACK A — Local MLflow (get it working first)

### Step A1 — Start the MLflow UI
In a terminal (leave it running):
```bash
mlflow ui --port 5000
```
Open http://127.0.0.1:5000 in your browser.

### Step A2 — Train + register the baseline (assignment steps 2 & 3)
In a second terminal (venv activated):
```bash
python src/train.py --model-type baseline
```
You should see `accuracy ~0.65`, `f1_at_risk 0.00`, and
`no Production model existed -> promoted v1`.

📸 **Screenshot:** the MLflow UI **Experiments** page showing the run + metrics.
📸 **Screenshot:** the **Models** page showing `student-clearance-classifier` v1 in **Production**.

### Step A3 — Serve the model for inference (assignment step 3)
In a third terminal:
```bash
mlflow models serve -m "models:/student-clearance-classifier/Production" -p 5001 --no-conda
```
Then, in another terminal, send a test request:
```bash
python src/predict_request.py
```
You should get predictions back like `[1, 1]`.

📸 **Screenshot:** the terminal showing the inference request and the returned predictions.

### Step A4 — Improve + AUTO-PROMOTE (assignment step 4)
```bash
python src/promote.py --model-type logreg
```
You should see:
```
current Production: v1 f1_at_risk=0.0000
candidate: v2 f1_at_risk=0.4106
PROMOTED v2 to Production (beat v1 by +0.4106)
```

📸 **Screenshot:** this promotion output.
📸 **Screenshot:** the MLflow **Models** page now showing **v2 = Production**, **v1 = Archived**.

### Step A5 — Re-serve the new model (auto-expose)
Restart the serve command from A3 (it now loads v2 automatically because it points at
`Production`), and re-run `python src/predict_request.py` to confirm the new model answers.

📸 **Screenshot:** inference working against the promoted model.

✅ **Track A done:** you have tracking → registry → serving → automated promotion, all local.

---

## TRACK B — Databricks Free Edition (cloud requirement)

Follow `DATABRICKS_SETUP.md` in detail; the short version:

### Step B1 — Create a free workspace
Sign up for Databricks **Free Edition**. You get serverless compute + Unity Catalog.

### Step B2 — Upload the dataset to a UC Volume
Catalog → `workspace` → `default` → Create **Volume** named `data` → upload the CSV.
Path becomes `/Volumes/workspace/default/data/global_student_academic_status.csv`.

### Step B3 — Import the notebook
Workspace → Import → `databricks/mlflow_pipeline_databricks.py`.
Edit `CATALOG`, `SCHEMA`, `CSV_PATH` at the top if yours differ.

### Step B4 — Run all cells
Attach to Serverless → **Run all**. The cells: train baseline → register to Unity Catalog →
set alias `@champion` → train improved model → **move `@champion` to the winner** → create a
serving endpoint.

📸 **Screenshot:** Experiments tab with runs + metrics.
📸 **Screenshot:** Catalog → Models showing the `@champion` alias on the winning version.
📸 **Screenshot:** the cell output `PROMOTED: @champion now points to v2`.

### Step B5 — Serve + query the endpoint
Serving tab → confirm the `student-clearance` endpoint is **Ready** → click **Query endpoint**
and send a sample row (JSON is in `DATABRICKS_SETUP.md`).

📸 **Screenshot:** the serving endpoint in **Ready** state + a query response.

✅ **Track B done:** the full lifecycle running on cloud infrastructure.

---

## Step 3 — Assemble screenshots

Put every screenshot above into the `screenshots/` folder with clear names, e.g.
`01_dataset_approval.png`, `02_signal_check.png`, `03_baseline_registered.png`,
`04_inference.png`, `05_promotion.png`, `06_registry_after_promote.png`,
`07_databricks_serving.png`.

---

## Step 4 — Push to GitHub (assignment step 5)

```bash
cd student-clearance-mlflow
git init
git add .
git commit -m "MLflow student-clearance lifecycle: tracking, registry, serving, auto-promotion"
git branch -M main
git remote add origin https://github.com/<your-username>/student-clearance-mlflow.git
git push -u origin main
```
The `.gitignore` already excludes `mlruns/` and caches, so only your code + docs +
screenshots get pushed.

✅ **Done when:** the repo is visible on GitHub with README, code, and screenshots.
Confirm the due date/time on Campusweb and MS Teams before pushing the final version.

---

## Step 5 — Prepare the live presentation (assignment step 6)

Rehearse this ~5-minute flow:
1. State the problem + why (30s) — from `PROJECT_OBJECTIVE.md`.
2. Show the MLflow UI / Databricks Experiments with tracked runs (1 min).
3. Show the registered model + Production/`@champion` version (30s).
4. Run `promote.py` (or the Databricks cell) live to show **automatic** promotion (1 min).
5. Hit the serving endpoint live to return a prediction (1 min).
6. Explain the low F1 honestly using the `signal_check.py` evidence (1 min).

**Have a fallback:** if live serving is slow (Databricks scale-to-zero) or Wi-Fi fails,
keep your screenshots open in a tab to walk through instead.

---

## Quick command reference

| Goal | Command |
|---|---|
| Prove no signal | `python analysis/signal_check.py` |
| MLflow UI | `mlflow ui --port 5000` |
| Train + register baseline | `python src/train.py --model-type baseline` |
| Serve Production model | `mlflow models serve -m "models:/student-clearance-classifier/Production" -p 5001 --no-conda` |
| Test inference | `python src/predict_request.py` |
| Improve + auto-promote | `python src/promote.py --model-type logreg` |
