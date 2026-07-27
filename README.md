# Student Academic Clearance — MLflow Lifecycle Project

Production-grade classic ML project built with [MLflow](https://mlflow.org/): experiment
tracking → model registry → serving → **automated model promotion**.

**Objective:** predict whether a student is academically **cleared** vs. **at-risk**
(has a pending course issue) from their program/demographic profile, so an institution
could flag at-risk students for advising. See [`PROJECT_OBJECTIVE.md`](PROJECT_OBJECTIVE.md).

## Repository layout

```
student-clearance-mlflow/
├── data/
│   └── global_student_academic_status.csv   # dataset (Kaggle, professor-approved)
├── src/
│   ├── data_prep.py        # load + split, NO leakage (drops derived/id columns)
│   ├── train.py            # train, track in MLflow, register model
│   ├── promote.py          # train improved model, auto-promote if it beats Production
│   └── predict_request.py  # sample REST inference call against the served model
├── screenshots/            # put your progress screenshots here for submission
├── MLproject               # MLflow project entry points
├── python_env.yaml
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Launch the MLflow UI (optional but recommended for screenshots)

```bash
mlflow ui --port 5000
# open http://127.0.0.1:5000
```

Runs, params, metrics, and registered model versions all appear here.

## 2. Train the baseline + register (Steps 2 & 3)

```bash
cd student-clearance-mlflow
python src/train.py --model-type baseline
```

This logs a run, registers `student-clearance-classifier`, and — because nothing is in
Production yet — promotes this first version to the **Production** stage so there is a
baseline to challenge later. The `baseline` model is a naive majority-class classifier
(predicts "cleared" for everyone) — the strawman the real model must beat.

**Why this model is "good":** it is judged on **recall / F1 for the at-risk class**
(`recall_at_risk`, `f1_at_risk`), not raw accuracy. A naive "everyone is cleared" model
scores ~65% accuracy but 0% recall on students who actually need help; the baseline must
beat that. Metrics for every run are tracked in MLflow so the comparison is auditable.

## 3. Serve the Production model for inference (Step 3)

```bash
mlflow models serve -m "models:/student-clearance-classifier/Production" -p 5001 --no-conda
```

In another terminal, send a request:

```bash
python src/predict_request.py
```

## 4. Improve and AUTO-PROMOTE (Step 4)

```bash
python src/promote.py --model-type logreg   # or: --model-type rf
```

`promote.py` trains a stronger model (logistic regression / random forest with class
weighting), scores both the
challenger and the current Production model on the same held-out test set using `f1_at_risk`,
and — **only if the challenger wins** — transitions it to Production and auto-archives the
old version. Re-run the `mlflow models serve` command above to expose the new model.

To fully automate exposure, wrap the serve command in a script that runs after `promote.py`
reports a promotion (e.g. a shell script or CI job).

## 5. Submission checklist (Step 5)

Capture screenshots into `screenshots/` showing:

- [ ] MLflow UI with multiple tracked runs and their metrics
- [ ] The registered model with a Production version (baseline)
- [ ] A successful inference response from the served endpoint
- [ ] `promote.py` output showing the challenger beating and replacing Production
- [ ] The registry after promotion (new Production version, old one Archived)

Push everything to GitHub before the due date.

## Notes

- **No data leakage:** `No. of Pending Courses` and `Pending Courses Name` are dropped
  because they are derived from the target.
- The dataset appears synthetic, so absolute accuracy is modest — the graded deliverable
  is the **lifecycle**, not the score.
- **Cloud (Databricks):** to run this on Databricks Free Edition — with the Unity Catalog
  registry and REST model serving — use the notebook in `databricks/` and follow
  [`DATABRICKS_SETUP.md`](DATABRICKS_SETUP.md). Note that Databricks' UC registry uses
  aliases (`@champion`) instead of the stages the local `src/` scripts use.
