# Project Objective — Student Academic Clearance Prediction (MLflow Lifecycle)

**Course module:** Production-grade ML with MLflow
**Dataset:** [University Academic Clearances & Pending Courses](https://www.kaggle.com/datasets/fa23bst011/university-academic-clearances-and-pending-courses) (Kaggle) — *approved by professor*
**Size:** 10,000 rows × 13 columns

## Problem statement

Build a production-grade MLflow pipeline that predicts a student's **academic clearance status**
— *cleared* vs. *not cleared* — from their program and demographic profile, so an institution
could proactively flag at-risk students for advising intervention.

## Task framing

**Binary classification.**

- **Target:** `Pending Type` → `1` if "All Clear", else `0` (has a pending issue: Fail / Withdraw / Dropped / Not Registered).
- **Class balance:** ~65% cleared, ~35% not cleared (mild imbalance).
- **Features used:** Department, University, Degree Level, Country, City, Semester, Section.
- **Dropped — identifiers:** Sr No., Student Name, Registration No.
- **Dropped — leakage:** `No. of Pending Courses`, `Pending Courses Name` (both are derived from the target; including them would make the model trivially perfect).

## Why the "good model" is good (justification)

Because missing an at-risk student is costlier than a false alarm, the model is judged primarily on
**recall and F1 for the "not cleared" class**, not raw accuracy. A baseline that predicts "everyone
is cleared" would score 65% accuracy but 0% recall on the students who actually need help — useless.
The chosen model must beat that on F1 / recall while keeping precision reasonable.

## Lifecycle plan (maps to assignment steps)

1. **Dataset** — confirmed & approved.
2. **Modeling** — baseline Logistic Regression with a clean preprocessing pipeline, all runs tracked in MLflow.
3. **Register & serve** — best model registered in the MLflow Model Registry and exposed for inference via a REST endpoint.
4. **Improve & auto-promote** — a tuned tree-based model (Random Forest / Gradient Boosting) is trained; if it beats the current Production model on F1, it is **automatically promoted** to Production and re-served.
5. **Submit** — GitHub repo with code + screenshots of each objective.
6. **Present** — live demo of the running pipeline.

## Model justification & limitations (why the F1 is low — and why that's correct)

The chosen model achieves a modest F1 on the "at-risk" class (≈0.41) with ROC-AUC ≈ 0.50.
This is **not a modeling failure** — it reflects a verified property of the dataset. Before and
after training I tested whether the features carry any signal about the target
(see `analysis/signal_check.py`, reproducible):

| Feature | Chi-square p-value | Mutual information | Reading |
|---|---|---|---|
| Department | 0.965 | 0.0002 | independent |
| University | 0.301 | 0.0024 | independent |
| Degree Level | 0.336 | 0.0002 | independent |
| Country | 0.614 | 0.0004 | independent |
| City | 0.011 | 0.0033 | negligible (spurious under multiple testing) |
| Semester | 0.611 | 0.0003 | independent |
| Section | 0.199 | 0.0002 | independent |

A 5-fold cross-validated Random Forest reaches **ROC-AUC = 0.506 ± 0.017** — i.e. chance.
The features are statistically independent of clearance status, so **no model can exceed
chance honestly**.

The only ways to force a higher score would be:

1. **Data leakage** — reintroducing `No. of Pending Courses` / `Pending Courses Name`, which
   are derived from the target and inflate F1 to ~1.0 dishonestly. Deliberately avoided.
2. **A deterministic lookup** — e.g. University → Country is 100% accurate, but it is a trivial
   dictionary, not a predictive model.

Accordingly, the "good model" is justified **not by an absolute threshold** but by beating a
naive majority-class baseline (F1 = 0.00 on at-risk students) on the metric that matters
(recall/F1 for the at-risk class). It does. Documenting this ceiling with statistical evidence
is itself sound data-science practice.

## Honest caveat

The dataset appears **synthetic**, so the absence of signal is expected. That is acceptable
here: the graded deliverable is the **MLflow lifecycle** (tracking → registry → serving →
automated promotion), not state-of-the-art accuracy.
