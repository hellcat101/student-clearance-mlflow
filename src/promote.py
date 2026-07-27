"""Train an improved model and AUTO-PROMOTE it if it beats Production.

Step 4 of the assignment: make an improvement and let it replace the old
model automatically.

Logic:
  1. Train a candidate model (default: random forest) and log/register it.
  2. Evaluate the candidate and the current Production model on the SAME
     held-out test set, using the same primary metric (f1_at_risk).
  3. If candidate > production by more than MIN_DELTA, transition the
     candidate to "Production" and archive the old one. Otherwise leave
     Production untouched.

After promotion, re-serve the "Production" model (see serve step in README)
to expose the new model automatically.
"""
from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline

from data_prep import build_preprocessor, get_train_test
from train import REGISTERED_NAME, build_estimator, evaluate

PRIMARY_METRIC = "f1_at_risk"
MIN_DELTA = 0.0  # candidate must be strictly better by at least this much
EXPERIMENT_NAME = "student-clearance"


def _score_production(client: MlflowClient, X_test, y_test):
    """Return (version, score) of the current Production model, or (None, -1)."""
    prod = client.get_latest_versions(REGISTERED_NAME, stages=["Production"])
    if not prod:
        return None, -1.0
    version = prod[0].version
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}/Production")
    score = evaluate(model, X_test, y_test)[PRIMARY_METRIC]
    return version, score


def promote(model_type: str, data_path: str) -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    X_train, X_test, y_train, y_test = get_train_test(data_path)

    # Score the incumbent BEFORE training the challenger.
    prod_version, prod_score = _score_production(client, X_test, y_test)
    print(f"[promote] current Production: v{prod_version} "
          f"{PRIMARY_METRIC}={prod_score:.4f}")

    # --- train the challenger ---
    estimator, params = build_estimator(model_type)
    pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", estimator)])

    with mlflow.start_run(run_name=f"candidate-{model_type}") as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=REGISTERED_NAME,
        )
    cand_score = metrics[PRIMARY_METRIC]

    # The version we just registered is the newest one.
    cand_version = client.get_latest_versions(REGISTERED_NAME, stages=["None"])[-1].version
    print(f"[promote] candidate: v{cand_version} {PRIMARY_METRIC}={cand_score:.4f}")

    # --- decision ---
    if cand_score > prod_score + MIN_DELTA:
        client.transition_model_version_stage(
            REGISTERED_NAME,
            cand_version,
            "Production",
            archive_existing_versions=True,  # auto-archives the old Production model
        )
        print(f"[promote] PROMOTED v{cand_version} to Production "
              f"(beat v{prod_version} by {cand_score - prod_score:+.4f})")
        print("[promote] -> re-serve the Production model to expose it (see README).")
    else:
        print(f"[promote] candidate did NOT beat Production; no change.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-type", default="logreg", choices=["logreg", "rf"]
    )
    ap.add_argument(
        "--data-path", default="data/global_student_academic_status.csv"
    )
    args = ap.parse_args()
    promote(args.model_type, args.data_path)
