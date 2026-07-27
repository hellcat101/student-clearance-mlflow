"""Train a model, log everything to MLflow, and register it.

Step 2 (modeling) + Step 3 (register) of the assignment.

Usage:
    python src/train.py --model-type logreg
    python src/train.py --model-type rf

Every run is tracked in MLflow (params, metrics, the fitted sklearn pipeline).
The fitted model is also registered in the Model Registry under REGISTERED_NAME.
The FIRST registered version is moved to the "Production" stage so there is
always a baseline in production for the promotion step to compare against.
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data_prep import build_preprocessor, get_train_test

REGISTERED_NAME = "student-clearance-classifier"
EXPERIMENT_NAME = "student-clearance"


def build_estimator(model_type: str):
    """Return an (unfitted) classifier and its hyper-params for logging."""
    if model_type == "baseline":
        # Naive reference model: always predicts the majority class ("cleared").
        # This is the "everyone is cleared" strawman we must beat: it scores ~65%
        # accuracy but 0% recall on at-risk students (f1_at_risk = 0).
        params = {"strategy": "most_frequent"}
        return DummyClassifier(**params), params
    if model_type == "logreg":
        params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
        return LogisticRegression(**params), params
    if model_type == "rf":
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
        }
        return RandomForestClassifier(**params), params
    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate(model, X, y) -> dict:
    """Metrics for the POSITIVE-of-interest class = 'not cleared' (label 0).

    We report recall/precision/f1 for the at-risk students (y == 0), because
    catching them is the point of the model. pos_label=0 makes that explicit.
    """
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]  # P(cleared)
    return {
        "accuracy": accuracy_score(y, preds),
        # metrics for the "not cleared" class (at-risk students)
        "recall_at_risk": recall_score(y, preds, pos_label=0),
        "precision_at_risk": precision_score(y, preds, pos_label=0, zero_division=0),
        "f1_at_risk": f1_score(y, preds, pos_label=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def train(model_type: str, data_path: str) -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = get_train_test(data_path)

    estimator, params = build_estimator(model_type)
    pipe = Pipeline(
        steps=[("prep", build_preprocessor()), ("clf", estimator)]
    )

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)

        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log + register the fitted pipeline in one call.
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=REGISTERED_NAME,
        )

        print(f"[train] run_id={run.info.run_id}")
        for k, v in metrics.items():
            print(f"  {k:18s}: {v:.4f}")

    # Ensure there is always a Production baseline: if nothing is in
    # Production yet, promote the version we just created.
    client = MlflowClient()
    prod = client.get_latest_versions(REGISTERED_NAME, stages=["Production"])
    if not prod:
        latest = client.get_latest_versions(REGISTERED_NAME, stages=["None"])[-1]
        client.transition_model_version_stage(
            REGISTERED_NAME, latest.version, "Production"
        )
        print(f"[train] no Production model existed -> promoted v{latest.version}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-type", default="baseline", choices=["baseline", "logreg", "rf"]
    )
    ap.add_argument(
        "--data-path", default="data/global_student_academic_status.csv"
    )
    args = ap.parse_args()
    train(args.model_type, args.data_path)
