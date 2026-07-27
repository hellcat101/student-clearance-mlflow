# Databricks notebook source
# ============================================================================
# Student Academic Clearance — full MLflow lifecycle on Databricks (Free Edition)
# ----------------------------------------------------------------------------
# Run this as a Databricks NOTEBOOK (each `# COMMAND ----------` is a cell) on a
# serverless cluster. It uses the Unity Catalog model registry (MLflow 3), which
# replaces the deprecated Production/Staging *stages* with *aliases* (@champion).
#
# Lifecycle:
#   1. Load data (upload the CSV to a UC Volume or DBFS first — see setup guide)
#   2. Train a naive baseline, register to Unity Catalog, set alias @champion
#   3. Train an improved model; if it beats @champion on f1_at_risk, MOVE the
#      @champion alias to the new version  ->  automatic promotion
#   4. Serve the @champion model as a REST endpoint (UI or the cell at the end)
# ============================================================================

# COMMAND ----------
# MAGIC %pip install -q "mlflow>=3.0" scikit-learn pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# On Databricks the tracking URI is already "databricks"; point the REGISTRY at
# Unity Catalog so registered models live in a catalog.schema namespace.
mlflow.set_registry_uri("databricks-uc")

# ---- EDIT THESE THREE VALUES for your workspace ----------------------------
CATALOG = "workspace"          # your UC catalog (Free Edition default: "workspace")
SCHEMA = "default"             # your UC schema
CSV_PATH = "/Volumes/workspace/default/data/global_student_academic_status.csv"
# ---------------------------------------------------------------------------

MODEL_NAME = f"{CATALOG}.{SCHEMA}.student_clearance_classifier"
CHAMPION = "champion"          # alias marking the "production" model

# When run as a notebook, MLflow logs to the notebook's own experiment by
# default, so no set_experiment() call is needed. Uncomment to use a named one:
# mlflow.set_experiment("/Shared/student-clearance")

client = MlflowClient()

FEATURE_COLS = ["Department", "University", "Degree Level",
                "Country", "City", "Semester", "Section"]
PRIMARY_METRIC = "f1_at_risk"

# COMMAND ----------
# ----- data prep (NO leakage: drop id + target-derived columns) -------------
def get_split(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    X = df[FEATURE_COLS].copy()
    y = (df["Pending Type"].str.strip() == "All Clear").astype(int)  # 1=cleared
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocessor():
    return ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS)],
        remainder="drop",
    )

def build(model_type):
    if model_type == "baseline":
        return DummyClassifier(strategy="most_frequent"), {"strategy": "most_frequent"}
    if model_type == "logreg":
        p = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
        return LogisticRegression(**p), p
    if model_type == "rf":
        p = {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5,
             "class_weight": "balanced", "random_state": 42}
        return RandomForestClassifier(**p), p
    raise ValueError(model_type)

def evaluate(model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "recall_at_risk": recall_score(y, preds, pos_label=0),
        "precision_at_risk": precision_score(y, preds, pos_label=0, zero_division=0),
        "f1_at_risk": f1_score(y, preds, pos_label=0),
        "roc_auc": roc_auc_score(y, proba),
    }

X_train, X_test, y_train, y_test = get_split(CSV_PATH)
print("train", X_train.shape, "test", X_test.shape, "cleared rate", round(y_train.mean(), 3))

# COMMAND ----------
# ----- reusable train+register function -------------------------------------
def train_and_register(model_type):
    est, params = build(model_type)
    pipe = Pipeline([("prep", preprocessor()), ("clf", est)])
    with mlflow.start_run(run_name=model_type):
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            registered_model_name=MODEL_NAME,   # registers into Unity Catalog
            input_example=X_train.head(3),
        )
    version = client.search_model_versions(
        f"name='{MODEL_NAME}'", order_by=["version_number DESC"], max_results=1
    )[0].version
    print(f"[{model_type}] registered v{version} | {PRIMARY_METRIC}={metrics[PRIMARY_METRIC]:.4f}")
    return version, metrics[PRIMARY_METRIC]

def champion_score():
    """Score the current @champion on the test set, or (None, -1) if none."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except Exception:
        return None, -1.0
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
    return mv.version, evaluate(model, X_test, y_test)[PRIMARY_METRIC]

# COMMAND ----------
# ===== STEP 2 & 3: train naive baseline, register, make it @champion ========
base_v, base_score = train_and_register("baseline")
if champion_score()[0] is None:
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, base_v)
    print(f"Set @{CHAMPION} -> v{base_v} (initial production model)")

# COMMAND ----------
# ===== STEP 4: train improved model and AUTO-PROMOTE via the alias ==========
prod_v, prod_score = champion_score()
print(f"Current @{CHAMPION}: v{prod_v}  {PRIMARY_METRIC}={prod_score:.4f}")

cand_v, cand_score = train_and_register("logreg")   # the improvement
print(f"Challenger: v{cand_v}  {PRIMARY_METRIC}={cand_score:.4f}")

if cand_score > prod_score:
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, cand_v)  # move alias
    print(f"PROMOTED: @{CHAMPION} now points to v{cand_v} "
          f"(beat v{prod_v} by {cand_score - prod_score:+.4f})")
else:
    print("Challenger did not beat champion; @champion unchanged.")

# COMMAND ----------
# ===== STEP 3/4: expose @champion as a REST serving endpoint ================
# Easiest: UI -> Serving -> Create serving endpoint -> pick this UC model,
# version = @champion. Or do it programmatically:
from mlflow.deployments import get_deploy_client

deploy = get_deploy_client("databricks")
champ = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
try:
    deploy.create_endpoint(
        name="student-clearance",
        config={
            "served_entities": [{
                "entity_name": MODEL_NAME,
                "entity_version": champ.version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }]
        },
    )
    print("Serving endpoint 'student-clearance' created.")
except Exception as e:
    # If it already exists, update it to the current champion version instead.
    print("create failed (may already exist), updating config:", e)
    deploy.update_endpoint(
        endpoint="student-clearance",
        config={"served_entities": [{
            "entity_name": MODEL_NAME,
            "entity_version": champ.version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }]},
    )
    print("Serving endpoint updated to champion v", champ.version)
