# Running the project on Databricks (Free Edition)

Databricks has MLflow built in — tracking, the Unity Catalog model registry, and REST
model serving — and it all works on the **free** tier with serverless compute. This guide
takes you from zero to a served, auto-promoting model.

> **Key difference vs. the local version:** Databricks uses the **Unity Catalog** registry
> (MLflow 3), which replaces the old `Production` / `Staging` **stages** with **aliases**
> like `@champion`. The local scripts (`src/`) use stages; the Databricks notebook
> (`databricks/mlflow_pipeline_databricks.py`) uses the `@champion` alias. Same lifecycle,
> registry-appropriate mechanics. Use the notebook on Databricks — not the `src/` scripts.

## 1. Create a free workspace

1. Go to the Databricks **Free Edition** sign-up and create an account (Google/email).
2. You land in a workspace that already has serverless compute and Unity Catalog enabled.
   The default catalog is usually named `workspace` and it has a `default` schema.

## 2. Upload the dataset to a Unity Catalog Volume

The notebook reads the CSV from a UC Volume path.

1. In the left sidebar: **Catalog** → expand `workspace` → `default` → **Create → Volume**,
   name it `data`.
2. Open the volume and **Upload** `data/global_student_academic_status.csv` into it.
3. The resulting path will be:
   `/Volumes/workspace/default/data/global_student_academic_status.csv`
   — this matches `CSV_PATH` in the notebook. (Adjust if your catalog/schema differ.)

## 3. Import and run the notebook

1. **Workspace** → your home folder → **Import** →
   upload `databricks/mlflow_pipeline_databricks.py`. Databricks recognizes the
   `# COMMAND ----------` markers and imports it as a notebook with cells.
2. At the top of the notebook, confirm/edit these three values to match your workspace:
   `CATALOG`, `SCHEMA`, `CSV_PATH`.
3. Attach it to **Serverless** compute and **Run all**.

What the cells do, mapped to the assignment:

| Cell | Assignment step | What happens |
|------|-----------------|--------------|
| install + imports | — | installs MLflow 3, points the registry at `databricks-uc` |
| data prep | 2 | leakage-safe train/test split |
| train baseline | 2 & 3 | registers `workspace.default.student_clearance_classifier` v1, sets alias `@champion` |
| improve + promote | 4 | trains logistic regression; if it beats `@champion` on `f1_at_risk`, **moves `@champion` to the new version** |
| serve | 3 & 4 | creates/updates a REST serving endpoint pointing at `@champion` |

## 4. Take your submission screenshots here

- **Experiments** tab → the runs with their metrics (`f1_at_risk`, `roc_auc`, …).
- **Catalog → Models** → `student_clearance_classifier` showing versions and the
  `@champion` alias on the winning version.
- **Serving** tab → the `student-clearance` endpoint in **Ready** state.
- The notebook cell output showing `PROMOTED: @champion now points to v2`.

## 5. Query the served endpoint (inference)

From the endpoint page, click **Query endpoint**, or call it with a token:

```bash
curl -X POST \
  https://<your-workspace-host>/serving-endpoints/student-clearance/invocations \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["Department","University","Degree Level","Country","City","Semester","Section"],
          "data": [["Mathematics","Massachusetts Institute of Technology (MIT)","MS","United States","Cambridge","3rd","A"]]
        }
      }'
```

## Alternative: run locally but track to Databricks

If you'd rather keep running the `src/` scripts from your laptop but store everything in
Databricks, you can, but note the stages-vs-aliases difference above means the auto-promote
in `src/promote.py` (which uses stages) won't behave correctly against a UC registry. For a
clean Databricks demo, use the notebook. To only send *tracking* (runs/metrics) to
Databricks while keeping a local registry, set:

```bash
export DATABRICKS_HOST="https://<your-workspace-host>"
export DATABRICKS_TOKEN="<personal-access-token>"   # User Settings -> Developer -> Access tokens
export MLFLOW_TRACKING_URI="databricks"
```

## Notes / gotchas

- **MLflow 3 required** for the alias-based UC workflow (`pip install "mlflow>=3.0"`). The
  notebook installs it in the first cell.
- Free Edition serving uses **scale-to-zero**, so the first request after idle may be slow
  while the endpoint wakes up — fine for a demo, just click **Query** once to warm it.
- Model names in Unity Catalog are **three-level**: `catalog.schema.model`.
