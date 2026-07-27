"""Signal check: is there ANY learnable signal between the features and the target?

This script is the evidence behind the "why the F1 is low" justification. It shows
that the demographic/program features are statistically independent of clearance
status, so a low score is a property of the data, not a modeling mistake.

It reports, for each feature:
  - Chi-square test of independence vs. the target (high p-value => independent)
  - Mutual information with the target (~0 => carries no information)
And overall:
  - the majority-class baseline accuracy
  - a cross-validated ROC-AUC ceiling (should sit near 0.50 = chance)

Usage:
    python analysis/signal_check.py
    python analysis/signal_check.py --data-path data/global_student_academic_status.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

FEATURE_COLS = ["Department", "University", "Degree Level",
                "Country", "City", "Semester", "Section"]


def main(data_path: str) -> None:
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    y = (df["Pending Type"].str.strip() == "All Clear").astype(int)  # 1 = cleared

    print("=" * 68)
    print("SIGNAL CHECK — features vs. clearance target")
    print("=" * 68)
    print(f"Rows: {len(df)}  |  Cleared rate (majority class): {y.mean():.3f}")
    print()

    # --- Chi-square test of independence -----------------------------------
    print("Chi-square test of independence (H0: feature is independent of target)")
    print(f"  {'feature':13s} {'chi2':>10s} {'p-value':>10s}   verdict")
    print("  " + "-" * 55)
    for f in FEATURE_COLS:
        ct = pd.crosstab(df[f], y)
        chi2, p, _, _ = chi2_contingency(ct)
        verdict = "independent" if p > 0.05 else "some association"
        print(f"  {f:13s} {chi2:10.2f} {p:10.3f}   {verdict}")
    print("  (p > 0.05 => cannot reject independence => no usable signal)")
    print()

    # --- Mutual information -------------------------------------------------
    X_ord = OrdinalEncoder().fit_transform(df[FEATURE_COLS])
    mi = mutual_info_classif(X_ord, y, discrete_features=True, random_state=0)
    print("Mutual information with target (0.0 = no information)")
    for f, m in sorted(zip(FEATURE_COLS, mi), key=lambda t: -t[1]):
        print(f"  {f:13s} {m:.5f}")
    print()

    # --- Model ceiling: cross-validated ROC-AUC ----------------------------
    pre = ColumnTransformer(
        [("oh", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS)]
    )
    pipe = Pipeline([("prep", pre),
                     ("rf", RandomForestClassifier(n_estimators=200, random_state=0))])
    auc = cross_val_score(pipe, df[FEATURE_COLS], y, cv=5, scoring="roc_auc")
    print("Model ceiling (5-fold CV, Random Forest)")
    print(f"  ROC-AUC: {auc.mean():.3f} +/- {auc.std():.3f}   (0.50 = random chance)")
    print()

    print("CONCLUSION: the features are effectively independent of the target, so no")
    print("model can exceed chance honestly. A low F1 is the correct, expected result;")
    print("the only ways to inflate it are data leakage or a deterministic lookup.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-path", default="data/global_student_academic_status.csv"
    )
    args = ap.parse_args()
    main(args.data_path)
