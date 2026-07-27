"""Data loading and preprocessing for the student-clearance project.

Produces a clean train/test split with NO leakage:
  - identifier columns are dropped
  - the columns that are derived from the target are dropped
  - target = 1 if the student is "All Clear", else 0
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Columns we must never feed the model.
ID_COLS = ["Sr No.", "Student Name", "Registration No."]
LEAKAGE_COLS = ["No. of Pending Courses", "Pending Courses Name"]
TARGET_COL = "Pending Type"

# The only features the model is allowed to see.
FEATURE_COLS = [
    "Department",
    "University",
    "Degree Level",
    "Country",
    "City",
    "Semester",
    "Section",
]

RANDOM_STATE = 42


def load_dataframe(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 = cleared (All Clear), 0 = has a pending issue."""
    return (df[TARGET_COL].str.strip() == "All Clear").astype(int)


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode every (categorical) feature column."""
    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURE_COLS,
            )
        ],
        remainder="drop",
    )


def get_train_test(data_path: str, test_size: float = 0.2):
    df = load_dataframe(data_path)
    X = df[FEATURE_COLS].copy()
    y = make_target(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


if __name__ == "__main__":
    # Quick sanity check.
    Xtr, Xte, ytr, yte = get_train_test("data/global_student_academic_status.csv")
    print("Train:", Xtr.shape, "Test:", Xte.shape)
    print("Cleared rate (train):", round(ytr.mean(), 3))
