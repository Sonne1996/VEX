from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

INPUT_PARQUET = Path("../dataset/v1_0_release/v1_0_stable.parquet")
OUTPUT_PARQUET = Path("result_global_prior.parquet")

GRADE_COL = "grade"
SPLIT_COL = "split"
ID_COL = "grading_id"

PRED_COL = "grade_prior_global"

ALLOWED_GRADES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)


# =========================================================
# HELPERS
# =========================================================

def open_parquet_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Input file must be a parquet file, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_parquet(path)


def save_parquet(path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved parquet to: {path.resolve()}")


def round_to_allowed_grade(value: float) -> float:
    return float(ALLOWED_GRADES[np.argmin(np.abs(ALLOWED_GRADES - value))])


# =========================================================
# CORE LOGIC
# =========================================================

def fit_global_prior(train_df: pd.DataFrame, grade_col: str) -> float:
    """
    Learns one constant grade from the training data.
    """
    train_grades = pd.to_numeric(train_df[grade_col], errors="coerce").dropna()

    if train_grades.empty:
        raise ValueError("Training split contains no valid grades.")

    mean_grade = float(train_grades.mean())
    return round_to_allowed_grade(mean_grade)


def predict_global_prior(df: pd.DataFrame, constant_grade: float) -> pd.Series:
    """
    Predicts the same constant grade for all rows.
    """
    return pd.Series([constant_grade] * len(df), index=df.index, dtype=float)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    df = open_parquet_file(INPUT_PARQUET)

    required_cols = [ID_COL, SPLIT_COL, GRADE_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    train_df = df[df[SPLIT_COL] == "train"].copy()
    test_mask = df[SPLIT_COL] == "test"

    if train_df.empty:
        raise ValueError("No train rows found.")
    if not test_mask.any():
        raise ValueError("No test rows found.")

    constant_grade = fit_global_prior(train_df, GRADE_COL)
    print(f"Learned global prior grade: {constant_grade}")

    df[PRED_COL] = np.nan
    df.loc[test_mask, PRED_COL] = predict_global_prior(df.loc[test_mask], constant_grade)

    save_parquet(OUTPUT_PARQUET, df)

    print("\nSummary:")
    print(df[[SPLIT_COL, PRED_COL]].groupby(SPLIT_COL).count())
    print(df.loc[test_mask, PRED_COL].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()