#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate all teacher-selection model predictions against the gold label.

Input scope:
    dataset/additional/teacher_selection_dataset/gold_with_all_models.parquet
    filtered to label_type == "gold" and split == "test"

Reference:
    grade

Model columns:
    all columns starting with "new_grade_" or "eval_grade_"

Outputs:
    results/tables/teacher_eval_outputs/teacher_eval_report.txt
    results/tables/teacher_eval_outputs/teacher_eval_summary.csv
    results/tables/teacher_eval_outputs/teacher_eval_confusion_matrices.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_PARQUET = (
    PROJECT_ROOT
    / "dataset"
    / "additional"
    / "teacher_selection_dataset"
    / "gold_with_all_models.parquet"
)

OUTPUT_DIR = SCRIPT_DIR / "teacher_eval_outputs"
REPORT_TXT = OUTPUT_DIR / "teacher_eval_report.txt"
SUMMARY_CSV = OUTPUT_DIR / "teacher_eval_summary.csv"
CONFUSION_CSV = OUTPUT_DIR / "teacher_eval_confusion_matrices.csv"

GOLD_COL = "grade"
LABEL_TYPE_COL = "label_type"
SPLIT_COL = "split"
QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"

MODEL_PREFIXES = ("new_grade_", "eval_grade_")


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in input parquet:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def infer_model_columns(df: pd.DataFrame) -> list[str]:
    model_cols: list[str] = []
    for col in df.columns:
        if not col.startswith(MODEL_PREFIXES):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            model_cols.append(col)
    return model_cols


def mae_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def mse_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(np.mean((y_true - y_pred) ** 2))


def ordinal_encode_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    mapping = {value: idx for idx, value in enumerate(values)}
    true_codes = np.array([mapping[value] for value in y_true], dtype=int)
    pred_codes = np.array([mapping[value] for value in y_pred], dtype=int)
    return true_codes, pred_codes


def tau_b_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return np.nan

    true_codes, pred_codes = ordinal_encode_pair(y_true, y_pred)

    if len(np.unique(true_codes)) <= 1 or len(np.unique(pred_codes)) <= 1:
        return np.nan

    n_classes = int(max(true_codes.max(), pred_codes.max()) + 1)
    contingency = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true_value, pred_value in zip(true_codes, pred_codes, strict=False):
        contingency[int(true_value), int(pred_value)] += 1

    concordant = 0
    discordant = 0

    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            count = int(contingency[true_idx, pred_idx])
            if count == 0:
                continue

            concordant += count * int(
                contingency[:true_idx, :pred_idx].sum()
                + contingency[true_idx + 1 :, pred_idx + 1 :].sum()
            )
            discordant += count * int(
                contingency[:true_idx, pred_idx + 1 :].sum()
                + contingency[true_idx + 1 :, :pred_idx].sum()
            )

    concordant //= 2
    discordant //= 2

    def pair_count(values: np.ndarray) -> int:
        values = values.astype(np.int64)
        return int(np.sum(values * (values - 1) // 2))

    tied_true = pair_count(contingency.sum(axis=1)) - pair_count(contingency.ravel())
    tied_pred = pair_count(contingency.sum(axis=0)) - pair_count(contingency.ravel())

    denominator = np.sqrt(
        float(concordant + discordant + tied_true)
        * float(concordant + discordant + tied_pred)
    )

    if denominator == 0:
        return np.nan

    return float((concordant - discordant) / denominator)


def qwk_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    true_codes, pred_codes = ordinal_encode_pair(y_true, y_pred)

    if len(np.unique(true_codes)) == 1 and len(np.unique(pred_codes)) == 1:
        return 1.0 if np.array_equal(true_codes, pred_codes) else 0.0

    n_classes = int(max(true_codes.max(), pred_codes.max()) + 1)
    observed = np.zeros((n_classes, n_classes), dtype=float)

    for true_value, pred_value in zip(true_codes, pred_codes, strict=False):
        observed[int(true_value), int(pred_value)] += 1.0

    total = observed.sum()
    if total == 0:
        return np.nan

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / total

    denom = float((n_classes - 1) ** 2)
    if denom == 0:
        return np.nan

    weights = np.fromfunction(
        lambda i, j: ((i - j) ** 2) / denom,
        (n_classes, n_classes),
        dtype=float,
    )

    observed_weighted = float((weights * observed).sum())
    expected_weighted = float((weights * expected).sum())

    if expected_weighted == 0:
        return 1.0 if np.array_equal(true_codes, pred_codes) else 0.0

    return 1.0 - (observed_weighted / expected_weighted)


def format_metric(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.6f}"


def evaluate_model(scope_df: pd.DataFrame, model_col: str) -> tuple[dict[str, Any], pd.DataFrame]:
    work_df = scope_df[[GOLD_COL, model_col, QUESTION_ID_COL, STUDENT_ID_COL]].copy()
    work_df[GOLD_COL] = pd.to_numeric(work_df[GOLD_COL], errors="coerce")
    work_df[model_col] = pd.to_numeric(work_df[model_col], errors="coerce")

    valid_df = work_df.dropna(subset=[GOLD_COL, model_col]).copy()
    y_true = valid_df[GOLD_COL].to_numpy(dtype=float)
    y_pred = valid_df[model_col].to_numpy(dtype=float)

    row = {
        "model": model_col,
        "source_rows": int(len(work_df)),
        "valid_rows": int(len(valid_df)),
        "missing_gold": int(work_df[GOLD_COL].isna().sum()),
        "missing_prediction": int(work_df[model_col].isna().sum()),
        "unique_questions": int(valid_df[QUESTION_ID_COL].nunique()),
        "unique_students": int(valid_df[STUDENT_ID_COL].nunique()),
        "mae": mae_safe(y_true, y_pred),
        "mse": mse_safe(y_true, y_pred),
        "tau_b": tau_b_safe(y_true, y_pred),
        "qwk": qwk_safe(y_true, y_pred),
    }

    confusion = pd.crosstab(
        valid_df[GOLD_COL],
        valid_df[model_col],
        rownames=["gold"],
        colnames=["prediction"],
        dropna=False,
    )
    confusion_long = confusion.stack().reset_index(name="count")
    confusion_long.insert(0, "model", model_col)

    return row, confusion_long


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PARQUET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PARQUET)

    required = [GOLD_COL, LABEL_TYPE_COL, SPLIT_COL, QUESTION_ID_COL, STUDENT_ID_COL]
    require_columns(df, required)

    model_cols = infer_model_columns(df)
    if not model_cols:
        raise ValueError(
            "No model prediction columns found. Expected columns starting with "
            f"{MODEL_PREFIXES}."
        )

    label_type = df[LABEL_TYPE_COL].astype("string").str.lower()
    split = df[SPLIT_COL].astype("string").str.lower()
    scope_df = df[(label_type == "gold") & (split == "test")].copy()

    if scope_df.empty:
        raise ValueError("No rows found after filtering label_type == 'gold' and split == 'test'.")

    rows: list[dict[str, Any]] = []
    confusion_frames: list[pd.DataFrame] = []

    for model_col in model_cols:
        row, confusion_long = evaluate_model(scope_df, model_col)
        rows.append(row)
        confusion_frames.append(confusion_long)

    summary_df = pd.DataFrame(rows).sort_values(
        by=["qwk", "tau_b", "mae"],
        ascending=[False, False, True],
        na_position="last",
    )
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    confusion_df = pd.concat(confusion_frames, ignore_index=True)
    confusion_df.to_csv(CONFUSION_CSV, index=False, encoding="utf-8")

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("TEACHER-SELECTION MODEL EVALUATION REPORT")
    lines.append("=" * 100)
    lines.append(f"Input parquet:          {INPUT_PARQUET}")
    lines.append(f"Gold/reference column:  {GOLD_COL}")
    lines.append("Filter:                 label_type == 'gold' and split == 'test'")
    lines.append("")
    lines.append("Scope:")
    lines.append(f"  Source rows:          {len(df)}")
    lines.append(f"  Filtered rows:        {len(scope_df)}")
    lines.append(f"  Unique questions:     {scope_df[QUESTION_ID_COL].nunique()}")
    lines.append(f"  Unique students:      {scope_df[STUDENT_ID_COL].nunique()}")
    lines.append("")
    lines.append("Detected model columns:")
    for model_col in model_cols:
        lines.append(f"  - {model_col}")
    lines.append("")
    lines.append("Metrics against gold:")
    lines.append(
        summary_df[
            [
                "model",
                "valid_rows",
                "mae",
                "mse",
                "tau_b",
                "qwk",
                "missing_prediction",
            ]
        ].to_string(
            index=False,
            formatters={
                "mae": format_metric,
                "mse": format_metric,
                "tau_b": format_metric,
                "qwk": format_metric,
            },
        )
    )
    lines.append("")
    lines.append("Written files:")
    lines.append(f"  {REPORT_TXT}")
    lines.append(f"  {SUMMARY_CSV}")
    lines.append(f"  {CONFUSION_CSV}")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print(f"  {REPORT_TXT}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {CONFUSION_CSV}")


if __name__ == "__main__":
    main()
