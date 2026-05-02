#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from vex_config import (
    INPUT_PARQUET,
    TEST_ENV_FOLDER,
    TESTS_DATAFRAME,
    OUTPUT_REPORT_FILE,
    TEST_ENV_METRICS_FOLDER,
    OUTPUT_PARQUET,
    MODEL_COLUMNS,
    LINEAR_MIN_GRADE,
    LINEAR_MAX_GRADE,
    LINEAR_ROUNDING_STEP,
    LINEAR_PASS_THRESHOLD_NORM,
    LINEAR_DATAFRAME_FILE,
    DISTRIBUTION_DATAFRAME_FILE,
    DISTRIBUTION_PASS_THRESHOLD_NORM,
    DISTRIBUTION_PASSING_DISTRIBUTION,
    DISTRIBUTION_PASSING_LABELS,
    DISTRIBUTION_FAIL_LABEL,
    DISTRIBUTION_ORDERED_LABELS,
    TESTS_ROOT_FOLDER,
    TEST_RUN_FOLDER,
    TEST_METRICS_FOLDER,
    TEST_METRICS_FILE,
    WRITE_SCALE_SANITY_EXPORTS,
)

# =========================================================
# PERFORMANCE CONFIG
# =========================================================

# Set to 1 for fully sequential execution.
# Set to e.g. 4, 8, 12 depending on your CPU/RAM.
EVAL_WORKERS = 16

# If True, shows progress bars for item-level and exam-level evaluation.
SHOW_PROGRESS = True


# =========================================================
# COLUMN CONFIG
# =========================================================

TEST_ID_COL = "test_id"
TEST_SIZE_COL = "test_size"
QUESTION_ORDER_COL = "question_order"
QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"
ANSWER_ID_COL = "answer_id"

EXAM_LEVEL_COUNT_COLUMNS = [
    "n_students",
    "students_raw",
    "students_complete",
    "students_dropped_incomplete",
    "students_missing_human",
    "students_missing_prediction",
]

EXAM_LEVEL_METRIC_COLUMNS = [
    "el_tau_b",
    "el_acc_linear_abs",
    "el_qwk_linear_abs",
    "el_pass_acc_linear_abs",
    "el_pass_qwk_linear_abs",
    "el_acc_linear_mean",
    "el_qwk_linear_mean",
    "el_pass_acc_linear_mean",
    "el_pass_qwk_linear_mean",
    "el_acc_bologna",
    "el_qwk_bologna",
]

# In dataframe_env.parquet
HUMAN_GRADE_COL = "human_grade"

# In original INPUT_PARQUET
INPUT_HUMAN_GRADE_COL = "grade"

HUMAN_ONE_GOLD = "human_expert_one_gold"
HUMAND_TWO_MODEL = "human_expert_two"

HUMAN_COMPARISON_MODEL_COLUMNS = [HUMAND_TWO_MODEL]
EVALUATION_MODEL_COLUMNS = list(MODEL_COLUMNS) + HUMAN_COMPARISON_MODEL_COLUMNS


def _reference_col_for_model(model_col: str) -> str:
    if model_col in HUMAN_COMPARISON_MODEL_COLUMNS:
        return HUMAN_ONE_GOLD
    return HUMAN_GRADE_COL


# =========================================================
# PATH HELPERS
# =========================================================

def _test_env_root() -> Path:
    return Path(TEST_ENV_FOLDER)


def _dataframe_dir() -> Path:
    return _test_env_root() / TESTS_DATAFRAME


def _env_metrics_dir() -> Path:
    return _test_env_root() / TEST_ENV_METRICS_FOLDER


def _tests_root_dir() -> Path:
    return _test_env_root() / TESTS_ROOT_FOLDER


def _run_dir(test_number: int | str) -> Path:
    return _tests_root_dir() / TEST_RUN_FOLDER.format(test_number=test_number)


def _run_metrics_dir(test_number: int | str) -> Path:
    return _run_dir(test_number) / TEST_METRICS_FOLDER


def _run_metrics_file(test_number: int | str) -> Path:
    return _run_metrics_dir(test_number) / TEST_METRICS_FILE.format(
        test_number=test_number
    )


def _input_env_parquet() -> Path:
    return Path(OUTPUT_PARQUET)


def _input_original_parquet() -> Path:
    return Path(INPUT_PARQUET)


def _output_report_path() -> Path:
    return _env_metrics_dir() / OUTPUT_REPORT_FILE


def _exam_results_path() -> Path:
    return _env_metrics_dir() / "exam_level_precomputed_metrics.parquet"


def _exam_summary_by_test_size_path() -> Path:
    return _env_metrics_dir() / "exam_level_summary_by_test_size.parquet"


def _exam_summary_all_path() -> Path:
    return _env_metrics_dir() / "exam_level_summary_all.parquet"


def _env_exam_metrics_wide_path() -> Path:
    return _dataframe_dir() / "dataframe_env_exam_metrics_wide.parquet"


def _linear_size_dir(test_number: int | str, test_size: int) -> Path:
    return _run_dir(test_number) / "linear" / str(test_size)


def _bologna_size_dir(test_number: int | str, test_size: int) -> Path:
    return _run_dir(test_number) / "bologna" / str(test_size)


def _safe_file_token(value: str) -> str:
    token = str(value).strip()
    token = token.replace("/", "_")
    token = token.replace("\\", "_")
    token = token.replace(" ", "_")
    token = token.replace(":", "_")
    token = token.replace("*", "_")
    token = token.replace("?", "_")
    token = token.replace('"', "_")
    token = token.replace("<", "_")
    token = token.replace(">", "_")
    token = token.replace("|", "_")
    return token


def _linear_dataframe_path(
    test_number: int | str,
    test_size: int,
    h_or_m: str,
) -> Path:
    filename = LINEAR_DATAFRAME_FILE.format(
        h_or_m=_safe_file_token(h_or_m)
    )
    return _linear_size_dir(test_number, test_size) / filename


def _bologna_dataframe_path(
    test_number: int | str,
    test_size: int,
    h_or_m: str,
) -> Path:
    filename = DISTRIBUTION_DATAFRAME_FILE.format(
        h_or_m=_safe_file_token(h_or_m)
    )
    return _bologna_size_dir(test_number, test_size) / filename


# =========================================================
# VALIDATION
# =========================================================

def _validate_env_df(df: pd.DataFrame) -> None:
    required = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        ANSWER_ID_COL,
        HUMAN_GRADE_COL,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Pflichtspalten fehlen im dataframe_env.parquet: {missing}"
        )

    missing_models = [col for col in EVALUATION_MODEL_COLUMNS if col not in df.columns]
    if missing_models:
        raise ValueError(
            f"Folgende Modelspalten fehlen im dataframe_env.parquet: {missing_models}"
        )


def _validate_original_item_df(df: pd.DataFrame) -> None:
    required = [
        ANSWER_ID_COL,
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        INPUT_HUMAN_GRADE_COL,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Pflichtspalten fehlen im originalen INPUT_PARQUET: {missing}"
        )

    missing_models = [col for col in EVALUATION_MODEL_COLUMNS if col not in df.columns]
    if missing_models:
        raise ValueError(
            f"Folgende Modelspalten fehlen im originalen INPUT_PARQUET: {missing_models}"
        )


def _normalize_string_series(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip()


def _assert_no_duplicate_student_question_pairs(df: pd.DataFrame, context: str) -> None:
    """
    Hard guard:
    In a virtual exam / item set, each student-question pair must correspond
    to exactly one item. If duplicates exist, totals can be inflated.
    """
    key_cols = [STUDENT_ID_COL, QUESTION_ID_COL]

    missing = [col for col in key_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{context}: Spalten fehlen für Duplicate-Check: {missing}")

    duplicate_mask = df.duplicated(subset=key_cols, keep=False)

    if duplicate_mask.any():
        preview_cols = key_cols.copy()
        if ANSWER_ID_COL in df.columns:
            preview_cols.append(ANSWER_ID_COL)

        duplicates = (
            df.loc[duplicate_mask, preview_cols]
            .sort_values(key_cols)
            .head(50)
        )

        raise ValueError(
            f"{context}: Doppelte (member_id, question_id)-Paare gefunden. "
            f"Das ist für VEX nicht erlaubt, weil Exam-Totals sonst verfälscht werden.\n"
            f"Beispiele:\n{duplicates.to_string(index=False)}"
        )


def _assert_no_duplicate_exam_student_question_pairs(df_env: pd.DataFrame) -> None:
    duplicate_mask = df_env.duplicated(
        subset=[TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            df_env.loc[
                duplicate_mask,
                [
                    TEST_ID_COL,
                    TEST_SIZE_COL,
                    STUDENT_ID_COL,
                    QUESTION_ID_COL,
                    ANSWER_ID_COL,
                ],
            ]
            .sort_values([TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL])
            .head(100)
        )

        raise ValueError(
            "dataframe_env.parquet enthält doppelte "
            "(test_id, test_size, member_id, question_id)-Paare. "
            "Das macht Exam-Level-Summen ungültig.\n"
            f"Beispiele:\n{duplicates.to_string(index=False)}"
        )


# =========================================================
# METRIC HELPERS
# =========================================================

def _mae_safe(y_true: list[Any] | np.ndarray, y_pred: list[Any] | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    return float(np.mean(np.abs(y_true - y_pred)))


def _mse_safe(y_true: list[Any] | np.ndarray, y_pred: list[Any] | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    return float(np.mean((y_true - y_pred) ** 2))


def _rmse_safe(y_true: list[Any] | np.ndarray, y_pred: list[Any] | np.ndarray) -> float:
    mse = _mse_safe(y_true, y_pred)

    if pd.isna(mse):
        return np.nan

    return float(np.sqrt(mse))


def _accuracy_safe(y_true: list[Any] | np.ndarray, y_pred: list[Any] | np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    return float(np.mean(y_true == y_pred))


def _kendall_tau_b_safe(
    y_true: list[Any] | np.ndarray,
    y_pred: list[Any] | np.ndarray,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return np.nan

    try:
        tau, _ = kendalltau(y_true, y_pred)
        return float(tau) if pd.notna(tau) else np.nan
    except Exception:
        return np.nan


def _ordinal_encode_pair(
    y_true: list[Any] | np.ndarray,
    y_pred: list[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    combined = np.concatenate([y_true_arr, y_pred_arr])
    unique_sorted = sorted(pd.unique(combined))

    mapping = {label: idx for idx, label in enumerate(unique_sorted)}

    y_true_enc = np.array([mapping[x] for x in y_true_arr], dtype=int)
    y_pred_enc = np.array([mapping[x] for x in y_pred_arr], dtype=int)

    return y_true_enc, y_pred_enc


def _qwk_safe(y_true: list[Any] | np.ndarray, y_pred: list[Any] | np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    try:
        y_true_enc, y_pred_enc = _ordinal_encode_pair(y_true, y_pred)
        return float(cohen_kappa_score(y_true_enc, y_pred_enc, weights="quadratic"))
    except Exception:
        return np.nan


def _mean_safe(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def _std_safe(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) < 2:
        return np.nan

    return float(values.std(ddof=1))


# =========================================================
# LINEAR SCALE
# =========================================================

def _round_and_clip_linear_grades(grades: pd.Series) -> pd.Series:
    grades = pd.to_numeric(grades, errors="coerce")

    if LINEAR_ROUNDING_STEP and LINEAR_ROUNDING_STEP > 0:
        grades = (grades / LINEAR_ROUNDING_STEP).round() * LINEAR_ROUNDING_STEP

    grades = grades.clip(lower=LINEAR_MIN_GRADE, upper=LINEAR_MAX_GRADE)

    return grades


def _normalized_to_linear_grade_absolute(series: pd.Series) -> pd.Series:
    """
    Absolute linear scale:
        grade = min_grade + (max_grade - min_grade) * normalized_score

    With Swiss 1-6 grading:
        grade = 1 + 5 * normalized_score

    This is criterion-referenced and independent of the cohort distribution.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    grades = LINEAR_MIN_GRADE + (
        (LINEAR_MAX_GRADE - LINEAR_MIN_GRADE) * numeric
    )

    return _round_and_clip_linear_grades(grades)


def _normalized_to_pass_fail_absolute(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric >= LINEAR_PASS_THRESHOLD_NORM).astype("Int64")


def _totals_to_linear_grade_mean_centered(
    totals: pd.Series,
    max_total: float,
) -> pd.Series:
    """
    Mean-centered linear scale:

    If T <= mean:
        z = 0.5 * T / mean
    If T > mean:
        z = 0.5 + 0.5 * (T - mean) / (max_total - mean)

    Important:
    This is cohort-dependent. Therefore it is not an absolute grading scale.
    It is useful as a sensitivity analysis, not as the main grading scheme.
    """
    totals = pd.to_numeric(totals, errors="coerce")
    mean_total = totals.mean(skipna=True)

    if pd.isna(mean_total):
        return pd.Series(np.nan, index=totals.index, dtype="float64")

    if mean_total <= 0:
        z = pd.Series(0.0, index=totals.index, dtype="float64")
        z.loc[totals > 0] = 1.0
    elif mean_total >= max_total:
        z = 0.5 * totals / mean_total
    else:
        z = pd.Series(np.nan, index=totals.index, dtype="float64")

        lower_mask = totals <= mean_total
        upper_mask = totals > mean_total

        z.loc[lower_mask] = 0.5 * (totals.loc[lower_mask] / mean_total)
        z.loc[upper_mask] = 0.5 + 0.5 * (
            (totals.loc[upper_mask] - mean_total) / (max_total - mean_total)
        )

    z = z.clip(lower=0.0, upper=1.0)

    grades = LINEAR_MIN_GRADE + (
        (LINEAR_MAX_GRADE - LINEAR_MIN_GRADE) * z
    )

    return _round_and_clip_linear_grades(grades)


def _linear_grade_to_pass_fail(grades: pd.Series) -> pd.Series:
    grades = pd.to_numeric(grades, errors="coerce")

    pass_grade = LINEAR_MIN_GRADE + (
        (LINEAR_MAX_GRADE - LINEAR_MIN_GRADE) * LINEAR_PASS_THRESHOLD_NORM
    )

    return (grades >= pass_grade).astype("Int64")


# =========================================================
# BOLOGNA SCALE
# =========================================================

def _label_for_rank_position(position_1_based: int, cutoffs: list[int]) -> str:
    for label, cutoff in zip(DISTRIBUTION_PASSING_LABELS, cutoffs):
        if position_1_based <= cutoff:
            return label

    return DISTRIBUTION_PASSING_LABELS[-1]


def _assign_bologna_labels_from_normalized(
    normalized_scores: pd.Series,
    test_size: int,
) -> pd.Series:
    """
    Assign Bologna labels per exam instance.

    Rules:
    - F if below absolute pass threshold.
    - Passing students are ranked by total points.
    - Target distribution: A/B/C/D/E = configured percentages.
    - Tie handling: identical point totals always receive the same category.

    The tie category is chosen by the first rank of the tied group. If a tie
    crosses a Bologna boundary, the whole group receives the better category.
    This avoids assigning different Bologna grades to students with identical points.

    This function is variable for any test_size:
        pass_threshold_abs = test_size * BOLOGNA_PASS_THRESHOLD_NORM
    """
    scores = pd.to_numeric(normalized_scores, errors="coerce")

    if scores.empty:
        return pd.Series(dtype="object", index=normalized_scores.index)

    pass_threshold_abs = float(test_size) * float(DISTRIBUTION_PASS_THRESHOLD_NORM)
    absolute_points = scores * float(test_size)

    passed_mask = absolute_points >= pass_threshold_abs
    result = pd.Series(DISTRIBUTION_FAIL_LABEL, index=scores.index, dtype="object")

    passed = absolute_points[passed_mask].dropna()

    if passed.empty:
        return result

    n_passed = len(passed)
    cumulative = np.cumsum(DISTRIBUTION_PASSING_DISTRIBUTION)
    cutoffs = [int(np.ceil(n_passed * x)) for x in cumulative]
    cutoffs[-1] = n_passed

    passed_df = (
        pd.DataFrame(
            {
                "idx": passed.index,
                "points": passed.values,
            }
        )
        .sort_values(["points", "idx"], ascending=[False, True])
        .reset_index(drop=True)
    )

    current_rank_start = 1

    for points_value, group in passed_df.groupby("points", sort=False):
        group_size = len(group)
        rank_start = current_rank_start
        rank_end = current_rank_start + group_size - 1

        label = _label_for_rank_position(rank_start, cutoffs)

        result.loc[group["idx"].tolist()] = label

        current_rank_start = rank_end + 1

    return result


def _bologna_labels_to_ordinals(labels: pd.Series | list[str]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(DISTRIBUTION_ORDERED_LABELS)}
    return np.array([mapping[x] for x in labels], dtype=int)


# =========================================================
# SCALE EXPORTS FOR SANITY CHECKS
# =========================================================

def _student_scale_export_df(
    exam_df: pd.DataFrame,
    score_col: str,
    test_size: int,
) -> pd.DataFrame:
    """
    Builds a per-student sanity-check dataframe for one score column.

    This does not change the evaluation logic. It only exports the same
    student-level totals and derived grade-scale values into txt files.
    """
    subset = exam_df[
        [
            STUDENT_ID_COL,
            QUESTION_ID_COL,
            ANSWER_ID_COL,
            score_col,
        ]
    ].copy()

    subset[score_col] = pd.to_numeric(subset[score_col], errors="coerce")

    duplicate_mask = subset.duplicated(
        subset=[STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            subset.loc[
                duplicate_mask,
                [STUDENT_ID_COL, QUESTION_ID_COL, ANSWER_ID_COL],
            ]
            .sort_values([STUDENT_ID_COL, QUESTION_ID_COL])
            .head(50)
        )

        raise ValueError(
            "Doppelte (member_id, question_id)-Paare in einer virtuellen Prüfung. "
            "Das würde Scale-Exports verfälschen.\n"
            f"Beispiele:\n{duplicates.to_string(index=False)}"
        )

    grouped = (
        subset.groupby(STUDENT_ID_COL)
        .agg(
            n_rows=(QUESTION_ID_COL, "size"),
            n_questions=(QUESTION_ID_COL, "nunique"),
            valid_scores=(score_col, lambda s: int(s.notna().sum())),
            total_score=(score_col, "sum"),
        )
        .reset_index()
    )

    complete_mask = (
        (grouped["n_rows"] == int(test_size))
        & (grouped["n_questions"] == int(test_size))
        & (grouped["valid_scores"] == int(test_size))
    )

    grouped = grouped[complete_mask].copy()

    if grouped.empty:
        return grouped

    grouped["normalized_score"] = grouped["total_score"] / float(test_size)

    grouped["linear_grade_abs"] = _normalized_to_linear_grade_absolute(
        grouped["normalized_score"]
    )

    grouped["linear_pass_abs"] = _normalized_to_pass_fail_absolute(
        grouped["normalized_score"]
    )

    grouped["bologna_label"] = _assign_bologna_labels_from_normalized(
        grouped["normalized_score"],
        test_size=int(test_size),
    )

    grouped["bologna_pass"] = (
        grouped["bologna_label"] != DISTRIBUTION_FAIL_LABEL
    ).astype("Int64")

    grouped = grouped.sort_values(
        by=["total_score", STUDENT_ID_COL],
        ascending=[False, True],
    ).reset_index(drop=True)

    grouped.insert(0, "rank_by_total_score", grouped.index + 1)

    return grouped


def _write_scale_export_txt(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        output_path.write_text(
            "No complete students available for this test/model/scale.\n",
            encoding="utf-8",
        )
        return

    output_path.write_text(
        df.to_string(index=False),
        encoding="utf-8",
    )


def _write_scale_exports_for_single_exam(
    exam_df: pd.DataFrame,
    test_id: Any,
    test_size: int,
) -> None:
    """
    Writes sanity-check TXT files into:

    test_x/
      linear/<test_size>/
        linear_df_human.txt
        linear_df_<model>.txt

      bologna/<test_size>/
        bologna_df_human.txt
        bologna_df_<model>.txt

    The files are only exports. They do not affect metric computation.
    """
    test_number = _normalize_test_number_for_path(test_id)
    test_size_int = int(test_size)

    human_df = _student_scale_export_df(
        exam_df=exam_df,
        score_col=HUMAN_GRADE_COL,
        test_size=test_size_int,
    )

    human_linear = (
        human_df[
            [
                "rank_by_total_score",
                STUDENT_ID_COL,
                "n_rows",
                "n_questions",
                "valid_scores",
                "total_score",
                "normalized_score",
                "linear_grade_abs",
                "linear_pass_abs",
            ]
        ].copy()
        if not human_df.empty
        else human_df
    )

    human_bologna = (
        human_df[
            [
                "rank_by_total_score",
                STUDENT_ID_COL,
                "n_rows",
                "n_questions",
                "valid_scores",
                "total_score",
                "normalized_score",
                "bologna_label",
                "bologna_pass",
            ]
        ].copy()
        if not human_df.empty
        else human_df
    )

    _write_scale_export_txt(
        human_linear,
        _linear_dataframe_path(
            test_number=test_number,
            test_size=test_size_int,
            h_or_m="human",
        ),
    )

    _write_scale_export_txt(
        human_bologna,
        _bologna_dataframe_path(
            test_number=test_number,
            test_size=test_size_int,
            h_or_m="human",
        ),
    )

    for model_col in MODEL_COLUMNS:
        model_df = _student_scale_export_df(
            exam_df=exam_df,
            score_col=model_col,
            test_size=test_size_int,
        )

        model_linear = (
            model_df[
                [
                    "rank_by_total_score",
                    STUDENT_ID_COL,
                    "n_rows",
                    "n_questions",
                    "valid_scores",
                    "total_score",
                    "normalized_score",
                    "linear_grade_abs",
                    "linear_pass_abs",
                ]
            ].copy()
            if not model_df.empty
            else model_df
        )

        model_bologna = (
            model_df[
                [
                    "rank_by_total_score",
                    STUDENT_ID_COL,
                    "n_rows",
                    "n_questions",
                    "valid_scores",
                    "total_score",
                    "normalized_score",
                    "bologna_label",
                    "bologna_pass",
                ]
            ].copy()
            if not model_df.empty
            else model_df
        )

        _write_scale_export_txt(
            model_linear,
            _linear_dataframe_path(
                test_number=test_number,
                test_size=test_size_int,
                h_or_m=model_col,
            ),
        )

        _write_scale_export_txt(
            model_bologna,
            _bologna_dataframe_path(
                test_number=test_number,
                test_size=test_size_int,
                h_or_m=model_col,
            ),
        )


def _write_all_scale_exports(df_env: pd.DataFrame) -> int:
    written_exam_count = 0

    grouped = df_env.groupby(
        [TEST_ID_COL, TEST_SIZE_COL],
        sort=True,
    )

    iterator = grouped
    if SHOW_PROGRESS:
        iterator = tqdm(
            grouped,
            total=grouped.ngroups,
            desc="Writing scale sanity exports",
            unit="exam",
        )

    for (test_id, test_size), exam_df in iterator:
        _write_scale_exports_for_single_exam(
            exam_df=exam_df.copy(),
            test_id=test_id,
            test_size=int(test_size),
        )
        written_exam_count += 1

    return written_exam_count


# =========================================================
# ITEM-LEVEL BASE FROM ORIGINAL HELD-OUT INPUT
# =========================================================

def _build_item_df_from_original_input() -> pd.DataFrame:
    """
    Builds item-level evaluation data directly from INPUT_PARQUET.

    This is intentionally independent of dataframe_env.parquet because item-level
    metrics must be computed on the held-out responses, not on the synthetic
    virtual-exam environment.
    """
    input_path = _input_original_parquet()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Originales INPUT_PARQUET nicht gefunden: {input_path.resolve()}"
        )

    df = pd.read_parquet(input_path)
    _validate_original_item_df(df)

    keep_cols = list(
        dict.fromkeys(
            [
                ANSWER_ID_COL,
                QUESTION_ID_COL,
                STUDENT_ID_COL,
                INPUT_HUMAN_GRADE_COL,
                HUMAN_ONE_GOLD,
            ]
            + EVALUATION_MODEL_COLUMNS
        )
    )

    item_df = df[keep_cols].copy()

    item_df[ANSWER_ID_COL] = _normalize_string_series(item_df[ANSWER_ID_COL])
    item_df[QUESTION_ID_COL] = _normalize_string_series(item_df[QUESTION_ID_COL])
    item_df[STUDENT_ID_COL] = _normalize_string_series(item_df[STUDENT_ID_COL])

    item_df = item_df[
        (item_df[ANSWER_ID_COL] != "")
        & (item_df[QUESTION_ID_COL] != "")
        & (item_df[STUDENT_ID_COL] != "")
    ].copy()

    _assert_no_duplicate_student_question_pairs(
        item_df,
        context="Original INPUT_PARQUET",
    )

    item_df = item_df.rename(columns={INPUT_HUMAN_GRADE_COL: HUMAN_GRADE_COL})

    item_df = item_df.drop_duplicates(subset=[ANSWER_ID_COL]).copy()

    if item_df[ANSWER_ID_COL].duplicated().any():
        raise ValueError("Item-Level-Dataframe enthält doppelte answer_id.")

    return item_df


# =========================================================
# ITEM-LEVEL EVALUATION
# =========================================================

def _evaluate_item_level(df_items: pd.DataFrame, model_col: str) -> dict[str, float]:
    reference_col = _reference_col_for_model(model_col)

    subset = df_items[[reference_col, model_col]].copy()

    subset[reference_col] = pd.to_numeric(
        subset[reference_col],
        errors="coerce",
    )
    subset[model_col] = pd.to_numeric(
        subset[model_col],
        errors="coerce",
    )

    total_items = len(subset)
    missing_predictions = int(subset[model_col].isna().sum())

    subset = subset.dropna()

    y_true = subset[reference_col].to_numpy()
    y_pred = subset[model_col].to_numpy()

    return {
        "item_source_n": int(total_items),
        "item_n": int(len(subset)),
        "item_missing_predictions": missing_predictions,
        "item_mae": _mae_safe(y_true, y_pred),
        "item_mse": _mse_safe(y_true, y_pred),
        "item_rmse": _rmse_safe(y_true, y_pred),
        "item_tau_b": _kendall_tau_b_safe(y_true, y_pred),
        "item_qwk": _qwk_safe(y_true, y_pred),
    }


def _precompute_item_metrics(df_items: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    iterator = EVALUATION_MODEL_COLUMNS
    if SHOW_PROGRESS:
        iterator = tqdm(EVALUATION_MODEL_COLUMNS, desc="Item-level metrics", unit="model")

    for model_col in iterator:
        result[model_col] = _evaluate_item_level(df_items, model_col)

    return result


# =========================================================
# EXAM-LEVEL EVALUATION
# =========================================================

def _student_totals_for_exam(
    exam_df: pd.DataFrame,
    model_col: str,
    test_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Builds student totals for one exam.

    Missing prediction handling:
    A student is retained only if they have exactly test_size valid human scores
    and exactly test_size valid model predictions.
    """
    reference_col = _reference_col_for_model(model_col)

    exam_subset = exam_df[
        [STUDENT_ID_COL, QUESTION_ID_COL, ANSWER_ID_COL, reference_col, model_col]
    ].copy()

    exam_subset[reference_col] = pd.to_numeric(
        exam_subset[reference_col],
        errors="coerce",
    )
    exam_subset[model_col] = pd.to_numeric(
        exam_subset[model_col],
        errors="coerce",
    )

    duplicate_mask = exam_subset.duplicated(
        subset=[STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            exam_subset.loc[
                duplicate_mask,
                [STUDENT_ID_COL, QUESTION_ID_COL, ANSWER_ID_COL],
            ]
            .sort_values([STUDENT_ID_COL, QUESTION_ID_COL])
            .head(50)
        )

        raise ValueError(
            "Doppelte (member_id, question_id)-Paare in einer virtuellen Prüfung. "
            "Das würde Exam-Totals verfälschen.\n"
            f"Beispiele:\n{duplicates.to_string(index=False)}"
        )

    grouped = (
        exam_subset.groupby(STUDENT_ID_COL)
        .agg(
            n_rows=(QUESTION_ID_COL, "size"),
            n_questions=(QUESTION_ID_COL, "nunique"),
            human_valid=(reference_col, lambda s: int(s.notna().sum())),
            pred_valid=(model_col, lambda s: int(s.notna().sum())),
            gold_total=(reference_col, "sum"),
            pred_total=(model_col, "sum"),
        )
        .reset_index()
    )

    complete_mask = (
        (grouped["n_rows"] == int(test_size))
        & (grouped["n_questions"] == int(test_size))
        & (grouped["human_valid"] == int(test_size))
        & (grouped["pred_valid"] == int(test_size))
    )

    stats = {
        "students_raw": int(len(grouped)),
        "students_complete": int(complete_mask.sum()),
        "students_dropped_incomplete": int((~complete_mask).sum()),
        "students_missing_human": int((grouped["human_valid"] < int(test_size)).sum()),
        "students_missing_prediction": int((grouped["pred_valid"] < int(test_size)).sum()),
    }

    student_totals = grouped[complete_mask].copy()

    return student_totals, stats


def _evaluate_single_exam_for_model(
    exam_df: pd.DataFrame,
    model_col: str,
    test_id: Any,
    test_size: int,
) -> dict[str, Any] | None:
    test_size_int = int(test_size)

    student_totals, completion_stats = _student_totals_for_exam(
        exam_df=exam_df,
        model_col=model_col,
        test_size=test_size_int,
    )

    if student_totals.empty:
        return {
            "model_col": model_col,
            "test_id": test_id,
            "test_size": test_size_int,
            "n_students": 0,
            **completion_stats,

            "el_tau_b": np.nan,

            "el_acc_linear_abs": np.nan,
            "el_qwk_linear_abs": np.nan,
            "el_pass_acc_linear_abs": np.nan,
            "el_pass_qwk_linear_abs": np.nan,

            "el_acc_linear_mean": np.nan,
            "el_qwk_linear_mean": np.nan,
            "el_pass_acc_linear_mean": np.nan,
            "el_pass_qwk_linear_mean": np.nan,

            "el_acc_bologna": np.nan,
            "el_qwk_bologna": np.nan,
        }

    student_totals["gold_norm"] = (
        student_totals["gold_total"] / float(test_size_int)
    )
    student_totals["pred_norm"] = (
        student_totals["pred_total"] / float(test_size_int)
    )

    el_tau = _kendall_tau_b_safe(
        student_totals["gold_total"].to_numpy(),
        student_totals["pred_total"].to_numpy(),
    )

    # -------------------------------------------------
    # Absolute Linear Scale
    # -------------------------------------------------
    student_totals["gold_linear_abs"] = _normalized_to_linear_grade_absolute(
        student_totals["gold_norm"]
    )
    student_totals["pred_linear_abs"] = _normalized_to_linear_grade_absolute(
        student_totals["pred_norm"]
    )

    el_acc_linear_abs = _accuracy_safe(
        student_totals["gold_linear_abs"].to_numpy(),
        student_totals["pred_linear_abs"].to_numpy(),
    )

    el_qwk_linear_abs = _qwk_safe(
        student_totals["gold_linear_abs"].to_numpy(),
        student_totals["pred_linear_abs"].to_numpy(),
    )

    student_totals["gold_pass_abs"] = _normalized_to_pass_fail_absolute(
        student_totals["gold_norm"]
    )
    student_totals["pred_pass_abs"] = _normalized_to_pass_fail_absolute(
        student_totals["pred_norm"]
    )

    el_pass_acc_linear_abs = _accuracy_safe(
        student_totals["gold_pass_abs"].to_numpy(),
        student_totals["pred_pass_abs"].to_numpy(),
    )

    el_pass_qwk_linear_abs = _qwk_safe(
        student_totals["gold_pass_abs"].to_numpy(),
        student_totals["pred_pass_abs"].to_numpy(),
    )

    # -------------------------------------------------
    # Mean-Centered Linear Scale
    # -------------------------------------------------
    student_totals["gold_linear_mean"] = _totals_to_linear_grade_mean_centered(
        totals=student_totals["gold_total"],
        max_total=float(test_size_int),
    )
    student_totals["pred_linear_mean"] = _totals_to_linear_grade_mean_centered(
        totals=student_totals["pred_total"],
        max_total=float(test_size_int),
    )

    el_acc_linear_mean = _accuracy_safe(
        student_totals["gold_linear_mean"].to_numpy(),
        student_totals["pred_linear_mean"].to_numpy(),
    )

    el_qwk_linear_mean = _qwk_safe(
        student_totals["gold_linear_mean"].to_numpy(),
        student_totals["pred_linear_mean"].to_numpy(),
    )

    student_totals["gold_pass_mean"] = _linear_grade_to_pass_fail(
        student_totals["gold_linear_mean"]
    )
    student_totals["pred_pass_mean"] = _linear_grade_to_pass_fail(
        student_totals["pred_linear_mean"]
    )

    el_pass_acc_linear_mean = _accuracy_safe(
        student_totals["gold_pass_mean"].to_numpy(),
        student_totals["pred_pass_mean"].to_numpy(),
    )

    el_pass_qwk_linear_mean = _qwk_safe(
        student_totals["gold_pass_mean"].to_numpy(),
        student_totals["pred_pass_mean"].to_numpy(),
    )

    # -------------------------------------------------
    # Bologna Scale
    # -------------------------------------------------
    student_totals["gold_bologna"] = _assign_bologna_labels_from_normalized(
        student_totals["gold_norm"],
        test_size=test_size_int,
    )
    student_totals["pred_bologna"] = _assign_bologna_labels_from_normalized(
        student_totals["pred_norm"],
        test_size=test_size_int,
    )

    el_acc_bologna = _accuracy_safe(
        student_totals["gold_bologna"].to_numpy(),
        student_totals["pred_bologna"].to_numpy(),
    )

    el_qwk_bologna = _qwk_safe(
        _bologna_labels_to_ordinals(student_totals["gold_bologna"]),
        _bologna_labels_to_ordinals(student_totals["pred_bologna"]),
    )

    return {
        "model_col": model_col,
        "test_id": test_id,
        "test_size": test_size_int,
        "n_students": int(len(student_totals)),
        **completion_stats,

        "el_tau_b": el_tau,

        "el_acc_linear_abs": el_acc_linear_abs,
        "el_qwk_linear_abs": el_qwk_linear_abs,
        "el_pass_acc_linear_abs": el_pass_acc_linear_abs,
        "el_pass_qwk_linear_abs": el_pass_qwk_linear_abs,

        "el_acc_linear_mean": el_acc_linear_mean,
        "el_qwk_linear_mean": el_qwk_linear_mean,
        "el_pass_acc_linear_mean": el_pass_acc_linear_mean,
        "el_pass_qwk_linear_mean": el_pass_qwk_linear_mean,

        "el_acc_bologna": el_acc_bologna,
        "el_qwk_bologna": el_qwk_bologna,
    }


def _evaluate_single_exam_all_models(task: tuple[Any, int, pd.DataFrame]) -> list[dict[str, Any]]:
    test_id, test_size, exam_df = task

    rows: list[dict[str, Any]] = []

    for model_col in EVALUATION_MODEL_COLUMNS:
        row = _evaluate_single_exam_for_model(
            exam_df=exam_df,
            model_col=model_col,
            test_id=test_id,
            test_size=int(test_size),
        )

        if row is not None:
            rows.append(row)

    return rows


def _build_exam_tasks(df_env: pd.DataFrame) -> list[tuple[Any, int, pd.DataFrame]]:
    tasks: list[tuple[Any, int, pd.DataFrame]] = []

    for (test_id, test_size), exam_df in df_env.groupby(
        [TEST_ID_COL, TEST_SIZE_COL],
        sort=True,
    ):
        tasks.append((test_id, int(test_size), exam_df.copy()))

    return tasks


def _precompute_exam_results(df_env: pd.DataFrame) -> pd.DataFrame:
    tasks = _build_exam_tasks(df_env)

    if not tasks:
        return pd.DataFrame()

    all_rows: list[dict[str, Any]] = []

    workers = max(1, int(EVAL_WORKERS))

    if workers == 1:
        iterator = tasks
        if SHOW_PROGRESS:
            iterator = tqdm(tasks, desc="Exam-level metrics", unit="exam")

        for task in iterator:
            all_rows.extend(_evaluate_single_exam_all_models(task))

    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_evaluate_single_exam_all_models, task)
                for task in tasks
            ]

            iterator = as_completed(futures)
            if SHOW_PROGRESS:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"Exam-level metrics ({workers} workers)",
                    unit="exam",
                )

            for future in iterator:
                all_rows.extend(future.result())

    if not all_rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(all_rows)

    result_df[TEST_SIZE_COL] = pd.to_numeric(
        result_df[TEST_SIZE_COL],
        errors="coerce",
    ).astype("Int64")

    # Keep the output deterministic even when ProcessPoolExecutor finishes
    # virtual exam tasks in a different order across runs.
    result_df = result_df.sort_values(
        by=[TEST_SIZE_COL, TEST_ID_COL, "model_col"],
        kind="mergesort",
    ).reset_index(drop=True)

    return result_df


def _build_exam_metrics_wide_by_test(exam_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact env-level dataframe with one row per virtual exam.

    The source table is long, one row per (test_id, test_size, model_col). For
    quick plot loading this pivots those metrics into wide columns while keeping
    the dataframe at test granularity rather than repeating metrics on every
    answer row in dataframe_env.parquet.
    """
    if exam_results_df.empty:
        return pd.DataFrame(columns=[TEST_ID_COL, TEST_SIZE_COL])

    value_cols = EXAM_LEVEL_COUNT_COLUMNS + EXAM_LEVEL_METRIC_COLUMNS
    required = [TEST_ID_COL, TEST_SIZE_COL, "model_col", *value_cols]
    missing = [col for col in required if col not in exam_results_df.columns]
    if missing:
        raise ValueError(
            f"Exam-level result columns missing for wide env export: {missing}"
        )

    df = exam_results_df[required].copy()
    df["model_token"] = df["model_col"].map(_safe_file_token)

    wide = df.pivot_table(
        index=[TEST_ID_COL, TEST_SIZE_COL],
        columns="model_token",
        values=value_cols,
        aggfunc="first",
    )

    wide.columns = [
        f"{model_token}__{metric_col}"
        for metric_col, model_token in wide.columns.to_flat_index()
    ]

    wide = wide.reset_index()
    wide[TEST_SIZE_COL] = pd.to_numeric(
        wide[TEST_SIZE_COL],
        errors="coerce",
    ).astype("Int64")

    return wide.sort_values([TEST_SIZE_COL, TEST_ID_COL]).reset_index(drop=True)


def _aggregate_exam_results_for_plots(
    exam_results_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Average exam-level metrics after they were computed per virtual exam run.

    This is the table plot scripts should normally consume:
    first compute metrics per (test_id, test_size, model), then average over the
    N virtual runs. That avoids each figure reimplementing the same aggregation.
    """
    if exam_results_df.empty:
        return pd.DataFrame(columns=group_cols)

    value_cols = EXAM_LEVEL_COUNT_COLUMNS + EXAM_LEVEL_METRIC_COLUMNS
    required = [*group_cols, TEST_ID_COL, *value_cols]
    missing = [col for col in required if col not in exam_results_df.columns]
    if missing:
        raise ValueError(
            f"Exam-level result columns missing for summary export: {missing}"
        )

    df = exam_results_df[required].copy()

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby(group_cols, dropna=False)

    summary = grouped.agg(
        exam_instances=(TEST_ID_COL, "count"),
        n_runs=(TEST_ID_COL, "nunique"),
    ).reset_index()

    for col in EXAM_LEVEL_COUNT_COLUMNS:
        stats = grouped[col].agg(["mean", "std", "sum"]).reset_index()
        stats = stats.rename(
            columns={
                "mean": f"{col}_mean",
                "std": f"{col}_std",
                "sum": f"{col}_sum",
            }
        )
        summary = summary.merge(stats, on=group_cols, how="left")

    for col in EXAM_LEVEL_METRIC_COLUMNS:
        stats = grouped[col].agg(["mean", "std", "count"]).reset_index()
        stats = stats.rename(
            columns={
                "mean": f"{col}_mean",
                "std": f"{col}_std",
                "count": f"{col}_n",
            }
        )
        missing_counts = grouped[col].apply(lambda s: int(s.isna().sum())).reset_index()
        missing_counts = missing_counts.rename(columns={col: f"{col}_missing"})

        summary = summary.merge(stats, on=group_cols, how="left")
        summary = summary.merge(missing_counts, on=group_cols, how="left")

    return summary.sort_values(group_cols).reset_index(drop=True)


def _write_exam_metric_dataframes(
    exam_results_df: pd.DataFrame,
) -> dict[str, Path]:
    _env_metrics_dir().mkdir(parents=True, exist_ok=True)
    _dataframe_dir().mkdir(parents=True, exist_ok=True)

    exam_results_path = _exam_results_path()
    summary_by_test_size_path = _exam_summary_by_test_size_path()
    summary_all_path = _exam_summary_all_path()
    env_exam_metrics_wide_path = _env_exam_metrics_wide_path()

    exam_results_df.to_parquet(exam_results_path, index=False)

    summary_by_test_size = _aggregate_exam_results_for_plots(
        exam_results_df=exam_results_df,
        group_cols=["model_col", TEST_SIZE_COL],
    )
    summary_by_test_size.to_parquet(summary_by_test_size_path, index=False)

    summary_all = _aggregate_exam_results_for_plots(
        exam_results_df=exam_results_df,
        group_cols=["model_col"],
    )
    summary_all.to_parquet(summary_all_path, index=False)

    env_exam_metrics_wide = _build_exam_metrics_wide_by_test(exam_results_df)
    env_exam_metrics_wide.to_parquet(env_exam_metrics_wide_path, index=False)

    return {
        "exam_results": exam_results_path,
        "summary_by_test_size": summary_by_test_size_path,
        "summary_all": summary_all_path,
        "env_exam_metrics_wide": env_exam_metrics_wide_path,
    }


def _empty_exam_metrics() -> dict[str, float]:
    return {
        "exam_instances": 0,
        "exam_students_total": 0,
        "exam_students_raw_total": 0,
        "exam_students_dropped_incomplete_total": 0,
        "exam_students_missing_human_total": 0,
        "exam_students_missing_prediction_total": 0,

        "el_tau_b_mean": np.nan,
        "el_tau_b_std": np.nan,

        "el_acc_linear_abs_mean": np.nan,
        "el_acc_linear_abs_std": np.nan,
        "el_qwk_linear_abs_mean": np.nan,
        "el_qwk_linear_abs_std": np.nan,
        "el_pass_acc_linear_abs_mean": np.nan,
        "el_pass_acc_linear_abs_std": np.nan,
        "el_pass_qwk_linear_abs_mean": np.nan,
        "el_pass_qwk_linear_abs_std": np.nan,

        "el_acc_linear_mean_mean": np.nan,
        "el_acc_linear_mean_std": np.nan,
        "el_qwk_linear_mean_mean": np.nan,
        "el_qwk_linear_mean_std": np.nan,
        "el_pass_acc_linear_mean_mean": np.nan,
        "el_pass_acc_linear_mean_std": np.nan,
        "el_pass_qwk_linear_mean_mean": np.nan,
        "el_pass_qwk_linear_mean_std": np.nan,

        "el_acc_bologna_mean": np.nan,
        "el_acc_bologna_std": np.nan,
        "el_qwk_bologna_mean": np.nan,
        "el_qwk_bologna_std": np.nan,
    }


def _aggregate_exam_metrics(exam_results_df: pd.DataFrame, model_col: str) -> dict[str, float]:
    if exam_results_df.empty:
        return _empty_exam_metrics()

    df = exam_results_df[exam_results_df["model_col"] == model_col].copy()

    if df.empty:
        return _empty_exam_metrics()

    return {
        "exam_instances": int(len(df)),
        "exam_students_total": int(pd.to_numeric(df["n_students"], errors="coerce").fillna(0).sum()),
        "exam_students_raw_total": int(pd.to_numeric(df["students_raw"], errors="coerce").fillna(0).sum()),
        "exam_students_dropped_incomplete_total": int(pd.to_numeric(df["students_dropped_incomplete"], errors="coerce").fillna(0).sum()),
        "exam_students_missing_human_total": int(pd.to_numeric(df["students_missing_human"], errors="coerce").fillna(0).sum()),
        "exam_students_missing_prediction_total": int(pd.to_numeric(df["students_missing_prediction"], errors="coerce").fillna(0).sum()),

        "el_tau_b_mean": _mean_safe(df["el_tau_b"]),
        "el_tau_b_std": _std_safe(df["el_tau_b"]),

        "el_acc_linear_abs_mean": _mean_safe(df["el_acc_linear_abs"]),
        "el_acc_linear_abs_std": _std_safe(df["el_acc_linear_abs"]),
        "el_qwk_linear_abs_mean": _mean_safe(df["el_qwk_linear_abs"]),
        "el_qwk_linear_abs_std": _std_safe(df["el_qwk_linear_abs"]),
        "el_pass_acc_linear_abs_mean": _mean_safe(df["el_pass_acc_linear_abs"]),
        "el_pass_acc_linear_abs_std": _std_safe(df["el_pass_acc_linear_abs"]),
        "el_pass_qwk_linear_abs_mean": _mean_safe(df["el_pass_qwk_linear_abs"]),
        "el_pass_qwk_linear_abs_std": _std_safe(df["el_pass_qwk_linear_abs"]),

        "el_acc_linear_mean_mean": _mean_safe(df["el_acc_linear_mean"]),
        "el_acc_linear_mean_std": _std_safe(df["el_acc_linear_mean"]),
        "el_qwk_linear_mean_mean": _mean_safe(df["el_qwk_linear_mean"]),
        "el_qwk_linear_mean_std": _std_safe(df["el_qwk_linear_mean"]),
        "el_pass_acc_linear_mean_mean": _mean_safe(df["el_pass_acc_linear_mean"]),
        "el_pass_acc_linear_mean_std": _std_safe(df["el_pass_acc_linear_mean"]),
        "el_pass_qwk_linear_mean_mean": _mean_safe(df["el_pass_qwk_linear_mean"]),
        "el_pass_qwk_linear_mean_std": _std_safe(df["el_pass_qwk_linear_mean"]),

        "el_acc_bologna_mean": _mean_safe(df["el_acc_bologna"]),
        "el_acc_bologna_std": _std_safe(df["el_acc_bologna"]),
        "el_qwk_bologna_mean": _mean_safe(df["el_qwk_bologna"]),
        "el_qwk_bologna_std": _std_safe(df["el_qwk_bologna"]),
    }


# =========================================================
# REPORTING HELPERS
# =========================================================

def _format_metric(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.{digits}f}"


def _format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{_format_metric(mean, digits)} ± {_format_metric(std, digits)}"


def _filter_exam_results_for_scope(
    exam_results_df: pd.DataFrame,
    df_env_scope: pd.DataFrame,
) -> pd.DataFrame:
    if exam_results_df.empty or df_env_scope.empty:
        return exam_results_df.iloc[0:0].copy()

    scope_keys = (
        df_env_scope[[TEST_ID_COL, TEST_SIZE_COL]]
        .drop_duplicates()
        .copy()
    )

    scope_keys[TEST_SIZE_COL] = pd.to_numeric(
        scope_keys[TEST_SIZE_COL],
        errors="coerce",
    ).astype("Int64")

    filtered = exam_results_df.merge(
        scope_keys,
        on=[TEST_ID_COL, TEST_SIZE_COL],
        how="inner",
    )

    return filtered


def _build_report_section(
    df_env_scope: pd.DataFrame,
    df_items: pd.DataFrame,
    exam_results_scope_df: pd.DataFrame,
    item_metrics_by_model: dict[str, dict[str, float]],
    title: str,
) -> list[str]:
    lines: list[str] = []

    exam_instances = (
        df_env_scope[[TEST_ID_COL, TEST_SIZE_COL]]
        .drop_duplicates()
        .shape[0]
    )

    test_sizes_present = sorted(
        pd.to_numeric(df_env_scope[TEST_SIZE_COL], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    test_ids_present = sorted(
        df_env_scope[TEST_ID_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)
    lines.append(f"Rows in env scope: {len(df_env_scope)}")
    lines.append(f"Item-level source rows: {len(df_items)}")
    lines.append("Item-level source: original held-out INPUT_PARQUET, not dataframe_env.parquet")
    lines.append(f"Exam instances: {exam_instances}")
    lines.append(f"Test IDs present: {test_ids_present}")
    lines.append(f"Test sizes present: {test_sizes_present}")
    lines.append(f"Eval workers: {EVAL_WORKERS}")
    lines.append("")

    for model_col in EVALUATION_MODEL_COLUMNS:
        item_metrics = item_metrics_by_model[model_col]
        exam_metrics = _aggregate_exam_metrics(exam_results_scope_df, model_col)

        lines.append("-" * 100)
        lines.append(f"MODEL: {model_col}")
        lines.append("-" * 100)

        lines.append(f"Item-level source N:           {item_metrics['item_source_n']}")
        lines.append(f"Item-level valid N:            {item_metrics['item_n']}")
        lines.append(f"Item-level missing preds:      {item_metrics['item_missing_predictions']}")
        lines.append(f"Item-level MAE:                {_format_metric(item_metrics['item_mae'])}")
        lines.append(f"Item-level MSE:                {_format_metric(item_metrics['item_mse'])}")
        lines.append(f"Item-level RMSE:               {_format_metric(item_metrics['item_rmse'])}")
        lines.append(f"Item-level tau_b:              {_format_metric(item_metrics['item_tau_b'])}")
        lines.append(f"Item-level QWK:                {_format_metric(item_metrics['item_qwk'])}")
        lines.append("")

        lines.append(f"Exam instances:                {exam_metrics['exam_instances']}")
        lines.append(f"Exam students raw total:       {exam_metrics['exam_students_raw_total']}")
        lines.append(f"Exam students valid total:     {exam_metrics['exam_students_total']}")
        lines.append(f"Students dropped incomplete:   {exam_metrics['exam_students_dropped_incomplete_total']}")
        lines.append(f"Students missing human:        {exam_metrics['exam_students_missing_human_total']}")
        lines.append(f"Students missing predictions:  {exam_metrics['exam_students_missing_prediction_total']}")
        lines.append(
            "EL-tau_b:                      "
            + _format_mean_std(
                exam_metrics["el_tau_b_mean"],
                exam_metrics["el_tau_b_std"],
            )
        )
        lines.append("")

        lines.append("[Absolute Linear Scale]")
        lines.append(
            "EL-Acc Linear Abs:             "
            + _format_mean_std(
                exam_metrics["el_acc_linear_abs_mean"],
                exam_metrics["el_acc_linear_abs_std"],
            )
        )
        lines.append(
            "EL-QWK Linear Abs:             "
            + _format_mean_std(
                exam_metrics["el_qwk_linear_abs_mean"],
                exam_metrics["el_qwk_linear_abs_std"],
            )
        )
        lines.append(
            "EL-PassAcc Linear Abs:         "
            + _format_mean_std(
                exam_metrics["el_pass_acc_linear_abs_mean"],
                exam_metrics["el_pass_acc_linear_abs_std"],
            )
        )
        lines.append(
            "EL-PassQWK Linear Abs:         "
            + _format_mean_std(
                exam_metrics["el_pass_qwk_linear_abs_mean"],
                exam_metrics["el_pass_qwk_linear_abs_std"],
            )
        )
        lines.append("")

        lines.append("[Mean-Centered Linear Scale]")
        lines.append(
            "EL-Acc Linear Mean:            "
            + _format_mean_std(
                exam_metrics["el_acc_linear_mean_mean"],
                exam_metrics["el_acc_linear_mean_std"],
            )
        )
        lines.append(
            "EL-QWK Linear Mean:            "
            + _format_mean_std(
                exam_metrics["el_qwk_linear_mean_mean"],
                exam_metrics["el_qwk_linear_mean_std"],
            )
        )
        lines.append(
            "EL-PassAcc Linear Mean:        "
            + _format_mean_std(
                exam_metrics["el_pass_acc_linear_mean_mean"],
                exam_metrics["el_pass_acc_linear_mean_std"],
            )
        )
        lines.append(
            "EL-PassQWK Linear Mean:        "
            + _format_mean_std(
                exam_metrics["el_pass_qwk_linear_mean_mean"],
                exam_metrics["el_pass_qwk_linear_mean_std"],
            )
        )
        lines.append("")

        lines.append("[Bologna Scale]")
        lines.append(
            "EL-Acc Bologna:                "
            + _format_mean_std(
                exam_metrics["el_acc_bologna_mean"],
                exam_metrics["el_acc_bologna_std"],
            )
        )
        lines.append(
            "EL-QWK Bologna:                "
            + _format_mean_std(
                exam_metrics["el_qwk_bologna_mean"],
                exam_metrics["el_qwk_bologna_std"],
            )
        )
        lines.append("")
        lines.append("")

    return lines


def _build_report_for_scope(
    df_env_scope: pd.DataFrame,
    df_items: pd.DataFrame,
    exam_results_df: pd.DataFrame,
    item_metrics_by_model: dict[str, dict[str, float]],
    report_name: str,
) -> str:
    lines: list[str] = []

    test_sizes_present = sorted(
        pd.to_numeric(df_env_scope[TEST_SIZE_COL], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    for test_size in test_sizes_present:
        df_size = df_env_scope[df_env_scope[TEST_SIZE_COL] == test_size].copy()
        exam_results_size_df = _filter_exam_results_for_scope(
            exam_results_df=exam_results_df,
            df_env_scope=df_size,
        )

        if not df_size.empty:
            lines.extend(
                _build_report_section(
                    df_env_scope=df_size,
                    df_items=df_items,
                    exam_results_scope_df=exam_results_size_df,
                    item_metrics_by_model=item_metrics_by_model,
                    title=f"{report_name} - TEST SIZE {test_size}",
                )
            )

    exam_results_scope_df = _filter_exam_results_for_scope(
        exam_results_df=exam_results_df,
        df_env_scope=df_env_scope,
    )

    lines.extend(
        _build_report_section(
            df_env_scope=df_env_scope,
            df_items=df_items,
            exam_results_scope_df=exam_results_scope_df,
            item_metrics_by_model=item_metrics_by_model,
            title=f"{report_name} - ALL TEST SIZES",
        )
    )

    return "\n".join(lines)


def _build_global_report(
    df_env: pd.DataFrame,
    df_items: pd.DataFrame,
    exam_results_df: pd.DataFrame,
    item_metrics_by_model: dict[str, dict[str, float]],
) -> str:
    return _build_report_for_scope(
        df_env_scope=df_env,
        df_items=df_items,
        exam_results_df=exam_results_df,
        item_metrics_by_model=item_metrics_by_model,
        report_name="VEX EVALUATION REPORT",
    )


def _normalize_test_number_for_path(test_id: Any) -> int | str:
    """
    Supports:
    - 1 -> 1
    - "1" -> 1
    - "test_1" -> 1
    """
    if pd.isna(test_id):
        raise ValueError("test_id darf nicht NaN sein.")

    value = str(test_id).strip()

    if value.startswith("test_"):
        suffix = value.removeprefix("test_")
        if suffix.isdigit():
            return int(suffix)

    try:
        as_float = float(value)
        if as_float.is_integer():
            return int(as_float)
    except Exception:
        pass

    return value


def _sort_test_ids(test_ids: list[Any]) -> list[Any]:
    def sort_key(value: Any) -> tuple[int, Any]:
        normalized = _normalize_test_number_for_path(value)

        if isinstance(normalized, int):
            return (0, normalized)

        try:
            as_float = float(normalized)
            return (0, as_float)
        except Exception:
            return (1, str(normalized))

    return sorted(test_ids, key=sort_key)


# =========================================================
# PER-TEST REPORT WRITING
# =========================================================

def _write_single_test_reports(
    df_env: pd.DataFrame,
    df_items: pd.DataFrame,
    exam_results_df: pd.DataFrame,
    item_metrics_by_model: dict[str, dict[str, float]],
) -> list[Path]:
    written_files: list[Path] = []

    test_ids = df_env[TEST_ID_COL].dropna().unique().tolist()
    test_ids = _sort_test_ids(test_ids)

    iterator = test_ids
    if SHOW_PROGRESS:
        iterator = tqdm(test_ids, desc="Writing per-test reports", unit="test")

    for test_id in iterator:
        df_test = df_env[df_env[TEST_ID_COL] == test_id].copy()

        if df_test.empty:
            continue

        test_number = _normalize_test_number_for_path(test_id)

        exam_results_test_df = _filter_exam_results_for_scope(
            exam_results_df=exam_results_df,
            df_env_scope=df_test,
        )

        report_text = _build_report_for_scope(
            df_env_scope=df_test,
            df_items=df_items,
            exam_results_df=exam_results_test_df,
            item_metrics_by_model=item_metrics_by_model,
            report_name=f"VEX EVALUATION REPORT - TEST {test_number}",
        )

        output_path = _run_metrics_file(test_number)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")

        written_files.append(output_path)

    return written_files


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    input_path = _input_env_parquet()

    if not input_path.exists():
        raise FileNotFoundError(
            f"dataframe_env.parquet nicht gefunden: {input_path.resolve()}"
        )

    print("=" * 100)
    print("VEX EVALUATION")
    print("=" * 100)
    print(f"Env input:     {input_path.resolve()}")
    print(f"Original data: {_input_original_parquet().resolve()}")
    print(f"Eval workers:  {EVAL_WORKERS}")
    print("")

    df_env = pd.read_parquet(input_path)
    _validate_env_df(df_env)

    _assert_no_duplicate_exam_student_question_pairs(df_env)

    df_items = _build_item_df_from_original_input()

    _env_metrics_dir().mkdir(parents=True, exist_ok=True)

    print("Precomputing item-level metrics...")
    item_metrics_by_model = _precompute_item_metrics(df_items)

    print("")
    print("Precomputing exam-level metrics...")
    exam_results_df = _precompute_exam_results(df_env)

    exam_metric_paths = _write_exam_metric_dataframes(exam_results_df)

    print("Precomputed exam metric dataframes saved to:")
    for label, path in exam_metric_paths.items():
        print(f"  {label}: {path.resolve()}")

    if WRITE_SCALE_SANITY_EXPORTS:
        print("")
        print("Writing linear/Bologna sanity-check exports...")
        scale_export_count = _write_all_scale_exports(df_env)
        print(f"Scale sanity-check exports written for {scale_export_count} exam instances.")

    # -----------------------------------------------------
    # 1) Global report over the complete synthetic exam env
    # -----------------------------------------------------
    global_report_text = _build_global_report(
        df_env=df_env,
        df_items=df_items,
        exam_results_df=exam_results_df,
        item_metrics_by_model=item_metrics_by_model,
    )

    global_report_path = _output_report_path()
    global_report_path.write_text(global_report_text, encoding="utf-8")

    # -----------------------------------------------------
    # 2) Per-test reports
    # -----------------------------------------------------
    written_test_reports = _write_single_test_reports(
        df_env=df_env,
        df_items=df_items,
        exam_results_df=exam_results_df,
        item_metrics_by_model=item_metrics_by_model,
    )

    print("")
    print(global_report_text)
    print(f"\nGlobaler Report gespeichert unter: {global_report_path.resolve()}")

    print("")
    print("=" * 100)
    print("PER-TEST METRICS")
    print("=" * 100)
    print(f"Anzahl geschriebener Test-Reports: {len(written_test_reports)}")

    if written_test_reports:
        print(f"Erster Test-Report: {written_test_reports[0].resolve()}")
        print(f"Letzter Test-Report: {written_test_reports[-1].resolve()}")

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
