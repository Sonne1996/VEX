#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import queue
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
VEX_METRIC_DIR = PROJECT_ROOT / "vex_metric"

if not VEX_METRIC_DIR.exists():
    raise FileNotFoundError(
        f"Could not find vex_metric directory at: {VEX_METRIC_DIR.resolve()}"
    )

sys.path.insert(0, str(VEX_METRIC_DIR))

import vex_config as cfg


# =========================================================
# CONFIG
# =========================================================

TEST_SIZE = 20

# For debugging: 100 or 1_000
# For final reporting: 5_000 or 10_000
N_PERMUTATIONS = 10_000

RANDOM_SEED = 4242

OUTPUT_DIR = SCRIPT_PATH.parent / f"statistical_significance_q{TEST_SIZE}"

# Metric used only for displaying the main ranking table.
# Significance tests select their own best model per metric.
DISPLAY_RANKING_METRIC = "el_acc_linear_abs"

SIGNIFICANCE_TASKS = [
    {
        "test_family": "mcnemar",
        "metric_col": "el_acc_linear_abs",
        "scale": "linear_abs",
        "description": "Exact McNemar test for EL-Acc on absolute linear grades.",
    },
    {
        "test_family": "mcnemar",
        "metric_col": "el_acc_bologna",
        "scale": "bologna",
        "description": "Exact McNemar test for EL-Acc on Bologna labels.",
    },
    {
        "test_family": "permutation",
        "metric_col": "el_qwk_linear_abs",
        "scale": "linear_abs",
        "description": "Paired permutation test for EL-QWK on absolute linear grades.",
    },
    {
        "test_family": "permutation",
        "metric_col": "el_qwk_bologna",
        "scale": "bologna",
        "description": "Paired permutation test for EL-QWK on Bologna labels.",
    },
]

ALPHA = 0.05

# None = use available CPUs, capped by number of comparisons.
# For your machine, 6-8 may be more stable than 11 because every worker does heavy NumPy work.
N_JOBS: int | None = 8

# Worker progress update frequency.
PROGRESS_EVERY_PERMUTATIONS = 10

TEST_ID_COL = "test_id"
TEST_SIZE_COL = "test_size"
STUDENT_ID_COL = "member_id"
QUESTION_ID_COL = "question_id"
ANSWER_ID_COL = "answer_id"
HUMAN_GRADE_COL = "human_grade"

EXCLUDE_TFIDF = False

ScaleName = Literal["linear_abs", "bologna"]


# =========================================================
# MODEL HELPERS
# =========================================================

# Collects all model prediction columns that should be evaluated.
#
# The function starts from cfg.MODEL_COLUMNS and removes:
#   - optionally TF-IDF baselines if EXCLUDE_TFIDF=True
#
# The returned list defines exactly which model columns enter the ranking,
# exam-level label construction, and significance tests.
def get_eval_model_columns() -> list[str]:
    model_cols: list[str] = []

    for col in cfg.MODEL_COLUMNS:

        if EXCLUDE_TFIDF and col.startswith("pred_tfidf_"):
            continue

        model_cols.append(col)

    if not model_cols:
        raise ValueError("No model columns left after filtering.")

    return model_cols

# Converts a full dataframe model column name into a readable short name.
#
# Example:
#   "new_grade_openai/gpt-5.4" -> "openai/gpt-5.4"
#   "grade_deepseek/deepseek-v3.2" -> "deepseek/deepseek-v3.2"
#
# This is only used for display/reporting. It does not change the actual
# dataframe column names used internally.
def short_model_name(model_col: str) -> str:
    name = model_col

    for prefix in ["new_grade_", "grade_", "pred_"]:
        if name.startswith(prefix):
            name = name.removeprefix(prefix)

    return name

# Determines how many worker processes should be used for parallel testing.
#
# If N_JOBS is explicitly configured, the function uses at most N_JOBS workers,
# but never more workers than available model comparisons.
#
# If N_JOBS is None, it uses the available CPU count minus one, again capped by
# the number of comparisons.
def resolve_n_jobs(n_comparisons: int) -> int:
    if n_comparisons <= 0:
        return 1

    if N_JOBS is not None:
        return max(1, min(int(N_JOBS), int(n_comparisons)))

    try:
        import os

        cpu_count = os.cpu_count()
    except Exception:
        cpu_count = None

    if cpu_count is None:
        return min(4, int(n_comparisons))

    return max(1, min(int(cpu_count) - 1, int(n_comparisons)))


# =========================================================
# METRIC HELPERS
# =========================================================

# Computes exact classification accuracy.
#
# Formula:
#   accuracy = number of equal labels / number of labels
#
# In this script it is used for exact final-grade agreement after exam-level
# grades have already been rounded/converted.
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return float("nan")

    return float(np.mean(y_true == y_pred))

# Computes quadratic weighted kappa using sklearn as a safe reference version.
#
# QWK measures ordinal agreement between true and predicted labels while
# penalizing larger ordinal disagreements more strongly than smaller ones.
#
# Conceptually:
#   QWK = 1 - weighted_observed_disagreement / weighted_expected_disagreement
#
# The quadratic weight for class i vs. class j is:
#   w_ij = (i - j)^2 / (K - 1)^2
#
# where K is the number of ordered classes.
#
# Interpretation:
#   1.0  = perfect agreement
#   0.0  = agreement no better than chance
#   <0.0 = worse than chance
#
# This implementation is used outside the hot permutation loop because sklearn
# is correct and convenient, but too slow for thousands of permutations.
def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return float("nan")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))

    if len(labels) <= 1:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    mapping = {label: idx for idx, label in enumerate(labels)}

    y_true_enc = np.array([mapping[x] for x in y_true], dtype=int)
    y_pred_enc = np.array([mapping[x] for x in y_pred], dtype=int)

    return float(cohen_kappa_score(y_true_enc, y_pred_enc, weights="quadratic"))

# Fast integer implementation of quadratic weighted kappa.
#
# Inputs must already be integer-encoded labels:
#   y_true, y_pred in {0, ..., n_classes - 1}
#
# The function builds the observed confusion matrix O:
#   O_ij = number of samples with true class i and predicted class j
#
# It then computes the expected matrix E from the marginal distributions:
#   E_ij = (hist_true_i * hist_pred_j) / n
#
# With quadratic weights W, the QWK formula is:
#   QWK = 1 - sum(W * O) / sum(W * E)
#
# This avoids pandas/sklearn overhead and is used inside the permutation test,
# where QWK must be recomputed many times.
def qwk_fast_int(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    weights_matrix: np.ndarray,
) -> float:
    n = int(len(y_true))

    if n == 0 or n != len(y_pred):
        return float("nan")

    if np.all(y_true == y_true[0]) and np.all(y_pred == y_pred[0]):
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    flat_index = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)

    observed = np.bincount(
        flat_index,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)

    expected = np.outer(hist_true, hist_pred) / float(n)

    numerator = float(np.sum(weights_matrix * observed))
    denominator = float(np.sum(weights_matrix * expected))

    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0

    return float(1.0 - numerator / denominator)

# Creates the quadratic weight matrix used by QWK.
#
# For K ordered classes, the weight between class i and class j is:
#   w_ij = (i - j)^2 / (K - 1)^2
#
# This means:
#   - exact agreement has weight 0
#   - nearby disagreement has small penalty
#   - far disagreement has large penalty
#
# Example:
#   predicting grade 5 instead of 6 is punished less than predicting 2 instead
#   of 6.
def make_quadratic_weights(n_classes: int) -> np.ndarray:
    if n_classes <= 1:
        return np.zeros((n_classes, n_classes), dtype=float)

    idx = np.arange(n_classes, dtype=float)
    diff = idx[:, None] - idx[None, :]
    return (diff ** 2) / float((n_classes - 1) ** 2)

# Rounds linear grades to the configured grade step and clips them to the valid
# grade range.
#
# Example with LINEAR_ROUNDING_STEP = 0.25:
#   4.13 -> 4.25
#   4.12 -> 4.00
#
# After rounding, grades are clipped into:
#   [cfg.LINEAR_MIN_GRADE, cfg.LINEAR_MAX_GRADE]
#
# This ensures that all final linear grades are valid Swiss-style grades.
def round_and_clip_linear_grades(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    if cfg.LINEAR_ROUNDING_STEP and cfg.LINEAR_ROUNDING_STEP > 0:
        arr = np.round(arr / cfg.LINEAR_ROUNDING_STEP) * cfg.LINEAR_ROUNDING_STEP

    return np.clip(arr, cfg.LINEAR_MIN_GRADE, cfg.LINEAR_MAX_GRADE)

# Converts normalized scores into absolute linear grades.
#
# The normalized score is expected to be in [0, 1].
#
# Formula:
#   grade = min_grade + (max_grade - min_grade) * normalized_score
#
# With Swiss-style grades:
#   min_grade = 1.0
#   max_grade = 6.0
#
# So:
#   score 0.0 -> grade 1.0
#   score 0.6 -> grade 4.0
#   score 1.0 -> grade 6.0
#
# The resulting grade is then rounded and clipped.
def normalized_to_linear_grade_absolute(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)

    grades = cfg.LINEAR_MIN_GRADE + (
        (cfg.LINEAR_MAX_GRADE - cfg.LINEAR_MIN_GRADE) * arr
    )

    return round_and_clip_linear_grades(grades)

# Converts Bologna grade labels into ordinal integers.
#
# The order is defined by cfg.BOLOGNA_ORDERED_LABELS.
#
# Example:
#   ["F", "E", "D", "C", "B", "A"]
# becomes:
#   F -> 0
#   E -> 1
#   ...
#   A -> 5
#
# This is necessary because QWK requires ordered numeric classes.
def bologna_labels_to_ordinals(labels: pd.Series | np.ndarray) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(cfg.BOLOGNA_ORDERED_LABELS)}
    return np.array([mapping[x] for x in labels], dtype=int)

# Returns the Bologna passing label for a one-based rank position.
#
# The cutoffs define the upper rank boundary for each passing label.
#
# Example:
#   labels  = ["A", "B", "C", "D", "E"]
#   cutoffs = [2, 7, 13, 18, 20]
#
# Then:
#   rank 1-2   -> A
#   rank 3-7   -> B
#   rank 8-13  -> C
#   rank 14-18 -> D
#   rank 19-20 -> E
#
# Note: in the current script, Bologna assignment is handled directly in
# assign_bologna_labels_from_normalized(), including tie-aware handling.
def label_for_rank_position(position_1_based: int, cutoffs: list[int]) -> str:
    for label, cutoff in zip(cfg.BOLOGNA_PASSING_LABELS, cutoffs):
        if position_1_based <= cutoff:
            return label

    return cfg.BOLOGNA_PASSING_LABELS[-1]

# Assigns Bologna-style final labels to students within one virtual exam.
#
# Step 1:
#   Convert normalized exam scores into absolute points:
#     points = normalized_score * test_size
#
# Step 2:
#   Determine who passes:
#     passed if points >= test_size * cfg.BOLOGNA_PASS_THRESHOLD_NORM
#
# Step 3:
#   All failing students receive cfg.BOLOGNA_FAIL_LABEL.
#
# Step 4:
#   Passing students are ranked by points from highest to lowest.
#
# Step 5:
#   Passing labels are assigned according to the configured distribution:
#     cfg.BOLOGNA_PASSING_DISTRIBUTION
#
# Example:
#   [0.10, 0.25, 0.30, 0.25, 0.10]
# means roughly:
#   top 10% -> A
#   next 25% -> B
#   next 30% -> C
#   next 25% -> D
#   last 10% -> E
#
# Tie handling:
#   If students have the same number of points and the tie crosses a grade
#   boundary, the whole tie group receives the better grade touched by the
#   first rank of that tie group.
def assign_bologna_labels_from_normalized(
    normalized_scores: pd.Series,
    test_size: int,
) -> pd.Series:
    scores = pd.to_numeric(normalized_scores, errors="coerce")

    result = pd.Series(cfg.BOLOGNA_FAIL_LABEL, index=scores.index, dtype="object")

    absolute_points = scores * float(test_size)
    pass_threshold_abs = float(test_size) * float(cfg.BOLOGNA_PASS_THRESHOLD_NORM)

    passed_mask = absolute_points >= pass_threshold_abs
    passed = absolute_points[passed_mask].dropna()

    if passed.empty:
        return result

    n_passed = len(passed)

    cumulative = np.cumsum(cfg.BOLOGNA_PASSING_DISTRIBUTION)
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

    for _, group in passed_df.groupby("points", sort=False):
        group_size = len(group)
        rank_start = current_rank_start
        rank_end = current_rank_start + group_size - 1

        # Tie-aware boundary handling:
        # If a tie group crosses a grade boundary, the whole group receives
        # the better grade category touched by its first rank.
        #
        # Example:
        # A cutoff = 10
        # ranks 10 and 11 have identical points
        # -> both receive A, not A/B split.
        label = cfg.BOLOGNA_PASSING_LABELS[-1]

        for candidate_label, cutoff in zip(
            cfg.BOLOGNA_PASSING_LABELS,
            cutoffs,
        ):
            if rank_start <= cutoff:
                label = candidate_label
                break

        result.loc[group["idx"].tolist()] = label

        current_rank_start = rank_end + 1

    return result


# =========================================================
# DATA LOADING / VALIDATION
# =========================================================

# Resolves a configured path.
#
# If the path is absolute, it is returned unchanged.
# If the path is relative, it is interpreted relative to VEX_METRIC_DIR.
#
# This keeps config paths portable while still supporting absolute paths.
def cfg_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return VEX_METRIC_DIR / path

# Returns the input parquet path for the prepared dataframe environment.
#
# The path comes from cfg.OUTPUT_PARQUET and is resolved through cfg_path().
def input_env_path() -> Path:
    return cfg_path(cfg.OUTPUT_PARQUET)

# Validates that the dataframe environment contains all required columns.
#
# Required metadata columns:
#   - test_id
#   - test_size
#   - member_id
#   - question_id
#   - answer_id
#   - human_grade
#
# Also checks that every selected model column exists.
#
# Without these columns, exam-level aggregation and pairwise model comparison
# would be invalid.
def validate_env_df(df: pd.DataFrame, model_cols: list[str]) -> None:
    required = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        STUDENT_ID_COL,
        QUESTION_ID_COL,
        ANSWER_ID_COL,
        HUMAN_GRADE_COL,
    ]

    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns in dataframe_env.parquet: {missing_required}"
        )

    missing_models = [col for col in model_cols if col not in df.columns]
    if missing_models:
        raise ValueError(
            f"Missing model columns in dataframe_env.parquet: {missing_models}"
        )

# Checks that every virtual exam/student/question pair is unique.
#
# The uniqueness key is:
#   (test_id, test_size, member_id, question_id)
#
# This matters because exam-level scores are computed by summing one answer per
# question. If duplicates exist, a student's exam score would be inflated or
# otherwise corrupted.
def assert_no_duplicate_exam_student_question_pairs(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(
        subset=[TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if duplicate_mask.any():
        preview = (
            df.loc[
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
            "Duplicate (test_id, test_size, member_id, question_id) pairs found. "
            "Exam-level totals would be invalid.\n\n"
            f"{preview.to_string(index=False)}"
        )


# =========================================================
# EXAM-LEVEL LABEL CONSTRUCTION
# =========================================================

# Builds the exam-level label table used for ranking and significance testing.
#
# Input:
#   item-level dataframe with one row per answer/question/student/test/model
#
# For each model, the function:
#   1. Filters to the requested TEST_SIZE.
#   2. Groups by:
#        (test_id, test_size, member_id)
#   3. Keeps only complete virtual exams where:
#        n_rows == test_size
#        n_questions == test_size
#        all human grades are valid
#        all model predictions are valid
#   4. Computes total human and model scores:
#        gold_total = sum(human_grade)
#        pred_total = sum(model_prediction)
#   5. Converts totals to normalized scores:
#        gold_norm = gold_total / test_size
#        pred_norm = pred_total / test_size
#   6. Converts normalized scores to absolute linear grades.
#   7. Computes exact linear-grade correctness.
#   8. Assigns Bologna labels separately inside each virtual exam.
#   9. Converts Bologna labels to ordinal integers for QWK.
#  10. Computes exact Bologna-label correctness.
#
# Output:
#   one row per model, virtual exam, and student.
def build_exam_level_labels(
    df_env: pd.DataFrame,
    model_cols: list[str],
    test_size: int,
) -> pd.DataFrame:
    df = df_env[df_env[TEST_SIZE_COL].astype(int) == int(test_size)].copy()

    if df.empty:
        raise ValueError(f"No rows found for test_size={test_size}.")

    keep_cols = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        STUDENT_ID_COL,
        QUESTION_ID_COL,
        ANSWER_ID_COL,
        HUMAN_GRADE_COL,
    ] + model_cols

    df = df[keep_cols].copy()

    df[HUMAN_GRADE_COL] = pd.to_numeric(df[HUMAN_GRADE_COL], errors="coerce")

    for model_col in model_cols:
        df[model_col] = pd.to_numeric(df[model_col], errors="coerce")

    all_model_rows: list[pd.DataFrame] = []

    group_cols = [TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL]

    for model_col in tqdm(
        model_cols,
        desc="Building exam-level labels",
        unit="model",
    ):
        subset = df[
            [
                TEST_ID_COL,
                TEST_SIZE_COL,
                STUDENT_ID_COL,
                QUESTION_ID_COL,
                HUMAN_GRADE_COL,
                model_col,
            ]
        ].copy()

        grouped = (
            subset.groupby(group_cols)
            .agg(
                n_rows=(QUESTION_ID_COL, "size"),
                n_questions=(QUESTION_ID_COL, "nunique"),
                human_valid=(HUMAN_GRADE_COL, lambda s: int(s.notna().sum())),
                pred_valid=(model_col, lambda s: int(s.notna().sum())),
                gold_total=(HUMAN_GRADE_COL, "sum"),
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

        grouped = grouped[complete_mask].copy()

        if grouped.empty:
            print(f"WARNING: No complete exam-level rows for {model_col}")
            continue

        grouped["gold_norm"] = grouped["gold_total"] / float(test_size)
        grouped["pred_norm"] = grouped["pred_total"] / float(test_size)

        grouped["gold_linear_abs"] = normalized_to_linear_grade_absolute(
            grouped["gold_norm"].to_numpy()
        )
        grouped["pred_linear_abs"] = normalized_to_linear_grade_absolute(
            grouped["pred_norm"].to_numpy()
        )

        grouped["correct_linear_abs"] = (
            grouped["gold_linear_abs"].to_numpy()
            == grouped["pred_linear_abs"].to_numpy()
        ).astype(int)

        bologna_frames: list[pd.DataFrame] = []

        for _, exam_df in grouped.groupby(TEST_ID_COL, sort=False):
            exam_df = exam_df.copy()

            exam_df["gold_bologna"] = assign_bologna_labels_from_normalized(
                exam_df["gold_norm"],
                test_size=int(test_size),
            )

            exam_df["pred_bologna"] = assign_bologna_labels_from_normalized(
                exam_df["pred_norm"],
                test_size=int(test_size),
            )

            bologna_frames.append(exam_df)

        grouped = pd.concat(bologna_frames, ignore_index=True)

        grouped["gold_bologna_ord"] = bologna_labels_to_ordinals(
            grouped["gold_bologna"].to_numpy()
        )
        grouped["pred_bologna_ord"] = bologna_labels_to_ordinals(
            grouped["pred_bologna"].to_numpy()
        )

        grouped["correct_bologna"] = (
            grouped["gold_bologna"].to_numpy()
            == grouped["pred_bologna"].to_numpy()
        ).astype(int)

        grouped["model_col"] = model_col
        grouped["model"] = short_model_name(model_col)

        all_model_rows.append(grouped)

    if not all_model_rows:
        raise ValueError("No exam-level labels could be built.")

    return pd.concat(all_model_rows, ignore_index=True)


# =========================================================
# RANKING
# =========================================================

# Computes mean exam-level metrics for one model and one grading scale.
#
# For each virtual exam separately:
#   - EL-Acc is computed as exact final-grade agreement across students.
#   - EL-QWK is computed as ordinal agreement across students.
#
# Then the function averages the metric values over all virtual exams.
def compute_el_metric_mean_per_model(
    df_model: pd.DataFrame,
    scale: ScaleName,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []

    for _, exam_df in df_model.groupby(TEST_ID_COL, sort=False):
        if scale == "linear_abs":
            y_true = exam_df["gold_linear_abs"].to_numpy()
            y_pred = exam_df["pred_linear_abs"].to_numpy()
            correct = exam_df["correct_linear_abs"].to_numpy()
        elif scale == "bologna":
            y_true = exam_df["gold_bologna_ord"].to_numpy()
            y_pred = exam_df["pred_bologna_ord"].to_numpy()
            correct = exam_df["correct_bologna"].to_numpy()
        else:
            raise ValueError(f"Unknown scale: {scale}")

        rows.append(
            {
                "el_acc": float(np.mean(correct)),
                "el_qwk": qwk(y_true, y_pred),
            }
        )

    metric_df = pd.DataFrame(rows)

    return {
        f"el_acc_{scale}": float(metric_df["el_acc"].mean()),
        f"el_qwk_{scale}": float(metric_df["el_qwk"].mean()),
    }

# Computes the full model ranking table.
#
# For each model, this calculates:
#   - EL-Acc on absolute linear grades
#   - EL-QWK on absolute linear grades
#   - EL-Acc on Bologna labels
#   - EL-QWK on Bologna labels
#
# The ranking is sorted primarily by DISPLAY_RANKING_METRIC and then by the
# remaining exam-level metrics as tie-breakers.
def compute_model_ranking(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_col, df_model in labels.groupby("model_col", sort=False):
        linear_metrics = compute_el_metric_mean_per_model(df_model, "linear_abs")
        bologna_metrics = compute_el_metric_mean_per_model(df_model, "bologna")

        rows.append(
            {
                "model_col": model_col,
                "model": short_model_name(model_col),
                "n_exam_student_rows": int(len(df_model)),
                **linear_metrics,
                **bologna_metrics,
            }
        )

    ranking = pd.DataFrame(rows)

    required_metric_cols = [
        "el_acc_linear_abs",
        "el_qwk_linear_abs",
        "el_acc_bologna",
        "el_qwk_bologna",
    ]

    missing = [col for col in required_metric_cols if col not in ranking.columns]
    if missing:
        raise ValueError(f"Missing ranking metric columns: {missing}")

    if DISPLAY_RANKING_METRIC not in ranking.columns:
        raise ValueError(
            f"Unknown DISPLAY_RANKING_METRIC={DISPLAY_RANKING_METRIC}. "
            f"Available metric columns: {[c for c in ranking.columns if c.startswith('el_')]}"
        )

    ranking = ranking.sort_values(
        by=[
            DISPLAY_RANKING_METRIC,
            "el_qwk_linear_abs",
            "el_qwk_bologna",
            "el_acc_linear_abs",
            "el_acc_bologna",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return ranking


# Selects the best model for a specific metric.
#
# This is important because significance testing is metric-specific:
#   - best model for EL-Acc linear may differ from best model for EL-QWK linear
#   - best model for Bologna may differ from best model for linear
#
# The selected metric is used first.
# Other exam-level metrics are used only as deterministic tie-breakers.
def select_best_model_for_metric(
    ranking: pd.DataFrame,
    metric_col: str,
) -> str:
    if metric_col not in ranking.columns:
        raise ValueError(
            f"Cannot select best model. Metric column does not exist: {metric_col}"
        )

    tie_breakers = [
        metric_col,
        "el_acc_linear_abs",
        "el_qwk_linear_abs",
        "el_acc_bologna",
        "el_qwk_bologna",
    ]

    tie_breakers = list(dict.fromkeys(tie_breakers))

    sorted_ranking = ranking.sort_values(
        by=tie_breakers,
        ascending=[False] * len(tie_breakers),
    ).reset_index(drop=True)

    return str(sorted_ranking.iloc[0]["model_col"])


# Computes which model is best for each configured significance task.
#
# Example:
#   - McNemar on el_acc_linear_abs selects the best model by el_acc_linear_abs.
#   - Permutation on el_qwk_bologna selects the best model by el_qwk_bologna.
#
# This avoids wrongly comparing every metric against a single globally best
# model selected by only one ranking criterion.
def compute_best_models_by_significance_task(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for task in SIGNIFICANCE_TASKS:
        metric_col = str(task["metric_col"])
        test_family = str(task["test_family"])
        scale = str(task["scale"])

        best_model_col = select_best_model_for_metric(
            ranking=ranking,
            metric_col=metric_col,
        )

        best_row = ranking[ranking["model_col"] == best_model_col].iloc[0]

        rows.append(
            {
                "test_family": test_family,
                "scale": scale,
                "selection_metric": metric_col,
                "best_model_col": best_model_col,
                "best_model": short_model_name(best_model_col),
                "best_metric_value": float(best_row[metric_col]),
                "description": str(task["description"]),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# PAIRING
# =========================================================


# Creates a paired comparison dataframe for two models.
#
# The two models are joined on:
#   (test_id, test_size, member_id)
#
# This ensures that both models are compared on exactly the same virtual
# exam/student rows.
#
# The function also verifies that the gold labels match between both sides.
# If gold labels differ, the comparison is invalid and an error is raised.
def paired_model_frame(
    labels: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    key_cols = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        STUDENT_ID_COL,
    ]

    keep_cols = key_cols + [
        "gold_linear_abs",
        "pred_linear_abs",
        "correct_linear_abs",
        "gold_bologna",
        "pred_bologna",
        "gold_bologna_ord",
        "pred_bologna_ord",
        "correct_bologna",
    ]

    df_a = labels[labels["model_col"] == model_a][keep_cols].copy()
    df_b = labels[labels["model_col"] == model_b][keep_cols].copy()

    df_a = df_a.rename(
        columns={
            "pred_linear_abs": "pred_linear_abs_a",
            "correct_linear_abs": "correct_linear_abs_a",
            "pred_bologna": "pred_bologna_a",
            "pred_bologna_ord": "pred_bologna_ord_a",
            "correct_bologna": "correct_bologna_a",
        }
    )

    df_b = df_b.rename(
        columns={
            "gold_linear_abs": "gold_linear_abs_b",
            "pred_linear_abs": "pred_linear_abs_b",
            "correct_linear_abs": "correct_linear_abs_b",
            "gold_bologna": "gold_bologna_b",
            "pred_bologna": "pred_bologna_b",
            "gold_bologna_ord": "gold_bologna_ord_b",
            "pred_bologna_ord": "pred_bologna_ord_b",
            "correct_bologna": "correct_bologna_b",
        }
    )

    merged = df_a.merge(
        df_b,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(f"No paired rows for {model_a} vs {model_b}.")

    if np.any(
        merged["gold_linear_abs"].to_numpy()
        != merged["gold_linear_abs_b"].to_numpy()
    ):
        raise ValueError(f"Gold linear labels mismatch for {model_a} vs {model_b}.")

    if np.any(
        merged["gold_bologna"].to_numpy()
        != merged["gold_bologna_b"].to_numpy()
    ):
        raise ValueError(f"Gold Bologna labels mismatch for {model_a} vs {model_b}.")

    merged = merged.drop(
        columns=[
            "gold_linear_abs_b",
            "gold_bologna_b",
            "gold_bologna_ord_b",
        ]
    )

    return merged


# =========================================================
# MCNEMAR FOR EL-ACC
# =========================================================

# Computes the exact two-sided McNemar p-value.
#
# McNemar is used for paired binary outcomes.
#
# Here the binary outcome is:
#   model prediction exactly matches final grade: yes/no
#
# Discordant cells:
#   b = model A correct, model B wrong
#   c = model A wrong, model B correct
#
# Null hypothesis:
#   both models have the same probability of being correct on paired examples
#
# Under the null, among discordant pairs:
#   b ~ Binomial(b + c, 0.5)
#
# This function uses an exact two-sided binomial test.
def exact_mcnemar_p_value(b: int, c: int) -> float:
    n = int(b + c)

    if n == 0:
        return 1.0

    return float(
        binomtest(
            k=min(b, c),
            n=n,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


# Runs an exact McNemar test for EL-Acc between two paired models.
#
# For every paired virtual exam/student row, the function checks whether:
#   - model A got the final grade exactly right
#   - model B got the final grade exactly right
#
# It builds the 2x2 paired correctness table:
#
#                         Model B correct    Model B wrong
#   Model A correct       both_correct       a_correct_b_wrong
#   Model A wrong         a_wrong_b_correct  both_wrong
#
# Only the discordant cells matter for McNemar:
#   b = a_correct_b_wrong
#   c = a_wrong_b_correct
#
# A small p-value means the difference in exact final-grade correctness is
# unlikely under the null hypothesis of equal paired accuracy.
def mcnemar_compare_el_acc(
    merged: pd.DataFrame,
    model_a: str,
    model_b: str,
    scale: ScaleName,
) -> dict[str, Any]:
    correct_col_a = f"correct_{scale}_a"
    correct_col_b = f"correct_{scale}_b"

    a_correct = merged[correct_col_a].astype(bool).to_numpy()
    b_correct = merged[correct_col_b].astype(bool).to_numpy()

    both_correct = int(np.sum(a_correct & b_correct))
    a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
    a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    p_value = exact_mcnemar_p_value(
        b=a_correct_b_wrong,
        c=a_wrong_b_correct,
    )

    return {
        "test": f"mcnemar_exact_el_acc_{scale}",
        "scale": scale,
        "model_a": short_model_name(model_a),
        "model_b": short_model_name(model_b),
        "model_a_col": model_a,
        "model_b_col": model_b,
        "n_paired": int(len(merged)),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant": int(a_correct_b_wrong + a_wrong_b_correct),
        "p_value": p_value,
        "significant_0_05": bool(p_value < ALPHA),
    }


# =========================================================
# FAST PERMUTATION TEST FOR EL-QWK
# =========================================================

# Encodes absolute linear grades into integer classes for fast QWK.
#
# Example with:
#   LINEAR_MIN_GRADE = 1.0
#   LINEAR_ROUNDING_STEP = 0.25
#
# Encoding:
#   1.00 -> 0
#   1.25 -> 1
#   1.50 -> 2
#   ...
#   6.00 -> 20
#
# QWK needs ordered class indices, not floating-point grade values.
def encode_linear_grades_for_fast_qwk(values: np.ndarray) -> np.ndarray:
    """
    Encodes linear grades into stable integer classes.

    Example with absolut quarter grades:
        1.00 -> 0
        1.25 -> 1
        ...
        6.00 -> 20
    """
    values = np.asarray(values, dtype=float)

    step = float(cfg.LINEAR_ROUNDING_STEP)
    min_grade = float(cfg.LINEAR_MIN_GRADE)

    encoded = np.rint((values - min_grade) / step).astype(np.int64)

    return encoded


# Prepares all arrays needed for fast repeated EL-QWK computation.
#
# The merged paired model dataframe is sorted by:
#   (test_id, member_id)
#
# The function then creates:
#   - gold_int: integer-encoded gold labels
#   - pred_a_int: integer-encoded predictions of model A
#   - pred_b_int: integer-encoded predictions of model B
#   - exam_slices: start/end index ranges for each virtual exam
#   - n_classes: number of ordered classes
#   - weights_matrix: quadratic QWK weight matrix
#
# exam_slices are needed because EL-QWK is computed per virtual exam first and
# then averaged across exams.
def prepare_fast_qwk_arrays(
    merged: pd.DataFrame,
    scale: ScaleName,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], int, np.ndarray]:
    """
    Returns:
        gold_int
        pred_a_int
        pred_b_int
        exam_slices
        n_classes
        weights_matrix
    """
    sort_cols = [TEST_ID_COL, STUDENT_ID_COL]
    work = merged.sort_values(sort_cols).reset_index(drop=True)

    exam_ids = work[TEST_ID_COL].to_numpy()

    boundaries: list[tuple[int, int]] = []
    start = 0

    for i in range(1, len(work)):
        if exam_ids[i] != exam_ids[start]:
            boundaries.append((start, i))
            start = i

    boundaries.append((start, len(work)))

    if scale == "linear_abs":
        gold_int = encode_linear_grades_for_fast_qwk(
            work["gold_linear_abs"].to_numpy()
        )
        pred_a_int = encode_linear_grades_for_fast_qwk(
            work["pred_linear_abs_a"].to_numpy()
        )
        pred_b_int = encode_linear_grades_for_fast_qwk(
            work["pred_linear_abs_b"].to_numpy()
        )

        n_classes = int(
            max(
                gold_int.max(),
                pred_a_int.max(),
                pred_b_int.max(),
            )
            + 1
        )

    elif scale == "bologna":
        gold_int = work["gold_bologna_ord"].to_numpy(dtype=np.int64)
        pred_a_int = work["pred_bologna_ord_a"].to_numpy(dtype=np.int64)
        pred_b_int = work["pred_bologna_ord_b"].to_numpy(dtype=np.int64)

        n_classes = int(
            max(
                gold_int.max(),
                pred_a_int.max(),
                pred_b_int.max(),
            )
            + 1
        )

    else:
        raise ValueError(f"Unknown scale: {scale}")

    weights_matrix = make_quadratic_weights(n_classes)

    return gold_int, pred_a_int, pred_b_int, boundaries, n_classes, weights_matrix


# Computes mean exam-level QWK using the fast integer QWK implementation.
#
# For each virtual exam slice:
#   qwk_exam = QWK(gold labels of students, predicted labels of students)
#
# Then:
#   mean_EL_QWK = mean(qwk_exam over all virtual exams)
#
# This is the central EL-QWK definition used in the permutation test.
def mean_el_qwk_fast(
    gold_int: np.ndarray,
    pred_int: np.ndarray,
    exam_slices: list[tuple[int, int]],
    n_classes: int,
    weights_matrix: np.ndarray,
) -> float:
    values = np.empty(len(exam_slices), dtype=float)

    for idx, (start, end) in enumerate(exam_slices):
        values[idx] = qwk_fast_int(
            y_true=gold_int[start:end],
            y_pred=pred_int[start:end],
            n_classes=n_classes,
            weights_matrix=weights_matrix,
        )

    return float(np.nanmean(values))


# Runs a paired permutation test for the difference in mean EL-QWK.
#
# Observed statistic:
#   delta_observed = mean_EL_QWK(model A) - mean_EL_QWK(model B)
#
# Null hypothesis:
#   the two paired model predictions are exchangeable, meaning there is no
#   systematic difference between model A and model B.
#
# Permutation procedure:
#   For each paired virtual exam/student row:
#     with probability 0.5, swap the prediction of model A and model B.
#
#   After swapping:
#     recompute mean_EL_QWK(permuted A)
#     recompute mean_EL_QWK(permuted B)
#     store:
#       delta_permuted = mean_EL_QWK(permuted A) - mean_EL_QWK(permuted B)
#
# Two-sided p-value:
#   p = (count(|delta_permuted| >= |delta_observed|) + 1) / (n_permutations + 1)
#
# The +1 correction prevents p=0 and is standard for Monte Carlo permutation
# testing.
#
# A small p-value means the observed EL-QWK difference is unlikely under paired
# exchangeability.
def permutation_compare_el_qwk(
    merged: pd.DataFrame,
    model_a: str,
    model_b: str,
    scale: ScaleName,
    n_permutations: int,
    seed: int,
    progress_queue: Any | None = None,
    progress_key: str | None = None,
    progress_every: int = PROGRESS_EVERY_PERMUTATIONS,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    (
        gold_int,
        pred_a_int,
        pred_b_int,
        exam_slices,
        n_classes,
        weights_matrix,
    ) = prepare_fast_qwk_arrays(
        merged=merged,
        scale=scale,
    )

    qwk_a = mean_el_qwk_fast(
        gold_int=gold_int,
        pred_int=pred_a_int,
        exam_slices=exam_slices,
        n_classes=n_classes,
        weights_matrix=weights_matrix,
    )

    qwk_b = mean_el_qwk_fast(
        gold_int=gold_int,
        pred_int=pred_b_int,
        exam_slices=exam_slices,
        n_classes=n_classes,
        weights_matrix=weights_matrix,
    )

    observed_delta = qwk_a - qwk_b
    observed_abs_delta = abs(observed_delta)

    perm_deltas = np.empty(n_permutations, dtype=float)

    n_rows = len(gold_int)

    reported_since_last = 0

    for i in range(n_permutations):
        swap_mask = rng.random(n_rows) < 0.5

        perm_a = np.where(swap_mask, pred_b_int, pred_a_int)
        perm_b = np.where(swap_mask, pred_a_int, pred_b_int)

        perm_qwk_a = mean_el_qwk_fast(
            gold_int=gold_int,
            pred_int=perm_a,
            exam_slices=exam_slices,
            n_classes=n_classes,
            weights_matrix=weights_matrix,
        )

        perm_qwk_b = mean_el_qwk_fast(
            gold_int=gold_int,
            pred_int=perm_b,
            exam_slices=exam_slices,
            n_classes=n_classes,
            weights_matrix=weights_matrix,
        )

        perm_deltas[i] = perm_qwk_a - perm_qwk_b

        reported_since_last += 1

        if (
            progress_queue is not None
            and progress_key is not None
            and reported_since_last >= progress_every
        ):
            progress_queue.put(
                {
                    "type": "permutation_progress",
                    "key": progress_key,
                    "increment": int(reported_since_last),
                }
            )
            reported_since_last = 0

    if (
        progress_queue is not None
        and progress_key is not None
        and reported_since_last > 0
    ):
        progress_queue.put(
            {
                "type": "permutation_progress",
                "key": progress_key,
                "increment": int(reported_since_last),
            }
        )

    count_extreme = int(np.sum(np.abs(perm_deltas) >= observed_abs_delta))
    p_value = float((count_extreme + 1) / (n_permutations + 1))

    return {
        "test": f"paired_permutation_el_qwk_{scale}",
        "scale": scale,
        "model_a": short_model_name(model_a),
        "model_b": short_model_name(model_b),
        "model_a_col": model_a,
        "model_b_col": model_b,
        "n_paired": int(len(merged)),
        "n_virtual_exams": int(merged[TEST_ID_COL].nunique()),
        "qwk_a": float(qwk_a),
        "qwk_b": float(qwk_b),
        "delta_qwk_a_minus_b": float(observed_delta),
        "abs_delta_qwk": float(observed_abs_delta),
        "n_permutations": int(n_permutations),
        "count_extreme": count_extreme,
        "p_value": p_value,
        "significant_0_05": bool(p_value < ALPHA),
    }


# =========================================================
# SIGNIFICANCE TEST RUNNER
# =========================================================


# Runs one significance comparison job.
#
# Each job compares:
#   metric-specific best model vs. one other model
#
# Depending on test_family:
#   - "mcnemar"    -> exact McNemar test for EL-Acc
#   - "permutation" -> paired permutation test for EL-QWK
#
# The function is designed to be usable both sequentially and inside a process
# pool worker.
def run_one_significance_comparison_worker(
    args: tuple[pd.DataFrame, str, str, str, str, Any | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels, best_model_col, other_model_col, test_family, scale, progress_queue = args

    comparison_name = (
        f"{test_family} | {scale} | "
        f"{short_model_name(best_model_col)} vs {short_model_name(other_model_col)}"
    )

    if progress_queue is not None:
        progress_queue.put(
            {
                "type": "comparison_started",
                "comparison": comparison_name,
            }
        )

    merged = paired_model_frame(
        labels=labels,
        model_a=best_model_col,
        model_b=other_model_col,
    )

    mcnemar_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []

    if test_family == "mcnemar":
        mcnemar_rows.append(
            mcnemar_compare_el_acc(
                merged=merged,
                model_a=best_model_col,
                model_b=other_model_col,
                scale=scale,  # type: ignore[arg-type]
            )
        )

    elif test_family == "permutation":
        progress_key = f"{comparison_name}"

        permutation_rows.append(
            permutation_compare_el_qwk(
                merged=merged,
                model_a=best_model_col,
                model_b=other_model_col,
                scale=scale,  # type: ignore[arg-type]
                n_permutations=N_PERMUTATIONS,
                seed=RANDOM_SEED,
                progress_queue=progress_queue,
                progress_key=progress_key,
                progress_every=PROGRESS_EVERY_PERMUTATIONS,
            )
        )

    else:
        raise ValueError(f"Unknown test_family: {test_family}")

    if progress_queue is not None:
        progress_queue.put(
            {
                "type": "comparison_finished",
                "comparison": comparison_name,
            }
        )

    return mcnemar_rows, permutation_rows


# Runs all configured significance comparisons sequentially.
#
# For each significance task:
#   1. Select the best model for that task's metric.
#   2. Compare that best model against every other model.
#   3. Store McNemar or permutation results.
#
# This version is slower but easier to debug because everything runs in one
# process.
def run_significance_tests_sequential(
    labels: pd.DataFrame,
    ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mcnemar_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []

    jobs: list[tuple[pd.DataFrame, str, str, str, str, Any | None]] = []

    for task in SIGNIFICANCE_TASKS:
        test_family = str(task["test_family"])
        metric_col = str(task["metric_col"])
        scale = str(task["scale"])

        best_model_col = select_best_model_for_metric(
            ranking=ranking,
            metric_col=metric_col,
        )

        other_model_cols = [
            str(x)
            for x in ranking["model_col"].tolist()
            if str(x) != best_model_col
        ]

        for other_model_col in other_model_cols:
            jobs.append(
                (
                    labels,
                    best_model_col,
                    other_model_col,
                    test_family,
                    scale,
                    None,
                )
            )

    total_permutations = sum(
        1
        for job in jobs
        if job[3] == "permutation"
    ) * int(N_PERMUTATIONS)

    with tqdm(
        total=total_permutations,
        desc="Permutation progress",
        unit="perm",
    ) as perm_bar:
        class LocalProgressQueue:
            def put(self, item: dict[str, Any]) -> None:
                if item.get("type") == "permutation_progress":
                    perm_bar.update(int(item.get("increment", 0)))
                elif item.get("type") == "comparison_started":
                    tqdm.write(f"Started:  {item.get('comparison')}")
                elif item.get("type") == "comparison_finished":
                    tqdm.write(f"Finished: {item.get('comparison')}")

        progress_queue = LocalProgressQueue()

        jobs_with_progress = [
            (
                labels,
                best_model_col,
                other_model_col,
                test_family,
                scale,
                progress_queue,
            )
            for (
                labels,
                best_model_col,
                other_model_col,
                test_family,
                scale,
                _,
            ) in jobs
        ]

        for job in tqdm(
            jobs_with_progress,
            desc="Running significance comparisons",
            unit="comparison",
        ):
            one_mcnemar, one_permutation = run_one_significance_comparison_worker(job)

            mcnemar_rows.extend(one_mcnemar)
            permutation_rows.extend(one_permutation)

    return pd.DataFrame(mcnemar_rows), pd.DataFrame(permutation_rows)


# Runs all configured significance comparisons in parallel.
#
# The function creates one worker job per:
#   metric-specific best model vs. other model comparison
#
# It uses ProcessPoolExecutor because permutation tests are CPU-heavy.
#
# Progress is reported through a multiprocessing queue:
#   - completed comparisons
#   - permutation steps
#   - started/finished comparison messages
#
# If only one worker is resolved, the function falls back to the sequential
# implementation.
def run_significance_tests_parallel(
    labels: pd.DataFrame,
    ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs_base: list[tuple[str, str, str, str]] = []

    for task in SIGNIFICANCE_TASKS:
        test_family = str(task["test_family"])
        metric_col = str(task["metric_col"])
        scale = str(task["scale"])

        best_model_col = select_best_model_for_metric(
            ranking=ranking,
            metric_col=metric_col,
        )

        other_model_cols = [
            str(x)
            for x in ranking["model_col"].tolist()
            if str(x) != best_model_col
        ]

        for other_model_col in other_model_cols:
            jobs_base.append(
                (
                    best_model_col,
                    other_model_col,
                    test_family,
                    scale,
                )
            )

    n_jobs = resolve_n_jobs(len(jobs_base))

    print(
        f"Running significance tests in parallel with {n_jobs} worker process"
        f"{'' if n_jobs == 1 else 'es'}."
    )

    if n_jobs == 1:
        return run_significance_tests_sequential(
            labels=labels,
            ranking=ranking,
        )

    total_permutations = sum(
        1
        for _, _, test_family, _ in jobs_base
        if test_family == "permutation"
    ) * int(N_PERMUTATIONS)

    mcnemar_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []

    with Manager() as manager:
        progress_queue = manager.Queue()

        worker_args = [
            (
                labels,
                best_model_col,
                other_model_col,
                test_family,
                scale,
                progress_queue,
            )
            for best_model_col, other_model_col, test_family, scale in jobs_base
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(run_one_significance_comparison_worker, arg)
                for arg in worker_args
            ]

            future_to_comparison = {
                future: (
                    f"{test_family} | {scale} | "
                    f"{short_model_name(best_model_col)} vs {short_model_name(other_model_col)}"
                )
                for future, (
                    _labels,
                    best_model_col,
                    other_model_col,
                    test_family,
                    scale,
                    _progress_queue,
                ) in zip(futures, worker_args)
            }

            completed_futures: set[Any] = set()

            with tqdm(
                total=len(futures),
                desc="Completed comparisons",
                unit="comparison",
                position=0,
            ) as comparison_bar, tqdm(
                total=total_permutations,
                desc="Permutation progress",
                unit="perm",
                position=1,
            ) as perm_bar:
                while len(completed_futures) < len(futures):
                    while True:
                        try:
                            item = progress_queue.get_nowait()
                        except queue.Empty:
                            break
                        except Exception:
                            break

                        item_type = item.get("type")

                        if item_type == "permutation_progress":
                            perm_bar.update(int(item.get("increment", 0)))
                        elif item_type == "comparison_started":
                            tqdm.write(f"Started:  {item.get('comparison')}")
                        elif item_type == "comparison_finished":
                            tqdm.write(f"Finished: {item.get('comparison')}")

                    for future in futures:
                        if future in completed_futures:
                            continue

                        if future.done():
                            completed_futures.add(future)
                            comparison_bar.update(1)

                            comparison_label = future_to_comparison[future]

                            try:
                                one_mcnemar, one_permutation = future.result()
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Significance worker failed for comparison "
                                    f"{comparison_label}: {repr(exc)}"
                                ) from exc

                            mcnemar_rows.extend(one_mcnemar)
                            permutation_rows.extend(one_permutation)

                    if len(completed_futures) < len(futures):
                        time.sleep(0.2)

                while True:
                    try:
                        item = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception:
                        break

                    if item.get("type") == "permutation_progress":
                        perm_bar.update(int(item.get("increment", 0)))
                    elif item.get("type") == "comparison_started":
                        tqdm.write(f"Started:  {item.get('comparison')}")
                    elif item.get("type") == "comparison_finished":
                        tqdm.write(f"Finished: {item.get('comparison')}")

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    permutation_df = pd.DataFrame(permutation_rows)

    if not mcnemar_df.empty:
        mcnemar_df = mcnemar_df.sort_values(
            by=["test", "scale", "model_a", "model_b"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    if not permutation_df.empty:
        permutation_df = permutation_df.sort_values(
            by=["test", "scale", "model_a", "model_b"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    return mcnemar_df, permutation_df


# Main dispatcher for significance testing.
#
# Currently this delegates to the parallel implementation.
# Keeping this wrapper makes it easy to switch between parallel and sequential
# execution later without changing main().
def run_significance_tests(
    labels: pd.DataFrame,
    ranking: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_significance_tests_parallel(
        labels=labels,
        ranking=ranking,
    )


# =========================================================
# SANITY CHECKS
# =========================================================

# Creates one standardized sanity-check result row.
#
# Each sanity check returns:
#   - check name
#   - passed flag
#   - severity level
#   - details string
def make_sanity_row(
    check_name: str,
    passed: bool,
    severity: str,
    details: str,
) -> dict[str, Any]:
    return {
        "check": check_name,
        "passed": bool(passed),
        "severity": severity,
        "details": details,
    }


# Runs sanity checks for the full significance pipeline.
#
# The checks verify, among other things:
#   - requested test size exists
#   - generated labels are not empty
#   - every model has the expected rows
#   - no duplicate exam/student rows exist
#   - gold labels are identical across models
#   - exam-level completeness is preserved
#   - correctness flags can be recomputed exactly
#   - Bologna labels are valid
#   - ranking can be recomputed exactly
#   - metric-specific best-vs-other pairings are valid
#
# Critical failures stop the script after the report is written.
# Warning failures are reported but do not necessarily invalidate the output.
def run_sanity_checks(
    df_env: pd.DataFrame,
    labels: pd.DataFrame,
    ranking: pd.DataFrame,
    model_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    best_model_cols = {
        str(task["metric_col"]): select_best_model_for_metric(
            ranking=ranking,
            metric_col=str(task["metric_col"]),
        )
        for task in SIGNIFICANCE_TASKS
    }

    def check_significance_selection_metrics_exist() -> dict[str, Any]:
        required = [str(task["metric_col"]) for task in SIGNIFICANCE_TASKS]
        missing = [col for col in required if col not in ranking.columns]

        return make_sanity_row(
            check_name="significance_selection_metrics_exist",
            passed=len(missing) == 0,
            severity="critical",
            details=f"required={required}; missing={missing}",
        )

    def check_env_test_size_exists() -> dict[str, Any]:
        n_rows = int((df_env[TEST_SIZE_COL].astype(int) == int(TEST_SIZE)).sum())
        return make_sanity_row(
            check_name="env_contains_requested_test_size",
            passed=n_rows > 0,
            severity="critical",
            details=f"Rows with test_size={TEST_SIZE}: {n_rows}",
        )

    def check_labels_not_empty() -> dict[str, Any]:
        return make_sanity_row(
            check_name="labels_not_empty",
            passed=not labels.empty,
            severity="critical",
            details=f"labels_rows={len(labels)}",
        )

    def check_one_row_per_model_exam_student() -> dict[str, Any]:
        key_cols = ["model_col", TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL]
        duplicate_count = int(labels.duplicated(subset=key_cols).sum())
        return make_sanity_row(
            check_name="one_row_per_model_exam_student",
            passed=duplicate_count == 0,
            severity="critical",
            details=f"duplicate_count={duplicate_count}",
        )

    def check_all_model_keys_identical() -> dict[str, Any]:
        key_cols = [TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL]

        key_sets: dict[str, set[tuple[Any, ...]]] = {}

        for model_col, df_model in labels.groupby("model_col"):
            key_sets[str(model_col)] = set(map(tuple, df_model[key_cols].to_numpy()))

        if not key_sets:
            return make_sanity_row(
                check_name="all_model_exam_student_keys_identical",
                passed=False,
                severity="critical",
                details="No model key sets found.",
            )

        reference_model = next(iter(key_sets))
        reference_keys = key_sets[reference_model]

        mismatches: list[str] = []

        for model_col, keys in key_sets.items():
            missing = len(reference_keys - keys)
            extra = len(keys - reference_keys)

            if missing != 0 or extra != 0:
                mismatches.append(
                    f"{short_model_name(model_col)}: "
                    f"missing_vs_ref={missing}, extra_vs_ref={extra}"
                )

        return make_sanity_row(
            check_name="all_model_exam_student_keys_identical",
            passed=len(mismatches) == 0,
            severity="critical",
            details=(
                "All models share identical exam-student keys."
                if not mismatches
                else "; ".join(mismatches)
            ),
        )

    def check_gold_labels_identical_across_models() -> dict[str, Any]:
        key_cols = [TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL]

        grouped = (
            labels.groupby(key_cols)
            .agg(
                gold_linear_unique=("gold_linear_abs", "nunique"),
                gold_bologna_unique=("gold_bologna", "nunique"),
                gold_bologna_ord_unique=("gold_bologna_ord", "nunique"),
            )
            .reset_index()
        )

        bad = grouped[
            (grouped["gold_linear_unique"] != 1)
            | (grouped["gold_bologna_unique"] != 1)
            | (grouped["gold_bologna_ord_unique"] != 1)
        ]

        return make_sanity_row(
            check_name="gold_labels_identical_across_models",
            passed=bad.empty,
            severity="critical",
            details=f"bad_exam_student_keys={len(bad)}",
        )

    def check_complete_exam_counts() -> dict[str, Any]:
        bad = labels[
            (labels["n_rows"].astype(int) != int(TEST_SIZE))
            | (labels["n_questions"].astype(int) != int(TEST_SIZE))
            | (labels["human_valid"].astype(int) != int(TEST_SIZE))
            | (labels["pred_valid"].astype(int) != int(TEST_SIZE))
        ]

        return make_sanity_row(
            check_name="complete_exam_counts",
            passed=bad.empty,
            severity="critical",
            details=f"incomplete_exam_student_rows={len(bad)}",
        )

    def check_no_nan_in_critical_label_columns() -> dict[str, Any]:
        critical_cols = [
            "gold_total",
            "pred_total",
            "gold_norm",
            "pred_norm",
            "gold_linear_abs",
            "pred_linear_abs",
            "correct_linear_abs",
            "gold_bologna",
            "pred_bologna",
            "gold_bologna_ord",
            "pred_bologna_ord",
            "correct_bologna",
        ]

        nan_counts = labels[critical_cols].isna().sum()
        bad = nan_counts[nan_counts > 0]

        return make_sanity_row(
            check_name="no_nan_in_critical_label_columns",
            passed=bad.empty,
            severity="critical",
            details=(
                "No NaNs in critical columns."
                if bad.empty
                else bad.to_dict().__repr__()
            ),
        )

    def check_correctness_flags_recompute() -> dict[str, Any]:
        linear_recomputed = (
            labels["gold_linear_abs"].to_numpy()
            == labels["pred_linear_abs"].to_numpy()
        ).astype(int)

        bologna_recomputed = (
            labels["gold_bologna"].to_numpy()
            == labels["pred_bologna"].to_numpy()
        ).astype(int)

        linear_bad = int(
            np.sum(linear_recomputed != labels["correct_linear_abs"].to_numpy())
        )
        bologna_bad = int(
            np.sum(bologna_recomputed != labels["correct_bologna"].to_numpy())
        )

        return make_sanity_row(
            check_name="correctness_flags_recompute",
            passed=(linear_bad == 0 and bologna_bad == 0),
            severity="critical",
            details=f"linear_bad={linear_bad}; bologna_bad={bologna_bad}",
        )

    def check_bologna_labels_valid() -> dict[str, Any]:
        valid_labels = set(cfg.BOLOGNA_ORDERED_LABELS)

        gold_invalid = sorted(set(labels["gold_bologna"].unique()) - valid_labels)
        pred_invalid = sorted(set(labels["pred_bologna"].unique()) - valid_labels)

        return make_sanity_row(
            check_name="bologna_labels_valid",
            passed=(len(gold_invalid) == 0 and len(pred_invalid) == 0),
            severity="critical",
            details=f"gold_invalid={gold_invalid}; pred_invalid={pred_invalid}",
        )

    def check_ranking_recompute_matches() -> dict[str, Any]:
        recomputed = compute_model_ranking(labels)

        merged = ranking.merge(
            recomputed,
            on=["model_col", "model"],
            suffixes=("_original", "_recomputed"),
            validate="one_to_one",
        )

        metric_cols = [
            "el_acc_linear_abs",
            "el_qwk_linear_abs",
            "el_acc_bologna",
            "el_qwk_bologna",
        ]

        max_diffs: dict[str, float] = {}

        for col in metric_cols:
            diff = np.abs(
                merged[f"{col}_original"].to_numpy(dtype=float)
                - merged[f"{col}_recomputed"].to_numpy(dtype=float)
            )
            max_diffs[col] = float(np.max(diff)) if len(diff) else float("nan")

        passed = all(v <= 1e-12 for v in max_diffs.values())

        return make_sanity_row(
            check_name="ranking_recompute_matches",
            passed=passed,
            severity="critical",
            details=f"max_diffs={max_diffs}",
        )

    def check_best_model_pairs_with_all_others() -> dict[str, Any]:
        failed: list[str] = []

        for metric_col, best_model_col in best_model_cols.items():
            other_model_cols = [
                str(x)
                for x in ranking["model_col"].tolist()
                if str(x) != best_model_col
            ]

            for other in other_model_cols:
                try:
                    merged = paired_model_frame(labels, best_model_col, other)

                    if merged.empty:
                        failed.append(
                            f"{metric_col}: {short_model_name(best_model_col)} vs "
                            f"{short_model_name(other)}: empty merge"
                        )
                except Exception as exc:
                    failed.append(
                        f"{metric_col}: {short_model_name(best_model_col)} vs "
                        f"{short_model_name(other)}: {repr(exc)}"
                    )

        return make_sanity_row(
            check_name="metric_specific_best_model_pairs_with_all_others",
            passed=len(failed) == 0,
            severity="critical",
            details=(
                "All metric-specific best-vs-other pairings valid."
                if not failed
                else "; ".join(failed)
            ),
        )

    def check_expected_models_present() -> dict[str, Any]:
        actual = set(labels["model_col"].unique())
        expected = set(model_cols)

        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        return make_sanity_row(
            check_name="expected_models_present",
            passed=(len(missing) == 0 and len(extra) == 0),
            severity="critical",
            details=f"missing={missing}; extra={extra}",
        )

    def check_exam_count_reasonable() -> dict[str, Any]:
        n_virtual_exams = int(labels[TEST_ID_COL].nunique())
        return make_sanity_row(
            check_name="virtual_exam_count_reasonable",
            passed=n_virtual_exams > 0,
            severity="critical",
            details=f"n_virtual_exams={n_virtual_exams}",
        )

    def check_model_row_counts_equal() -> dict[str, Any]:
        counts = labels.groupby("model_col").size().to_dict()
        unique_counts = sorted(set(int(v) for v in counts.values()))

        return make_sanity_row(
            check_name="model_row_counts_equal",
            passed=len(unique_counts) == 1,
            severity="warning",
            details=(
                "row_counts={"
                + ", ".join(f"{short_model_name(k)}: {v}" for k, v in counts.items())
                + "}"
            ),
        )

    sanity_checks = [
        check_significance_selection_metrics_exist,
        check_env_test_size_exists,
        check_labels_not_empty,
        check_expected_models_present,
        check_one_row_per_model_exam_student,
        check_all_model_keys_identical,
        check_gold_labels_identical_across_models,
        check_complete_exam_counts,
        check_no_nan_in_critical_label_columns,
        check_correctness_flags_recompute,
        check_bologna_labels_valid,
        check_ranking_recompute_matches,
        check_best_model_pairs_with_all_others,
        check_exam_count_reasonable,
        check_model_row_counts_equal,
    ]

    for check in tqdm(
        sanity_checks,
        desc="Running sanity checks",
        unit="check",
    ):
        try:
            rows.append(check())
        except Exception as exc:
            rows.append(
                make_sanity_row(
                    check_name=check.__name__,
                    passed=False,
                    severity="critical",
                    details=f"Exception during sanity check: {repr(exc)}",
                )
            )

    return pd.DataFrame(rows)


# Writes a detailed sanity-check report to disk.
#
# The report contains:
#   - total number of checks
#   - passed/failed counts
#   - critical failures
#   - warning failures
#   - full details for every sanity check
def write_sanity_report(
    sanity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append(f"VEX SANITY CHECK REPORT - Q{TEST_SIZE}")
    lines.append("=" * 100)
    lines.append("")

    n_total = int(len(sanity_df))
    n_passed = int(sanity_df["passed"].sum())
    n_failed = int((~sanity_df["passed"]).sum())

    critical_failed = sanity_df[
        (~sanity_df["passed"])
        & (sanity_df["severity"] == "critical")
    ]

    warning_failed = sanity_df[
        (~sanity_df["passed"])
        & (sanity_df["severity"] == "warning")
    ]

    lines.append(f"total_checks={n_total}")
    lines.append(f"passed_checks={n_passed}")
    lines.append(f"failed_checks={n_failed}")
    lines.append(f"critical_failed={len(critical_failed)}")
    lines.append(f"warning_failed={len(warning_failed)}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("ALL CHECKS")
    lines.append("=" * 100)
    lines.append(
        sanity_df[
            [
                "check",
                "passed",
                "severity",
                "details",
            ]
        ].to_string(index=False)
    )
    lines.append("")

    if not critical_failed.empty:
        lines.append("=" * 100)
        lines.append("CRITICAL FAILURES")
        lines.append("=" * 100)
        lines.append(critical_failed.to_string(index=False))
        lines.append("")

    if not warning_failed.empty:
        lines.append("=" * 100)
        lines.append("WARNING FAILURES")
        lines.append("=" * 100)
        lines.append(warning_failed.to_string(index=False))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# Stops the pipeline if any critical sanity check failed.
#
# This is intentionally called after writing output files, so the diagnostic
# reports are still available for debugging.
#
# The idea is simple:
#   if critical assumptions are violated, do not trust/report the significance
#   results as final.
def assert_no_critical_sanity_failures(sanity_df: pd.DataFrame) -> None:
    critical_failed = sanity_df[
        (~sanity_df["passed"])
        & (sanity_df["severity"] == "critical")
    ]

    if not critical_failed.empty:
        raise ValueError(
            "Critical sanity checks failed. Do not report the significance results yet.\n\n"
            f"{critical_failed.to_string(index=False)}"
        )


# =========================================================
# REPORTING
# =========================================================

# Formats p-values for text reports.
#
# Very small p-values are shown in scientific notation.
# Other p-values are shown with six decimal places.
def format_p_value(p: float) -> str:
    if pd.isna(p):
        return "nan"

    if p < 0.0001:
        return f"{p:.2e}"

    return f"{p:.6f}"


# Writes the main statistical significance report.
#
# The report includes:
#   - input/configuration summary
#   - model ranking
#   - metric-specific best models
#   - McNemar EL-Acc test results
#   - paired permutation EL-QWK test results
#   - sanity-check summary
#
# This is the main human-readable output of the script.
def write_text_report(
    ranking: pd.DataFrame,
    best_models_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    sanity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append(f"VEX STATISTICAL SIGNIFICANCE - Q{TEST_SIZE}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"input_path={input_env_path().resolve()}")
    lines.append(f"test_size={TEST_SIZE}")
    lines.append(f"display_ranking_metric={DISPLAY_RANKING_METRIC}")
    lines.append("best_model_selection=metric_specific")
    lines.append(f"n_permutations={N_PERMUTATIONS}")
    lines.append(f"random_seed={RANDOM_SEED}")
    lines.append(f"alpha={ALPHA}")
    lines.append(f"n_jobs={N_JOBS}")
    lines.append(f"progress_every_permutations={PROGRESS_EVERY_PERMUTATIONS}")
    lines.append("")
    lines.append("Important:")
    lines.append("- EL-Acc significance is tested with exact McNemar tests.")
    lines.append("- EL-QWK significance is tested with paired permutation tests.")
    lines.append("- EL-QWK is recomputed per virtual exam and then averaged, matching the paper definition.")
    lines.append("- Permutation p-values use +1 correction: (count_extreme + 1) / (n_permutations + 1).")
    lines.append("- The permutation loop uses a fast integer QWK implementation.")
    lines.append("- Best model selection is metric-specific.")
    lines.append("")

    lines.append("=" * 100)
    lines.append("MODEL RANKING")
    lines.append("=" * 100)
    lines.append(
        ranking[
            [
                "model",
                "n_exam_student_rows",
                "el_acc_linear_abs",
                "el_qwk_linear_abs",
                "el_acc_bologna",
                "el_qwk_bologna",
            ]
        ].to_string(index=False)
    )
    lines.append("")

    lines.append("=" * 100)
    lines.append("METRIC-SPECIFIC BEST MODELS")
    lines.append("=" * 100)
    lines.append(
        best_models_df[
            [
                "test_family",
                "scale",
                "selection_metric",
                "best_model",
                "best_metric_value",
            ]
        ].to_string(index=False)
    )
    lines.append("")

    lines.append("=" * 100)
    lines.append("MCNEMAR TESTS FOR EL-ACC")
    lines.append("=" * 100)
    lines.append(
        "Interpretation: tests whether model A and model B differ in exact final-grade correctness "
        "on the same paired virtual-exam/student rows."
    )
    lines.append("")
    lines.append(
        mcnemar_df[
            [
                "scale",
                "model_a",
                "model_b",
                "n_paired",
                "both_correct",
                "a_correct_b_wrong",
                "a_wrong_b_correct",
                "both_wrong",
                "discordant",
                "p_value",
                "significant_0_05",
            ]
        ].to_string(
            index=False,
            formatters={"p_value": format_p_value},
        )
    )
    lines.append("")

    lines.append("=" * 100)
    lines.append("PAIRED PERMUTATION TESTS FOR EL-QWK")
    lines.append("=" * 100)
    lines.append(
        "Interpretation: randomly swaps paired model predictions and tests whether the observed "
        "difference in mean EL-QWK is larger than expected under exchangeability."
    )
    lines.append("")
    lines.append(
        permutation_df[
            [
                "scale",
                "model_a",
                "model_b",
                "n_paired",
                "n_virtual_exams",
                "qwk_a",
                "qwk_b",
                "delta_qwk_a_minus_b",
                "count_extreme",
                "p_value",
                "significant_0_05",
            ]
        ].to_string(
            index=False,
            formatters={"p_value": format_p_value},
        )
    )
    lines.append("")

    lines.append("=" * 100)
    lines.append("SANITY CHECK SUMMARY")
    lines.append("=" * 100)

    n_total = int(len(sanity_df))
    n_passed = int(sanity_df["passed"].sum())
    n_failed = int((~sanity_df["passed"]).sum())
    n_critical_failed = int(
        ((~sanity_df["passed"]) & (sanity_df["severity"] == "critical")).sum()
    )

    lines.append(f"total_checks={n_total}")
    lines.append(f"passed_checks={n_passed}")
    lines.append(f"failed_checks={n_failed}")
    lines.append(f"critical_failed={n_critical_failed}")
    lines.append("")
    lines.append(
        sanity_df[
            [
                "check",
                "passed",
                "severity",
                "details",
            ]
        ].to_string(index=False)
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

# Main execution pipeline.
#
# Steps:
#   1. Resolve and validate the input dataframe path.
#   2. Select model columns for evaluation.
#   3. Print configuration and model list.
#   4. Load dataframe_env.parquet.
#   5. Validate required columns and duplicate constraints.
#   6. Build exam-level labels for all models.
#   7. Save exam-level significance labels.
#   8. Compute and save model ranking.
#   9. Select and save metric-specific best models.
#  10. Run significance tests:
#        - McNemar for EL-Acc
#        - paired permutation tests for EL-QWK
#  11. Save significance test outputs.
#  12. Run sanity checks.
#  13. Write sanity and final text reports.
#  14. Stop if critical sanity checks failed.
#  15. Print all output paths.
def main() -> int:
    env_path = input_env_path()

    if not env_path.exists():
        raise FileNotFoundError(
            f"dataframe_env.parquet not found: {env_path.resolve()}"
        )

    model_cols = get_eval_model_columns()

    n_comparisons = max(0, len(model_cols) - 1)
    n_significance_jobs = len(SIGNIFICANCE_TASKS) * n_comparisons
    n_permutation_tasks = sum(
        1
        for task in SIGNIFICANCE_TASKS
        if str(task["test_family"]) == "permutation"
    )
    total_permutation_steps = n_permutation_tasks * n_comparisons * N_PERMUTATIONS
    resolved_n_jobs = resolve_n_jobs(n_significance_jobs)

    print("=" * 100)
    print(f"VEX STATISTICAL SIGNIFICANCE - Q{TEST_SIZE}")
    print("=" * 100)
    print(f"Input:       {env_path.resolve()}")
    print(f"Output dir:  {OUTPUT_DIR.resolve()}")
    print(f"Display ranking metric: {DISPLAY_RANKING_METRIC}")
    print("Best model selection: metric-specific")
    print(f"N jobs:      {resolved_n_jobs}")
    print(f"Permutations per EL-QWK test: {N_PERMUTATIONS}")
    print(f"Total permutation steps:      {total_permutation_steps}")
    print(f"Progress update every:        {PROGRESS_EVERY_PERMUTATIONS} permutations")
    print("")

    print("Models used:")
    for col in model_cols:
        print(f"  - {col}")
    print("")

    df_env = pd.read_parquet(env_path)

    validate_env_df(df_env, model_cols)
    assert_no_duplicate_exam_student_question_pairs(df_env)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = build_exam_level_labels(
        df_env=df_env,
        model_cols=model_cols,
        test_size=TEST_SIZE,
    )

    labels_path = OUTPUT_DIR / f"q{TEST_SIZE}_exam_level_significance_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    ranking = compute_model_ranking(labels)

    best_models_df = compute_best_models_by_significance_task(ranking)

    best_models_path = OUTPUT_DIR / f"q{TEST_SIZE}_metric_specific_best_models.csv"
    best_models_df.to_csv(best_models_path, index=False)

    ranking_path = OUTPUT_DIR / f"q{TEST_SIZE}_model_ranking_for_significance.csv"
    ranking.to_csv(ranking_path, index=False)

    print("")
    print("Model ranking:")
    print(
        ranking[
            [
                "model",
                "el_acc_linear_abs",
                "el_qwk_linear_abs",
                "el_acc_bologna",
                "el_qwk_bologna",
            ]
        ].to_string(index=False)
    )
    print("")

    print("Metric-specific best models:")
    print(
        best_models_df[
            [
                "test_family",
                "scale",
                "selection_metric",
                "best_model",
                "best_metric_value",
            ]
        ].to_string(index=False)
    )
    print("")

    print("=" * 100)
    print("STEP 1/2: RUNNING SIGNIFICANCE TESTS")
    print("=" * 100)

    mcnemar_df, permutation_df = run_significance_tests(
        labels=labels,
        ranking=ranking,
    )

    mcnemar_path = OUTPUT_DIR / f"q{TEST_SIZE}_mcnemar_el_acc.csv"
    permutation_path = OUTPUT_DIR / f"q{TEST_SIZE}_permutation_el_qwk.csv"

    mcnemar_df.to_csv(mcnemar_path, index=False)
    permutation_df.to_csv(permutation_path, index=False)

    print("")
    print("=" * 100)
    print("STEP 2/2: RUNNING SANITY CHECKS")
    print("=" * 100)

    sanity_df = run_sanity_checks(
        df_env=df_env,
        labels=labels,
        ranking=ranking,
        model_cols=model_cols,
    )

    sanity_path = OUTPUT_DIR / f"q{TEST_SIZE}_sanity_checks.csv"
    sanity_report_path = OUTPUT_DIR / f"q{TEST_SIZE}_sanity_checks_report.txt"

    sanity_df.to_csv(sanity_path, index=False)
    write_sanity_report(
        sanity_df=sanity_df,
        output_path=sanity_report_path,
    )

    report_path = OUTPUT_DIR / f"q{TEST_SIZE}_statistical_significance_report.txt"

    write_text_report(
        ranking=ranking,
        best_models_df=best_models_df,
        mcnemar_df=mcnemar_df,
        permutation_df=permutation_df,
        sanity_df=sanity_df,
        output_path=report_path,
    )

    print("")
    print("Sanity check summary:")
    print(
        sanity_df[
            [
                "check",
                "passed",
                "severity",
                "details",
            ]
        ].to_string(index=False)
    )

    assert_no_critical_sanity_failures(sanity_df)

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Labels:              {labels_path.resolve()}")
    print(f"Ranking:             {ranking_path.resolve()}")
    print(f"Metric-specific best: {best_models_path.resolve()}")
    print(f"McNemar results:     {mcnemar_path.resolve()}")
    print(f"Permutation results: {permutation_path.resolve()}")
    print(f"Sanity checks:       {sanity_path.resolve()}")
    print(f"Sanity report:       {sanity_report_path.resolve()}")
    print(f"Text report:         {report_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())