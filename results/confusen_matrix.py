#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve()

PROJECT_ROOT = BASE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vex_metric.vex_config import (
    OUTPUT_PARQUET,
    TEST_ENV_FOLDER,
    TEST_ENV_METRICS_FOLDER,
    LINEAR_MIN_GRADE,
    LINEAR_MAX_GRADE,
    LINEAR_ROUNDING_STEP,
    LINEAR_PASS_THRESHOLD_NORM,
    BOLOGNA_PASS_THRESHOLD_NORM,
    BOLOGNA_PASSING_DISTRIBUTION,
    BOLOGNA_PASSING_LABELS,
    BOLOGNA_FAIL_LABEL,
    BOLOGNA_ORDERED_LABELS,
)

# =========================================================
# CONFIG
# =========================================================

GPT_MODEL_COL = "new_grade_openai/gpt-5.4"

TEST_ID_COL = "test_id"
TEST_SIZE_COL = "test_size"
QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"
ANSWER_ID_COL = "answer_id"
HUMAN_GRADE_COL = "human_grade"

OUTPUT_DIR = Path("confusion_matrices_gpt")

WRITE_ROW_NORMALIZED = True


# =========================================================
# PATH HELPERS
# =========================================================

def _input_env_parquet() -> Path:
    return Path(OUTPUT_PARQUET)


def _output_dir() -> Path:
    return OUTPUT_DIR


# =========================================================
# VALIDATION
# =========================================================

def _validate_df(df: pd.DataFrame) -> None:
    required = [
        TEST_ID_COL,
        TEST_SIZE_COL,
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        ANSWER_ID_COL,
        HUMAN_GRADE_COL,
        GPT_MODEL_COL,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Pflichtspalten fehlen im dataframe_env.parquet: {missing}"
        )


def _assert_no_duplicate_exam_student_question_pairs(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(
        subset=[TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
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
            .sort_values(
                [
                    TEST_ID_COL,
                    TEST_SIZE_COL,
                    STUDENT_ID_COL,
                    QUESTION_ID_COL,
                ]
            )
            .head(100)
        )

        raise ValueError(
            "dataframe_env.parquet enthält doppelte "
            "(test_id, test_size, member_id, question_id)-Paare. "
            "Das macht Exam-Level-Confusion-Matrices ungültig.\n"
            f"Beispiele:\n{duplicates.to_string(index=False)}"
        )


# =========================================================
# LINEAR SCALE
# =========================================================

def _round_and_clip_linear_grades(grades: pd.Series) -> pd.Series:
    grades = pd.to_numeric(grades, errors="coerce")

    if LINEAR_ROUNDING_STEP and LINEAR_ROUNDING_STEP > 0:
        grades = (grades / LINEAR_ROUNDING_STEP).round() * LINEAR_ROUNDING_STEP

    grades = grades.clip(
        lower=LINEAR_MIN_GRADE,
        upper=LINEAR_MAX_GRADE,
    )

    return grades


def _normalized_to_linear_grade_absolute(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    grades = LINEAR_MIN_GRADE + (
        (LINEAR_MAX_GRADE - LINEAR_MIN_GRADE) * numeric
    )

    return _round_and_clip_linear_grades(grades)


def _linear_grade_labels_float() -> list[float]:
    step = float(LINEAR_ROUNDING_STEP)

    if step <= 0:
        raise ValueError("LINEAR_ROUNDING_STEP muss > 0 sein.")

    labels = np.arange(
        float(LINEAR_MIN_GRADE),
        float(LINEAR_MAX_GRADE) + step / 2.0,
        step,
    )

    return [float(round(x, 4)) for x in labels]


def _linear_grade_labels_str() -> list[str]:
    return [_format_linear_grade_label(x) for x in _linear_grade_labels_float()]


def _format_linear_grade_label(value: Any) -> str:
    """
    Converts numeric linear grades into stable discrete class labels.

    Important:
    sklearn.confusion_matrix treats float arrays as continuous targets.
    Therefore linear grades must be converted to strings before building
    the confusion matrix.
    """
    if pd.isna(value):
        return "nan"

    numeric = float(value)

    if numeric.is_integer():
        return f"{numeric:.1f}"

    return f"{numeric:.1f}"


# =========================================================
# BOLOGNA SCALE
# =========================================================

def _label_for_rank_position(position_1_based: int, cutoffs: list[int]) -> str:
    for label, cutoff in zip(BOLOGNA_PASSING_LABELS, cutoffs):
        if position_1_based <= cutoff:
            return label

    return BOLOGNA_PASSING_LABELS[-1]


def _assign_bologna_labels_from_normalized(
    normalized_scores: pd.Series,
    test_size: int,
) -> pd.Series:
    scores = pd.to_numeric(normalized_scores, errors="coerce")

    if scores.empty:
        return pd.Series(dtype="object", index=normalized_scores.index)

    pass_threshold_abs = float(test_size) * float(BOLOGNA_PASS_THRESHOLD_NORM)
    absolute_points = scores * float(test_size)

    passed_mask = absolute_points >= pass_threshold_abs
    result = pd.Series(BOLOGNA_FAIL_LABEL, index=scores.index, dtype="object")

    passed = absolute_points[passed_mask].dropna()

    if passed.empty:
        return result

    n_passed = len(passed)

    cumulative = np.cumsum(BOLOGNA_PASSING_DISTRIBUTION)
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

        midpoint_rank = int(round((rank_start + rank_end) / 2))
        label = _label_for_rank_position(midpoint_rank, cutoffs)

        result.loc[group["idx"].tolist()] = label

        current_rank_start = rank_end + 1

    return result


# =========================================================
# EXAM-LEVEL STUDENT AGGREGATION
# =========================================================

def _student_totals_for_exam(
    exam_df: pd.DataFrame,
    test_size: int,
) -> pd.DataFrame:
    subset = exam_df[
        [
            STUDENT_ID_COL,
            QUESTION_ID_COL,
            ANSWER_ID_COL,
            HUMAN_GRADE_COL,
            GPT_MODEL_COL,
        ]
    ].copy()

    subset[HUMAN_GRADE_COL] = pd.to_numeric(
        subset[HUMAN_GRADE_COL],
        errors="coerce",
    )

    subset[GPT_MODEL_COL] = pd.to_numeric(
        subset[GPT_MODEL_COL],
        errors="coerce",
    )

    duplicate_mask = subset.duplicated(
        subset=[STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            subset.loc[
                duplicate_mask,
                [
                    STUDENT_ID_COL,
                    QUESTION_ID_COL,
                    ANSWER_ID_COL,
                ],
            ]
            .sort_values([STUDENT_ID_COL, QUESTION_ID_COL])
            .head(50)
        )

        raise ValueError(
            "Doppelte (member_id, question_id)-Paare in einer virtuellen Prüfung.\n"
            f"Beispiele:\n{duplicates.to_string(index=False)}"
        )

    grouped = (
        subset.groupby(STUDENT_ID_COL)
        .agg(
            n_rows=(QUESTION_ID_COL, "size"),
            n_questions=(QUESTION_ID_COL, "nunique"),
            human_valid=(HUMAN_GRADE_COL, lambda s: int(s.notna().sum())),
            pred_valid=(GPT_MODEL_COL, lambda s: int(s.notna().sum())),
            gold_total=(HUMAN_GRADE_COL, "sum"),
            pred_total=(GPT_MODEL_COL, "sum"),
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
        return grouped

    grouped["gold_norm"] = grouped["gold_total"] / float(test_size)
    grouped["pred_norm"] = grouped["pred_total"] / float(test_size)

    grouped["gold_linear_abs"] = _normalized_to_linear_grade_absolute(
        grouped["gold_norm"]
    )
    grouped["pred_linear_abs"] = _normalized_to_linear_grade_absolute(
        grouped["pred_norm"]
    )

    grouped["gold_linear_abs_label"] = grouped["gold_linear_abs"].map(
        _format_linear_grade_label
    )
    grouped["pred_linear_abs_label"] = grouped["pred_linear_abs"].map(
        _format_linear_grade_label
    )

    grouped["gold_bologna"] = _assign_bologna_labels_from_normalized(
        grouped["gold_norm"],
        test_size=int(test_size),
    )
    grouped["pred_bologna"] = _assign_bologna_labels_from_normalized(
        grouped["pred_norm"],
        test_size=int(test_size),
    )

    return grouped


def build_exam_level_gpt_predictions(df_env: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for (test_id, test_size), exam_df in df_env.groupby(
        [TEST_ID_COL, TEST_SIZE_COL],
        sort=True,
    ):
        test_size_int = int(test_size)

        student_df = _student_totals_for_exam(
            exam_df=exam_df.copy(),
            test_size=test_size_int,
        )

        if student_df.empty:
            continue

        student_df.insert(0, TEST_ID_COL, test_id)
        student_df.insert(1, TEST_SIZE_COL, test_size_int)

        rows.append(student_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# =========================================================
# CONFUSION MATRIX HELPERS
# =========================================================

def _row_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(float)

    row_sums = matrix.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )

    return normalized


def _save_confusion_matrix_csv(
    matrix: np.ndarray,
    labels: list[Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_matrix = pd.DataFrame(
        matrix,
        index=[f"human_{label}" for label in labels],
        columns=[f"gpt_{label}" for label in labels],
    )

    df_matrix.to_csv(output_path, index=True, encoding="utf-8")


def _plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[Any],
    title: str,
    output_path: Path,
    value_format: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("GPT predicted grade/label")
    ax.set_ylabel("Human grade/label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha="right")
    ax.set_yticklabels([str(label) for label in labels])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]

            if value_format == "int":
                text = f"{int(value)}"
            else:
                text = f"{value:.2f}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_matrix_bundle(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    labels: list[Any],
    output_prefix: Path,
    title: str,
) -> None:
    subset = df[[y_true_col, y_pred_col]].copy()
    subset[y_true_col] = subset[y_true_col].astype(str)
    subset[y_pred_col] = subset[y_pred_col].astype(str)

    y_true = subset[y_true_col].to_numpy()
    y_pred = subset[y_pred_col].to_numpy()

    labels = [str(label) for label in labels]

    matrix = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    _save_confusion_matrix_csv(
        matrix=matrix,
        labels=labels,
        output_path=output_prefix.with_suffix(".csv"),
    )

    _plot_confusion_matrix(
        matrix=matrix,
        labels=labels,
        title=title,
        output_path=output_prefix.with_suffix(".png"),
        value_format="int",
    )

    if WRITE_ROW_NORMALIZED:
        normalized = _row_normalize_matrix(matrix)

        normalized_prefix = output_prefix.parent / f"{output_prefix.name}_row_normalized"

        _save_confusion_matrix_csv(
            matrix=normalized,
            labels=labels,
            output_path=normalized_prefix.with_suffix(".csv"),
        )

        _plot_confusion_matrix(
            matrix=normalized,
            labels=labels,
            title=f"{title} - row normalized",
            output_path=normalized_prefix.with_suffix(".png"),
            value_format="float",
        )


def write_confusion_matrices(exam_level_df: pd.DataFrame) -> None:
    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    linear_labels = _linear_grade_labels_str()
    bologna_labels = BOLOGNA_ORDERED_LABELS

    # -----------------------------------------------------
    # Overall matrices across all virtual exams
    # -----------------------------------------------------
    _write_matrix_bundle(
        df=exam_level_df,
        y_true_col="gold_linear_abs_label",
        y_pred_col="pred_linear_abs_label",
        labels=linear_labels,
        output_prefix=out_dir / "overall_linear_abs_confusion_matrix_gpt",
        title="GPT vs Human - Absolute Linear Grade Scale - Overall",
    )

    _write_matrix_bundle(
        df=exam_level_df,
        y_true_col="gold_bologna",
        y_pred_col="pred_bologna",
        labels=bologna_labels,
        output_prefix=out_dir / "overall_bologna_confusion_matrix_gpt",
        title="GPT vs Human - Bologna Grade Scale - Overall",
    )

    # -----------------------------------------------------
    # Matrices per test size
    # -----------------------------------------------------
    for test_size in sorted(exam_level_df[TEST_SIZE_COL].dropna().astype(int).unique()):
        df_size = exam_level_df[
            exam_level_df[TEST_SIZE_COL].astype(int) == int(test_size)
        ].copy()

        size_dir = out_dir / f"test_size_{test_size}"
        size_dir.mkdir(parents=True, exist_ok=True)

        _write_matrix_bundle(
            df=df_size,
            y_true_col="gold_linear_abs_label",
            y_pred_col="pred_linear_abs_label",
            labels=linear_labels,
            output_prefix=size_dir / f"linear_abs_confusion_matrix_gpt_q{test_size}",
            title=f"GPT vs Human - Absolute Linear Grade Scale - Q{test_size}",
        )

        _write_matrix_bundle(
            df=df_size,
            y_true_col="gold_bologna",
            y_pred_col="pred_bologna",
            labels=bologna_labels,
            output_prefix=size_dir / f"bologna_confusion_matrix_gpt_q{test_size}",
            title=f"GPT vs Human - Bologna Grade Scale - Q{test_size}",
        )


# =========================================================
# REPORT
# =========================================================

def write_summary(exam_level_df: pd.DataFrame) -> None:
    out_dir = _output_dir()
    summary_path = out_dir / "confusion_matrix_summary.txt"

    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("GPT CONFUSION MATRIX SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Input dataframe: {Path(OUTPUT_PARQUET).resolve()}")
    lines.append(f"GPT model column: {GPT_MODEL_COL}")
    lines.append(f"Output folder: {out_dir.resolve()}")
    lines.append("")
    lines.append(f"Rows after exam-level aggregation: {len(exam_level_df)}")
    lines.append(f"Unique tests: {exam_level_df[TEST_ID_COL].nunique()}")
    lines.append(
        "Test sizes: "
        + ", ".join(
            map(
                str,
                sorted(
                    exam_level_df[TEST_SIZE_COL]
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                ),
            )
        )
    )
    lines.append("")

    lines.append("[Linear absolute grade distribution]")
    lines.append("")
    lines.append("Human:")
    lines.append(
        exam_level_df["gold_linear_abs_label"]
        .value_counts()
        .reindex(_linear_grade_labels_str(), fill_value=0)
        .to_string()
    )
    lines.append("")
    lines.append("GPT:")
    lines.append(
        exam_level_df["pred_linear_abs_label"]
        .value_counts()
        .reindex(_linear_grade_labels_str(), fill_value=0)
        .to_string()
    )
    lines.append("")

    lines.append("[Bologna grade distribution]")
    lines.append("")
    lines.append("Human:")
    lines.append(
        exam_level_df["gold_bologna"]
        .value_counts()
        .reindex(BOLOGNA_ORDERED_LABELS, fill_value=0)
        .to_string()
    )
    lines.append("")
    lines.append("GPT:")
    lines.append(
        exam_level_df["pred_bologna"]
        .value_counts()
        .reindex(BOLOGNA_ORDERED_LABELS, fill_value=0)
        .to_string()
    )
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


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
    print("GPT CONFUSION MATRICES")
    print("=" * 100)
    print(f"Input:      {input_path.resolve()}")
    print(f"GPT column: {GPT_MODEL_COL}")
    print(f"Output:     {_output_dir().resolve()}")
    print("")

    df_env = pd.read_parquet(input_path)

    _validate_df(df_env)
    _assert_no_duplicate_exam_student_question_pairs(df_env)

    print("Building exam-level GPT predictions...")
    exam_level_df = build_exam_level_gpt_predictions(df_env)

    if exam_level_df.empty:
        raise ValueError(
            "Keine vollständigen exam-level Studentenzeilen gefunden. "
            "Confusion-Matrices können nicht gebaut werden."
        )

    _output_dir().mkdir(parents=True, exist_ok=True)

    exam_level_output_path = _output_dir() / "gpt_exam_level_labels.parquet"
    exam_level_df.to_parquet(exam_level_output_path, index=False)

    print(f"Exam-level labels saved to: {exam_level_output_path.resolve()}")

    print("Writing confusion matrices...")
    write_confusion_matrices(exam_level_df)

    print("Writing summary...")
    write_summary(exam_level_df)

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Output folder: {_output_dir().resolve()}")


if __name__ == "__main__":
    main()