#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
VEX_METRIC_DIR = PROJECT_ROOT / "vex_metric"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VEX_METRIC_DIR))

import vex_config as cfg


# =========================================================
# CONFIG
# =========================================================

# None means: use all model columns configured in vex_metric/vex_config.py.
SELECTED_MODEL_COLUMNS: list[str] | None = None

INPUT_PARQUET = Path(cfg.OUTPUT_PARQUET)
OUTPUT_DIR = SCRIPT_PATH.parent / "confusion_matrices"

WRITE_ROW_NORMALIZED = True
WRITE_EXAM_LEVEL_LABELS = True
CLEAN_OUTPUT_DIR = True


# =========================================================
# COLUMN CONFIG
# =========================================================

TEST_ID_COL = "test_id"
TEST_SIZE_COL = "test_size"
STUDENT_ID_COL = "member_id"
QUESTION_ID_COL = "question_id"
ANSWER_ID_COL = "answer_id"
HUMAN_GRADE_COL = "human_grade"


# =========================================================
# PATH / NAME HELPERS
# =========================================================

def _model_columns() -> list[str]:
    if SELECTED_MODEL_COLUMNS is None:
        return list(cfg.MODEL_COLUMNS)
    return list(SELECTED_MODEL_COLUMNS)


def _safe_file_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or "model"


def _short_model_name(model_col: str) -> str:
    name = str(model_col)
    for prefix in ("new_grade_", "grade_", "pred_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
    return name.replace("/", "_")


def _scope_dir(scope_name: str) -> Path:
    return OUTPUT_DIR / scope_name


def _with_extension(path_without_extension: Path, extension: str) -> Path:
    return path_without_extension.parent / f"{path_without_extension.name}{extension}"


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Pandas kann die Parquet-Datei nicht lesen. "
            "Installiere dafuer pyarrow oder fastparquet, z.B. `pip install pyarrow`."
        ) from exc


def _write_dataframe_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path


# =========================================================
# VALIDATION
# =========================================================

def _validate_env_df(df: pd.DataFrame, model_cols: list[str]) -> None:
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
            f"Pflichtspalten fehlen im dataframe_env.parquet: {missing_required}"
        )

    missing_models = [col for col in model_cols if col not in df.columns]
    if missing_models:
        raise ValueError(
            f"Modellspalten fehlen im dataframe_env.parquet: {missing_models}"
        )


def _assert_no_duplicate_exam_student_question_pairs(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(
        subset=[TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL],
        keep=False,
    )

    if not duplicate_mask.any():
        return

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
        .sort_values([TEST_ID_COL, TEST_SIZE_COL, STUDENT_ID_COL, QUESTION_ID_COL])
        .head(100)
    )

    raise ValueError(
        "dataframe_env.parquet enthaelt doppelte "
        "(test_id, test_size, member_id, question_id)-Paare. "
        "Das macht Exam-Level-Confusion-Matrices ungueltig.\n"
        f"Beispiele:\n{duplicates.to_string(index=False)}"
    )


# =========================================================
# LINEAR SCALE
# =========================================================

def _round_and_clip_linear_grades(grades: pd.Series) -> pd.Series:
    grades = pd.to_numeric(grades, errors="coerce")

    if cfg.LINEAR_ROUNDING_STEP and cfg.LINEAR_ROUNDING_STEP > 0:
        grades = (
            grades / float(cfg.LINEAR_ROUNDING_STEP)
        ).round() * float(cfg.LINEAR_ROUNDING_STEP)

    return grades.clip(
        lower=float(cfg.LINEAR_MIN_GRADE),
        upper=float(cfg.LINEAR_MAX_GRADE),
    )


def _normalized_to_linear_grade_absolute(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    grades = float(cfg.LINEAR_MIN_GRADE) + (
        (float(cfg.LINEAR_MAX_GRADE) - float(cfg.LINEAR_MIN_GRADE)) * numeric
    )

    return _round_and_clip_linear_grades(grades)


def _format_linear_grade_label(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.1f}"


def _linear_grade_labels() -> list[str]:
    step = float(cfg.LINEAR_ROUNDING_STEP)
    labels = np.arange(
        float(cfg.LINEAR_MIN_GRADE),
        float(cfg.LINEAR_MAX_GRADE) + step / 2.0,
        step,
    )
    return [_format_linear_grade_label(value) for value in labels]


# =========================================================
# BOLOGNA SCALE
# =========================================================

def _label_for_rank_position(position_1_based: int, cutoffs: list[int]) -> str:
    for label, cutoff in zip(cfg.BOLOGNA_PASSING_LABELS, cutoffs):
        if position_1_based <= cutoff:
            return label
    return cfg.BOLOGNA_PASSING_LABELS[-1]


def _assign_bologna_labels_from_normalized(
    normalized_scores: pd.Series,
    test_size: int,
) -> pd.Series:
    scores = pd.to_numeric(normalized_scores, errors="coerce")
    result = pd.Series(cfg.BOLOGNA_FAIL_LABEL, index=scores.index, dtype="object")

    if scores.empty:
        return result

    absolute_points = scores * float(test_size)
    pass_threshold_abs = float(test_size) * float(cfg.BOLOGNA_PASS_THRESHOLD_NORM)

    passed = absolute_points[absolute_points >= pass_threshold_abs].dropna()
    if passed.empty:
        return result

    n_passed = len(passed)
    cumulative = np.cumsum(cfg.BOLOGNA_PASSING_DISTRIBUTION)
    cutoffs = [int(np.ceil(n_passed * value)) for value in cumulative]
    cutoffs[-1] = n_passed

    ranked = (
        pd.DataFrame({"idx": passed.index, "points": passed.values})
        .sort_values(["points", "idx"], ascending=[False, True])
        .reset_index(drop=True)
    )

    rank_start = 1
    for points_value, tied_group in ranked.groupby("points", sort=False):
        group_size = len(tied_group)
        rank_end = rank_start + group_size - 1
        midpoint_rank = int(round((rank_start + rank_end) / 2))
        label = _label_for_rank_position(midpoint_rank, cutoffs)
        result.loc[tied_group["idx"].tolist()] = label
        rank_start = rank_end + 1

    return result


# =========================================================
# EXAM-LEVEL LABELS
# =========================================================

def _complete_student_exam_rows(
    exam_df: pd.DataFrame,
    model_col: str,
    test_size: int,
) -> pd.DataFrame:
    subset = exam_df[
        [
            STUDENT_ID_COL,
            QUESTION_ID_COL,
            HUMAN_GRADE_COL,
            model_col,
        ]
    ].copy()

    subset[HUMAN_GRADE_COL] = pd.to_numeric(
        subset[HUMAN_GRADE_COL],
        errors="coerce",
    )
    subset[model_col] = pd.to_numeric(subset[model_col], errors="coerce")

    grouped = (
        subset.groupby(STUDENT_ID_COL)
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

    complete = grouped[complete_mask].copy()
    if complete.empty:
        return complete

    complete["gold_norm"] = complete["gold_total"] / float(test_size)
    complete["pred_norm"] = complete["pred_total"] / float(test_size)

    complete["gold_linear_abs"] = _normalized_to_linear_grade_absolute(
        complete["gold_norm"]
    )
    complete["pred_linear_abs"] = _normalized_to_linear_grade_absolute(
        complete["pred_norm"]
    )
    complete["gold_linear_abs_label"] = complete["gold_linear_abs"].map(
        _format_linear_grade_label
    )
    complete["pred_linear_abs_label"] = complete["pred_linear_abs"].map(
        _format_linear_grade_label
    )

    complete["gold_bologna"] = _assign_bologna_labels_from_normalized(
        complete["gold_norm"],
        test_size=int(test_size),
    )
    complete["pred_bologna"] = _assign_bologna_labels_from_normalized(
        complete["pred_norm"],
        test_size=int(test_size),
    )

    return complete


def build_exam_level_labels(
    df_env: pd.DataFrame,
    model_cols: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for model_col in model_cols:
        print(f"Building exam-level labels for: {model_col}")

        for (test_id, test_size), exam_df in df_env.groupby(
            [TEST_ID_COL, TEST_SIZE_COL],
            sort=True,
        ):
            test_size_int = int(test_size)

            labels = _complete_student_exam_rows(
                exam_df=exam_df,
                model_col=model_col,
                test_size=test_size_int,
            )

            if labels.empty:
                continue

            labels.insert(0, "model_col", model_col)
            labels.insert(1, "model", _short_model_name(model_col))
            labels.insert(2, TEST_ID_COL, test_id)
            labels.insert(3, TEST_SIZE_COL, test_size_int)

            rows.append(labels)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# =========================================================
# CONFUSION MATRICES
# =========================================================

def _row_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )


def _confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    table = pd.crosstab(
        pd.Series(y_true, name="human"),
        pd.Series(y_pred, name="model"),
        dropna=False,
    )
    table = table.reindex(index=labels, columns=labels, fill_value=0)
    return table.to_numpy(dtype=int)


def _save_matrix_csv(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_matrix = pd.DataFrame(
        matrix,
        index=[f"human_{label}" for label in labels],
        columns=[f"model_{label}" for label in labels],
    )
    df_matrix.to_csv(output_path, index=True, encoding="utf-8")


def _plot_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
    *,
    normalized: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Model prediction")
    ax.set_ylabel("Human reference")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    max_value = float(np.nanmax(matrix)) if matrix.size else 0.0
    threshold = max_value / 2.0 if max_value > 0 else 0.0

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            color = "white" if value > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                color=color,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _matrix_summary_row(
    *,
    model_col: str,
    scope: str,
    scale: str,
    matrix: np.ndarray,
) -> dict[str, Any]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else np.nan

    return {
        "model_col": model_col,
        "model": _short_model_name(model_col),
        "scope": scope,
        "scale": scale,
        "n": total,
        "correct": correct,
        "accuracy": accuracy,
    }


def _write_matrix_bundle(
    *,
    df: pd.DataFrame,
    model_col: str,
    scope: str,
    scale: str,
    y_true_col: str,
    y_pred_col: str,
    labels: list[str],
    output_prefix: Path,
) -> dict[str, Any]:
    subset = df[[y_true_col, y_pred_col]].dropna().copy()
    subset[y_true_col] = subset[y_true_col].astype(str)
    subset[y_pred_col] = subset[y_pred_col].astype(str)

    matrix = _confusion_matrix(
        y_true=subset[y_true_col].to_numpy(),
        y_pred=subset[y_pred_col].to_numpy(),
        labels=labels,
    )

    _save_matrix_csv(
        matrix=matrix,
        labels=labels,
        output_path=_with_extension(output_prefix, ".csv"),
    )
    _plot_matrix(
        matrix=matrix,
        labels=labels,
        title=f"{_short_model_name(model_col)} - {scope} - {scale}",
        output_path=_with_extension(output_prefix, ".png"),
        normalized=False,
    )

    if WRITE_ROW_NORMALIZED:
        normalized_matrix = _row_normalize_matrix(matrix)
        normalized_prefix = output_prefix.parent / (
            f"{output_prefix.name}_row_normalized"
        )
        _save_matrix_csv(
            matrix=normalized_matrix,
            labels=labels,
            output_path=_with_extension(normalized_prefix, ".csv"),
        )
        _plot_matrix(
            matrix=normalized_matrix,
            labels=labels,
            title=f"{_short_model_name(model_col)} - {scope} - {scale} - row normalized",
            output_path=_with_extension(normalized_prefix, ".png"),
            normalized=True,
        )

    return _matrix_summary_row(
        model_col=model_col,
        scope=scope,
        scale=scale,
        matrix=matrix,
    )


def write_confusion_matrices(exam_labels: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    linear_labels = _linear_grade_labels()
    bologna_labels = list(cfg.BOLOGNA_ORDERED_LABELS)

    scopes: list[tuple[str, pd.DataFrame]] = [("overall", exam_labels)]
    for test_size in sorted(exam_labels[TEST_SIZE_COL].dropna().astype(int).unique()):
        scope_df = exam_labels[
            exam_labels[TEST_SIZE_COL].astype(int) == int(test_size)
        ].copy()
        scopes.append((f"test_size_{test_size}", scope_df))

    for model_col, model_df in exam_labels.groupby("model_col", sort=False):
        model_token = _safe_file_token(_short_model_name(model_col))

        for scope_name, scope_df in scopes:
            scoped_model_df = scope_df[scope_df["model_col"] == model_col].copy()
            if scoped_model_df.empty:
                continue

            scope_dir = _scope_dir(scope_name)

            summary_rows.append(
                _write_matrix_bundle(
                    df=scoped_model_df,
                    model_col=str(model_col),
                    scope=scope_name,
                    scale="linear_abs",
                    y_true_col="gold_linear_abs_label",
                    y_pred_col="pred_linear_abs_label",
                    labels=linear_labels,
                    output_prefix=scope_dir / f"{model_token}_linear_abs",
                )
            )

            summary_rows.append(
                _write_matrix_bundle(
                    df=scoped_model_df,
                    model_col=str(model_col),
                    scope=scope_name,
                    scale="bologna",
                    y_true_col="gold_bologna",
                    y_pred_col="pred_bologna",
                    labels=bologna_labels,
                    output_prefix=scope_dir / f"{model_token}_bologna",
                )
            )

    return pd.DataFrame(summary_rows)


# =========================================================
# SUMMARY
# =========================================================

def write_text_summary(
    exam_labels: pd.DataFrame,
    matrix_summary: pd.DataFrame,
) -> None:
    output_path = OUTPUT_DIR / "confusion_matrix_summary.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("VEX CONFUSION MATRICES")
    lines.append("=" * 100)
    lines.append(f"Input dataframe: {INPUT_PARQUET.resolve()}")
    lines.append(f"Output folder:   {OUTPUT_DIR.resolve()}")
    lines.append(f"Models:          {exam_labels['model_col'].nunique()}")
    lines.append(f"Rows:            {len(exam_labels)}")
    lines.append(
        "Test sizes:      "
        + ", ".join(
            map(
                str,
                sorted(exam_labels[TEST_SIZE_COL].dropna().astype(int).unique()),
            )
        )
    )
    lines.append("")

    if not matrix_summary.empty:
        lines.append("Accuracy summary:")
        lines.append(
            matrix_summary.sort_values(["scope", "scale", "accuracy"], ascending=[
                True,
                True,
                False,
            ]).to_string(index=False)
        )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    model_cols = _model_columns()

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"dataframe_env.parquet nicht gefunden: {INPUT_PARQUET.resolve()}"
        )

    print("=" * 100)
    print("VEX CONFUSION MATRICES")
    print("=" * 100)
    print(f"Input:  {INPUT_PARQUET.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Models: {len(model_cols)}")
    print("")

    df_env = _read_parquet(INPUT_PARQUET)
    _validate_env_df(df_env, model_cols)
    _assert_no_duplicate_exam_student_question_pairs(df_env)

    if CLEAN_OUTPUT_DIR and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exam_labels = build_exam_level_labels(
        df_env=df_env,
        model_cols=model_cols,
    )

    if exam_labels.empty:
        raise ValueError(
            "Keine vollstaendigen exam-level Studentenzeilen gefunden. "
            "Confusion-Matrices koennen nicht gebaut werden."
        )

    if WRITE_EXAM_LEVEL_LABELS:
        labels_path = OUTPUT_DIR / "exam_level_confusion_labels.parquet"
        written_labels_path = _write_dataframe_parquet_or_csv(
            exam_labels,
            labels_path,
        )
        print(f"Exam-level labels saved to: {written_labels_path.resolve()}")

    matrix_summary = write_confusion_matrices(exam_labels)
    summary_csv_path = OUTPUT_DIR / "confusion_matrix_summary.csv"
    matrix_summary.to_csv(summary_csv_path, index=False, encoding="utf-8")

    write_text_summary(
        exam_labels=exam_labels,
        matrix_summary=matrix_summary,
    )

    print("")
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Summary CSV: {summary_csv_path.resolve()}")
    print(f"Output dir:   {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
