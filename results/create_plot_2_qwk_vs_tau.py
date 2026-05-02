#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Figure 2 style item-level vs. exam-level plots for VEX.

Run from:
    VEX/results/

Example:
    python create_plot_2_item_vs_exam.py

Inputs:
    - cfg.INPUT_PARQUET
    - cfg.TEST_ENV_FOLDER / cfg.TEST_ENV_METRICS_FOLDER / exam_level_precomputed_metrics.parquet

The item-level data preparation mirrors vex_metric/evaluate_dataframe.py:
    - read cfg.INPUT_PARQUET
    - keep answer_id, question_id, member_id, grade, and model columns
    - normalize IDs
    - drop rows with empty IDs
    - guard against duplicate (member_id, question_id) pairs
    - rename grade -> human_grade
    - drop duplicate answer_id

Additionally, this script filters to label_type == "gold" when that column is
available, because the figure should evaluate against gold labels only.

Metrics used:
    Item level:
        - item_mse
        - item_qwk
        - item_tau_b (sanity reference only)

    Exam level (means over virtual exams):
        - el_qwk_linear_abs
        - el_qwk_bologna
        - el_acc_linear_abs
        - el_acc_bologna

Outputs:
    results/figures_plot_2/
        figure_2_item_qwk_vs_el_qwk_linear_bologna.pdf
        figure_2_item_qwk_vs_el_qwk_linear_bologna.png

        figure_2_item_mse_vs_el_qwk_linear_bologna.pdf
        figure_2_item_mse_vs_el_qwk_linear_bologna.png

        figure_2_item_mse_vs_el_acc_linear_bologna.pdf
        figure_2_item_mse_vs_el_acc_linear_bologna.png

        figure_2_item_vs_exam_data.csv
        figure_2_sanity_check.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
)


# =========================================================
# PATH SETUP
# =========================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent
VEX_METRIC_DIR = PROJECT_ROOT / "vex_metric"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VEX_METRIC_DIR))

import vex_config as cfg


def resolve_project_path(path_like: str | Path) -> Path:
    """
    Resolve a path robustly when the script is run from VEX/results/.

    Search order:
        1) absolute path as-is
        2) PROJECT_ROOT / relative path
        3) VEX_METRIC_DIR / relative path
        4) SCRIPT_DIR / relative path
        5) fallback PROJECT_ROOT / relative path
    """
    path_obj = Path(path_like)

    if path_obj.is_absolute():
        return path_obj

    candidates = [
        PROJECT_ROOT / path_obj,
        VEX_METRIC_DIR / path_obj,
        SCRIPT_DIR / path_obj,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / path_obj


# =========================================================
# CONFIG
# =========================================================

INPUT_PARQUET = resolve_project_path(cfg.INPUT_PARQUET)

EXAM_METRICS_PARQUET = resolve_project_path(
    Path(cfg.TEST_ENV_FOLDER)
    / cfg.TEST_ENV_METRICS_FOLDER
    / "exam_level_precomputed_metrics.parquet"
)

OUTPUT_DIR = SCRIPT_DIR / "figures_plot_2"

MODEL_COLUMNS = list(cfg.MODEL_COLUMNS)

MODEL_COL = "model_col"
TEST_SIZE_COL = "test_size"
QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"
ANSWER_ID_COL = "answer_id"
INPUT_HUMAN_GRADE_COL = "grade"
HUMAN_GRADE_COL = "human_grade"
LABEL_TYPE_COL = "label_type"
GOLD_LABEL_VALUE = "gold"

LINEAR_EL_QWK_COL = "el_qwk_linear_abs"
BOLOGNA_EL_QWK_COL = "el_qwk_bologna"
LINEAR_EL_ACC_COL = "el_acc_linear_abs"
BOLOGNA_EL_ACC_COL = "el_acc_bologna"

DATA_CSV = OUTPUT_DIR / "figure_2_item_vs_exam_data.csv"
SANITY_CHECK_TXT = OUTPUT_DIR / "figure_2_sanity_check.txt"

FIG_QWK_VS_ELQWK_PDF = OUTPUT_DIR / "figure_2_item_qwk_vs_el_qwk_linear_bologna.pdf"
FIG_QWK_VS_ELQWK_PNG = OUTPUT_DIR / "figure_2_item_qwk_vs_el_qwk_linear_bologna.png"

FIG_MSE_VS_ELQWK_PDF = OUTPUT_DIR / "figure_2_item_mse_vs_el_qwk_linear_bologna.pdf"
FIG_MSE_VS_ELQWK_PNG = OUTPUT_DIR / "figure_2_item_mse_vs_el_qwk_linear_bologna.png"

FIG_MSE_VS_ELACC_PDF = OUTPUT_DIR / "figure_2_item_mse_vs_el_acc_linear_bologna.pdf"
FIG_MSE_VS_ELACC_PNG = OUTPUT_DIR / "figure_2_item_mse_vs_el_acc_linear_bologna.png"

STALE_OUTPUTS = [
    OUTPUT_DIR / "figure_2_qwk_vs_tau.pdf",
    OUTPUT_DIR / "figure_2_qwk_vs_tau.png",
    OUTPUT_DIR / "figure_2_scatter_item_qwk_vs_item_tau.pdf",
    OUTPUT_DIR / "figure_2_scatter_item_qwk_vs_item_tau.png",
    OUTPUT_DIR / "figure_2_rank_shift_item_qwk_vs_item_tau.pdf",
    OUTPUT_DIR / "figure_2_rank_shift_item_qwk_vs_item_tau.png",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_data.csv",
    OUTPUT_DIR / "figure_2_item_qwk_vs_el_qwk.pdf",
    OUTPUT_DIR / "figure_2_item_qwk_vs_el_qwk.png",
    OUTPUT_DIR / "figure_2_scatter_item_qwk_vs_el_qwk.pdf",
    OUTPUT_DIR / "figure_2_scatter_item_qwk_vs_el_qwk.png",
    OUTPUT_DIR / "figure_2_rank_shift_item_qwk_vs_el_qwk.pdf",
    OUTPUT_DIR / "figure_2_rank_shift_item_qwk_vs_el_qwk.png",
    OUTPUT_DIR / "figure_2_item_qwk_vs_el_qwk_data.csv",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_q10.pdf",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_q10.png",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_q15.pdf",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_q15.png",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_by_test_size.pdf",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_by_test_size.png",
    OUTPUT_DIR / "figure_2_qwk_vs_tau_per_virtual_test.csv",
]


# =========================================================
# MODEL LABELS
# =========================================================

DISPLAY_NAMES = {
    "new_grade_deepseek/deepseek-v3.2-thinking": "DeepSeek Thinking",
    "new_grade_deepseek/deepseek-v3.2": "DeepSeek",
    "new_grade_google/gemini-2.5-pro": "Gemini",
    "new_grade_anthropic/claude-sonnet-4.6": "Claude",
    "new_grade_openai/gpt-5.4": "GPT",
    "new_grade_llama32_3b_base": "Llama Base",
    "new_grade_gemma_e4_base": "Gemma Base",
    "new_grade_llama32_3b_ft": "Llama FT",
    "new_grade_gemma_e4_ft": "Gemma FT",
    "grade_bert_base": "BERT Base",
    "grade_bert_ft": "BERT FT",
    "grade_mdeberta_base": "mDeBERTa Base",
    "grade_mdeberta_ft": "mDeBERTa FT",
    "grade_prior_global": "Global Prior",
    "grade_prior_template_overlap": "Template",
    "pred_tfidf_v5_answer_char_3_5": "TF-IDF Char",
    "pred_tfidf_v1_answer_word_unigram": "TF-IDF Word",
    "pred_tfidf_v4_question_and_answer_separate": "TF-IDF QA",
}

FAMILY_COLORS = {
    "LLM": "#1f77b4",
    "Transformer": "#2ca02c",
    "Prior": "#9467bd",
    "TF-IDF": "#ff7f0e",
    "Other": "#666666",
}

FAMILY_MARKERS = {
    "LLM": "o",
    "Transformer": "s",
    "Prior": "D",
    "TF-IDF": "^",
    "Other": "o",
}


def display_name(model_col: str) -> str:
    return DISPLAY_NAMES.get(model_col, short_model_name(model_col))


def short_model_name(model_col: str) -> str:
    name = str(model_col)
    for prefix in ("new_grade_", "grade_", "pred_"):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
    return name.replace("/", "_")


def model_family(model_col: str) -> str:
    if model_col.startswith("new_grade_"):
        return "LLM"
    if model_col.startswith("grade_bert") or model_col.startswith("grade_mdeberta"):
        return "Transformer"
    if model_col.startswith("grade_prior"):
        return "Prior"
    if model_col.startswith("pred_tfidf"):
        return "TF-IDF"
    return "Other"


# =========================================================
# ITEM DATAFRAME PREPARATION
# =========================================================

def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Pandas cannot read the input parquet file. Install pyarrow or "
            "fastparquet, e.g. `pip install pyarrow`."
        ) from exc


def validate_original_item_df(df: pd.DataFrame) -> None:
    required = [
        ANSWER_ID_COL,
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        INPUT_HUMAN_GRADE_COL,
        *MODEL_COLUMNS,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in INPUT_PARQUET: {missing}")


def normalize_string_series(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip()


def assert_no_duplicate_student_question_pairs(
    df: pd.DataFrame,
    context: str,
) -> None:
    key_cols = [STUDENT_ID_COL, QUESTION_ID_COL]
    duplicate_mask = df.duplicated(subset=key_cols, keep=False)

    if duplicate_mask.any():
        preview_cols = [STUDENT_ID_COL, QUESTION_ID_COL, ANSWER_ID_COL]
        duplicates = (
            df.loc[duplicate_mask, preview_cols]
            .sort_values(key_cols)
            .head(50)
        )
        raise ValueError(
            f"{context}: duplicate (member_id, question_id) pairs found. "
            "This would make item-level evaluation ambiguous.\n"
            f"Examples:\n{duplicates.to_string(index=False)}"
        )


def build_gold_item_df_from_original_input(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_original_item_df(df)

    stats: dict[str, Any] = {
        "raw_rows": int(len(df)),
        "label_type_column_present": LABEL_TYPE_COL in df.columns,
        "gold_filter_applied": False,
        "rows_after_gold_filter": int(len(df)),
        "rows_after_empty_id_filter": None,
        "rows_after_answer_id_dedup": None,
        "duplicate_answer_ids_dropped": 0,
    }

    if LABEL_TYPE_COL in df.columns:
        label_values = (
            df[LABEL_TYPE_COL]
            .where(df[LABEL_TYPE_COL].notna(), "")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df = df[label_values == GOLD_LABEL_VALUE].copy()
        stats["gold_filter_applied"] = True
        stats["rows_after_gold_filter"] = int(len(df))

    keep_cols = [
        ANSWER_ID_COL,
        QUESTION_ID_COL,
        STUDENT_ID_COL,
        INPUT_HUMAN_GRADE_COL,
        *MODEL_COLUMNS,
    ]

    item_df = df[keep_cols].copy()

    item_df[ANSWER_ID_COL] = normalize_string_series(item_df[ANSWER_ID_COL])
    item_df[QUESTION_ID_COL] = normalize_string_series(item_df[QUESTION_ID_COL])
    item_df[STUDENT_ID_COL] = normalize_string_series(item_df[STUDENT_ID_COL])

    item_df = item_df[
        (item_df[ANSWER_ID_COL] != "")
        & (item_df[QUESTION_ID_COL] != "")
        & (item_df[STUDENT_ID_COL] != "")
    ].copy()
    stats["rows_after_empty_id_filter"] = int(len(item_df))

    assert_no_duplicate_student_question_pairs(
        item_df,
        context="Original INPUT_PARQUET after gold filter",
    )

    item_df = item_df.rename(columns={INPUT_HUMAN_GRADE_COL: HUMAN_GRADE_COL})

    before_dedup = len(item_df)
    item_df = item_df.drop_duplicates(subset=[ANSWER_ID_COL]).copy()
    stats["duplicate_answer_ids_dropped"] = int(before_dedup - len(item_df))
    stats["rows_after_answer_id_dedup"] = int(len(item_df))

    if item_df[ANSWER_ID_COL].duplicated().any():
        raise ValueError("Item-level dataframe contains duplicate answer_id values.")

    return item_df, stats


# =========================================================
# ITEM-LEVEL METRICS
# =========================================================

def _contingency_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    table = np.zeros((len(labels), len(labels)), dtype=float)

    for true_value, pred_value in zip(y_true, y_pred, strict=False):
        table[label_to_idx[true_value], label_to_idx[pred_value]] += 1.0

    return table, labels


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan

    if len(np.unique(y_true)) == 1 and len(np.unique(y_pred)) == 1:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    table, labels = _contingency_table(y_true, y_pred)
    n_labels = len(labels)

    if n_labels == 1:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0

    total = table.sum()
    if total == 0:
        return np.nan

    true_hist = table.sum(axis=1)
    pred_hist = table.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / total

    denom = float((n_labels - 1) ** 2)
    weights = np.fromfunction(
        lambda i, j: ((i - j) ** 2) / denom,
        (n_labels, n_labels),
        dtype=float,
    )

    observed_weighted = float((weights * table).sum())
    expected_weighted = float((weights * expected).sum())

    if expected_weighted == 0:
        return np.nan

    return 1.0 - (observed_weighted / expected_weighted)


def kendall_tau_b_from_ordinal_items(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return np.nan

    table, _labels = _contingency_table(y_true, y_pred)
    n_rows, n_cols = table.shape

    concordant = 0.0
    discordant = 0.0

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            count = table[row_idx, col_idx]
            if count == 0:
                continue

            concordant += count * table[row_idx + 1 :, col_idx + 1 :].sum()
            discordant += count * table[row_idx + 1 :, :col_idx].sum()

    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)

    ties_true = sum(value * (value - 1.0) / 2.0 for value in row_totals)
    ties_pred = sum(value * (value - 1.0) / 2.0 for value in col_totals)
    ties_both = sum(value * (value - 1.0) / 2.0 for value in table.ravel())

    ties_true_only = ties_true - ties_both
    ties_pred_only = ties_pred - ties_both

    denominator = np.sqrt(
        (concordant + discordant + ties_true_only)
        * (concordant + discordant + ties_pred_only)
    )

    if denominator == 0:
        return np.nan

    return float((concordant - discordant) / denominator)


def item_metrics_for_model(
    item_df: pd.DataFrame,
    model_col: str,
) -> dict[str, Any]:
    subset = item_df[[HUMAN_GRADE_COL, model_col]].copy()
    subset[HUMAN_GRADE_COL] = pd.to_numeric(
        subset[HUMAN_GRADE_COL],
        errors="coerce",
    )
    subset[model_col] = pd.to_numeric(subset[model_col], errors="coerce")

    total_items = len(subset)
    missing_predictions = int(subset[model_col].isna().sum())
    subset = subset.dropna()

    y_true = subset[HUMAN_GRADE_COL].to_numpy(dtype=float)
    y_pred = subset[model_col].to_numpy(dtype=float)

    if len(subset) == 0:
        mae = mse = rmse = np.nan
    else:
        errors = y_true - y_pred
        mae = float(np.mean(np.abs(errors)))
        mse = float(np.mean(errors**2))
        rmse = float(np.sqrt(mse))

    return {
        "model_col": model_col,
        "model": display_name(model_col),
        "family": model_family(model_col),
        "item_source_n": int(total_items),
        "item_n": int(len(subset)),
        "item_missing_predictions": missing_predictions,
        "item_mae": mae,
        "item_mse": mse,
        "item_rmse": rmse,
        "item_qwk": quadratic_weighted_kappa(y_true, y_pred),
        "item_tau_b": kendall_tau_b_from_ordinal_items(y_true, y_pred),
    }


def compute_item_metrics(item_df: pd.DataFrame) -> pd.DataFrame:
    rows = [item_metrics_for_model(item_df, model_col) for model_col in MODEL_COLUMNS]
    metrics_df = pd.DataFrame(rows)

    metrics_df["item_qwk_rank"] = metrics_df["item_qwk"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    metrics_df["item_tau_b_rank"] = metrics_df["item_tau_b"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    metrics_df["item_mse_rank"] = metrics_df["item_mse"].rank(
        ascending=True,
        method="min",
        na_option="bottom",
    )

    return metrics_df.sort_values(
        ["family", "item_qwk_rank", "item_tau_b_rank", "model"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


# =========================================================
# EXAM-LEVEL METRICS
# =========================================================

def validate_exam_metrics_df(df: pd.DataFrame) -> None:
    required = [
        MODEL_COL,
        TEST_SIZE_COL,
        LINEAR_EL_QWK_COL,
        BOLOGNA_EL_QWK_COL,
        LINEAR_EL_ACC_COL,
        BOLOGNA_EL_ACC_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in exam metrics parquet: {missing}")


def compute_exam_summary(
    exam_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_exam_metrics_df(exam_df)

    subset = exam_df[
        [
            MODEL_COL,
            TEST_SIZE_COL,
            LINEAR_EL_QWK_COL,
            BOLOGNA_EL_QWK_COL,
            LINEAR_EL_ACC_COL,
            BOLOGNA_EL_ACC_COL,
        ]
    ].copy()

    subset[MODEL_COL] = normalize_string_series(subset[MODEL_COL])
    subset[TEST_SIZE_COL] = pd.to_numeric(subset[TEST_SIZE_COL], errors="coerce")
    subset[LINEAR_EL_QWK_COL] = pd.to_numeric(subset[LINEAR_EL_QWK_COL], errors="coerce")
    subset[BOLOGNA_EL_QWK_COL] = pd.to_numeric(subset[BOLOGNA_EL_QWK_COL], errors="coerce")
    subset[LINEAR_EL_ACC_COL] = pd.to_numeric(subset[LINEAR_EL_ACC_COL], errors="coerce")
    subset[BOLOGNA_EL_ACC_COL] = pd.to_numeric(subset[BOLOGNA_EL_ACC_COL], errors="coerce")

    subset = subset[subset[MODEL_COL].isin(MODEL_COLUMNS)].copy()

    stats: dict[str, Any] = {
        "exam_metric_rows_raw": int(len(exam_df)),
        "exam_metric_rows_for_configured_models": int(len(subset)),
        "exam_metric_models": int(subset[MODEL_COL].nunique()),
        "exam_metric_test_sizes": sorted(
            int(value)
            for value in subset[TEST_SIZE_COL].dropna().unique().tolist()
        ),
        "exam_metric_linear_qwk_source_column": LINEAR_EL_QWK_COL,
        "exam_metric_bologna_qwk_source_column": BOLOGNA_EL_QWK_COL,
        "exam_metric_linear_acc_source_column": LINEAR_EL_ACC_COL,
        "exam_metric_bologna_acc_source_column": BOLOGNA_EL_ACC_COL,
    }

    summary = (
        subset.groupby(MODEL_COL, dropna=False)
        .agg(
            el_qwk_linear=("el_qwk_linear_abs", "mean"),
            el_qwk_linear_std=("el_qwk_linear_abs", "std"),
            el_qwk_linear_n=("el_qwk_linear_abs", "count"),
            el_qwk_linear_missing=("el_qwk_linear_abs", lambda s: int(s.isna().sum())),

            el_qwk_bologna=("el_qwk_bologna", "mean"),
            el_qwk_bologna_std=("el_qwk_bologna", "std"),
            el_qwk_bologna_n=("el_qwk_bologna", "count"),
            el_qwk_bologna_missing=("el_qwk_bologna", lambda s: int(s.isna().sum())),

            el_acc_linear=("el_acc_linear_abs", "mean"),
            el_acc_linear_std=("el_acc_linear_abs", "std"),
            el_acc_linear_n=("el_acc_linear_abs", "count"),
            el_acc_linear_missing=("el_acc_linear_abs", lambda s: int(s.isna().sum())),

            el_acc_bologna=("el_acc_bologna", "mean"),
            el_acc_bologna_std=("el_acc_bologna", "std"),
            el_acc_bologna_n=("el_acc_bologna", "count"),
            el_acc_bologna_missing=("el_acc_bologna", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    return summary, stats


def merge_item_and_exam_metrics(
    item_metrics_df: pd.DataFrame,
    exam_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    plot_df = item_metrics_df.merge(
        exam_summary_df,
        on=MODEL_COL,
        how="left",
        validate="one_to_one",
    )

    plot_df["el_qwk_linear_rank"] = plot_df["el_qwk_linear"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    plot_df["el_qwk_bologna_rank"] = plot_df["el_qwk_bologna"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    plot_df["el_acc_linear_rank"] = plot_df["el_acc_linear"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    plot_df["el_acc_bologna_rank"] = plot_df["el_acc_bologna"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )

    return plot_df.sort_values(
        ["family", "item_qwk_rank", "el_qwk_linear_rank", "model"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


# =========================================================
# SANITY CHECK
# =========================================================

def write_sanity_check(
    plot_df: pd.DataFrame,
    item_stats: dict[str, Any],
    exam_stats: dict[str, Any],
) -> None:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("FIGURE 2 SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Item input parquet: {INPUT_PARQUET.resolve()}")
    lines.append(f"Exam metrics parquet: {EXAM_METRICS_PARQUET.resolve()}")
    lines.append(
        "Item dataframe logic: evaluate_dataframe.py::_build_item_df_from_original_input, "
        "plus label_type == 'gold' filter when available."
    )
    lines.append("")
    lines.append("Input/filter counts:")
    for key, value in item_stats.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Exam-level metric counts:")
    for key, value in exam_stats.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Metrics:")
    lines.append("  item_mse       = mean((human_grade - model_prediction)^2) over gold items")
    lines.append("  item_qwk       = quadratic weighted kappa over gold items")
    lines.append("  el_qwk_linear  = mean el_qwk_linear_abs over virtual exams")
    lines.append("  el_qwk_bologna = mean el_qwk_bologna over virtual exams")
    lines.append("  el_acc_linear  = mean el_acc_linear_abs over virtual exams")
    lines.append("  el_acc_bologna = mean el_acc_bologna over virtual exams")
    lines.append("  item_tau_b     = Kendall's tau-b over gold items, sanity reference only")
    lines.append("")

    table_cols = [
        "model_col",
        "model",
        "family",
        "item_source_n",
        "item_n",
        "item_missing_predictions",
        "item_mse",
        "item_qwk",
        "item_tau_b",
        "el_qwk_linear",
        "el_qwk_linear_std",
        "el_qwk_linear_n",
        "el_qwk_linear_missing",
        "el_qwk_bologna",
        "el_qwk_bologna_std",
        "el_qwk_bologna_n",
        "el_qwk_bologna_missing",
        "el_acc_linear",
        "el_acc_linear_std",
        "el_acc_linear_n",
        "el_acc_linear_missing",
        "el_acc_bologna",
        "el_acc_bologna_std",
        "el_acc_bologna_n",
        "el_acc_bologna_missing",
        "item_mse_rank",
        "item_qwk_rank",
        "item_tau_b_rank",
        "el_qwk_linear_rank",
        "el_qwk_bologna_rank",
        "el_acc_linear_rank",
        "el_acc_bologna_rank",
    ]

    table = plot_df[table_cols].copy()

    for col in [
        "item_mse",
        "item_qwk",
        "item_tau_b",
        "el_qwk_linear",
        "el_qwk_linear_std",
        "el_qwk_bologna",
        "el_qwk_bologna_std",
        "el_acc_linear",
        "el_acc_linear_std",
        "el_acc_bologna",
        "el_acc_bologna_std",
    ]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(6)

    lines.append("Per-model values used in the plots:")
    lines.append(table.to_string(index=False))
    lines.append("")
    lines.append("Sorted by item_qwk descending:")
    lines.append(
        table.sort_values(["item_qwk_rank", "el_qwk_linear_rank", "model"]).to_string(index=False)
    )
    lines.append("")
    lines.append("Sorted by item_mse ascending:")
    lines.append(
        table.sort_values(["item_mse_rank", "el_qwk_linear_rank", "model"]).to_string(index=False)
    )
    lines.append("")
    lines.append("Sorted by el_qwk_linear descending:")
    lines.append(
        table.sort_values(["el_qwk_linear_rank", "item_qwk_rank", "model"]).to_string(index=False)
    )
    lines.append("")
    lines.append("Sorted by el_acc_linear descending:")
    lines.append(
        table.sort_values(["el_acc_linear_rank", "item_mse_rank", "model"]).to_string(index=False)
    )

    SANITY_CHECK_TXT.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# PLOTTING
# =========================================================

def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def finite_limits(values: pd.Series, pad: float = 0.04) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return 0.0, 1.0

    lower = float(finite.min())
    upper = float(finite.max())

    if lower == upper:
        lower -= pad
        upper += pad
    else:
        span = upper - lower
        lower -= span * pad
        upper += span * pad

    return lower, upper


def plot_scatter_generic(
    plot_df: pd.DataFrame,
    ax: plt.Axes,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    panel_label: str,
) -> None:
    scatter_df = plot_df.dropna(subset=[x_col, y_col]).copy()

    for family, family_df in scatter_df.groupby("family", sort=False):
        ax.scatter(
            family_df[x_col],
            family_df[y_col],
            s=70,
            marker=FAMILY_MARKERS.get(family, "o"),
            color=FAMILY_COLORS.get(family, "#666666"),
            edgecolor="black",
            linewidth=0.5,
            label=family,
            alpha=0.9,
            zorder=2,
        )

    label_offsets = {
        "GPT": (7, -2),
        "Gemini": (7, 5),
        "Claude": (7, 6),
        "DeepSeek Thinking": (7, -2),
        "DeepSeek": (7, -8),
        "Gemma FT": (7, 3),
        "Gemma Base": (7, -8),
        "TF-IDF Char": (7, -5),
        "TF-IDF Word": (7, 4),
        "TF-IDF QA": (7, 9),
    }

    for _, row in scatter_df.iterrows():
        xytext = label_offsets.get(str(row["model"]), (5, 4))
        ax.annotate(
            str(row["model"]),
            xy=(row[x_col], row[y_col]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=7,
        )

    x_min, x_max = finite_limits(scatter_df[x_col])
    y_min, y_max = finite_limits(scatter_df[y_col])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{panel_label} {x_label} vs. {y_label}")

    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_slope_chart_generic(
    plot_df: pd.DataFrame,
    ax: plt.Axes,
    x_col: str,
    y_col: str,
    x_label_short: str,
    y_label_short: str,
    panel_label: str,
    x_ascending: bool,
    y_ascending: bool,
) -> None:
    ranked = plot_df.dropna(subset=[x_col, y_col]).copy()

    x_order = ranked.sort_values(
        [x_col, "model"],
        ascending=[x_ascending, True],
        na_position="last",
    )
    y_order = ranked.sort_values(
        [y_col, "model"],
        ascending=[y_ascending, True],
        na_position="last",
    )

    x_plot_rank = {
        model_col: rank
        for rank, model_col in enumerate(x_order["model_col"], start=1)
    }
    y_plot_rank = {
        model_col: rank
        for rank, model_col in enumerate(y_order["model_col"], start=1)
    }

    ranked["x_plot_rank"] = ranked["model_col"].map(x_plot_rank)
    ranked["y_plot_rank"] = ranked["model_col"].map(y_plot_rank)

    max_rank = len(ranked)

    left_x = 0.25
    right_x = 0.40

    ax.set_xlim(0.05, 0.62)
    ax.set_ylim(max_rank + 0.75, 0.25)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels([x_label_short, y_label_short])
    ax.set_yticks(np.arange(1, max_rank + 1, 2))
    ax.set_ylabel("Model rank (1 = best)")
    ax.set_title(f"{panel_label} Rank Shift: {x_label_short} vs. {y_label_short}")

    for _, row in ranked.iterrows():
        color = FAMILY_COLORS.get(row["family"], "#666666")

        ax.plot(
            [left_x, right_x],
            [row["x_plot_rank"], row["y_plot_rank"]],
            color=color,
            linewidth=1.3,
            alpha=0.8,
        )
        ax.scatter(
            [left_x, right_x],
            [row["x_plot_rank"], row["y_plot_rank"]],
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=35,
            zorder=3,
        )

        ax.text(
            left_x - 0.03,
            row["x_plot_rank"],
            str(row["model"]),
            ha="right",
            va="center",
            fontsize=7,
        )
        ax.text(
            right_x + 0.03,
            row["y_plot_rank"],
            str(row["model"]),
            ha="left",
            va="center",
            fontsize=7,
        )

    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def save_comparison_figure(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    x_label_short: str,
    x_ascending: bool,
    linear_metric_col: str,
    linear_metric_label: str,
    linear_metric_short: str,
    linear_ascending: bool,
    bologna_metric_col: str,
    bologna_metric_label: str,
    bologna_metric_short: str,
    bologna_ascending: bool,
    output_pdf: Path,
    output_png: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10.8, 9.8),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
    )

    # Top row: linear
    plot_scatter_generic(
        plot_df=plot_df,
        ax=axes[0, 0],
        x_col=x_col,
        y_col=linear_metric_col,
        x_label=x_label,
        y_label=linear_metric_label,
        panel_label="(a)",
    )
    plot_slope_chart_generic(
        plot_df=plot_df,
        ax=axes[0, 1],
        x_col=x_col,
        y_col=linear_metric_col,
        x_label_short=x_label_short,
        y_label_short=linear_metric_short,
        panel_label="(b)",
        x_ascending=x_ascending,
        y_ascending=linear_ascending,
    )

    # Bottom row: bologna
    plot_scatter_generic(
        plot_df=plot_df,
        ax=axes[1, 0],
        x_col=x_col,
        y_col=bologna_metric_col,
        x_label=x_label,
        y_label=bologna_metric_label,
        panel_label="(c)",
    )
    plot_slope_chart_generic(
        plot_df=plot_df,
        ax=axes[1, 1],
        x_col=x_col,
        y_col=bologna_metric_col,
        x_label_short=x_label_short,
        y_label_short=bologna_metric_short,
        panel_label="(d)",
        x_ascending=x_ascending,
        y_ascending=bologna_ascending,
    )

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [output_pdf, output_png]


def save_plots(plot_df: pd.DataFrame) -> list[Path]:
    written: list[Path] = []

    # 1) Item-QWK vs EL-QWK
    written.extend(
        save_comparison_figure(
            plot_df=plot_df,
            x_col="item_qwk",
            x_label="Item-level QWK",
            x_label_short="Item-QWK",
            x_ascending=False,
            linear_metric_col="el_qwk_linear",
            linear_metric_label="EL-QWK Linear Abs",
            linear_metric_short="EL-QWK",
            linear_ascending=False,
            bologna_metric_col="el_qwk_bologna",
            bologna_metric_label="EL-QWK Bologna",
            bologna_metric_short="EL-QWK",
            bologna_ascending=False,
            output_pdf=FIG_QWK_VS_ELQWK_PDF,
            output_png=FIG_QWK_VS_ELQWK_PNG,
        )
    )

    # 2) Item-MSE vs EL-QWK
    written.extend(
        save_comparison_figure(
            plot_df=plot_df,
            x_col="item_mse",
            x_label="Item-level MSE",
            x_label_short="Item-MSE",
            x_ascending=True,
            linear_metric_col="el_qwk_linear",
            linear_metric_label="EL-QWK Linear Abs",
            linear_metric_short="EL-QWK",
            linear_ascending=False,
            bologna_metric_col="el_qwk_bologna",
            bologna_metric_label="EL-QWK Bologna",
            bologna_metric_short="EL-QWK",
            bologna_ascending=False,
            output_pdf=FIG_MSE_VS_ELQWK_PDF,
            output_png=FIG_MSE_VS_ELQWK_PNG,
        )
    )

    # 3) Item-MSE vs EL-Acc
    written.extend(
        save_comparison_figure(
            plot_df=plot_df,
            x_col="item_mse",
            x_label="Item-level MSE",
            x_label_short="Item-MSE",
            x_ascending=True,
            linear_metric_col="el_acc_linear",
            linear_metric_label="EL-Acc Linear Abs",
            linear_metric_short="EL-Acc",
            linear_ascending=False,
            bologna_metric_col="el_acc_bologna",
            bologna_metric_label="EL-Acc Bologna",
            bologna_metric_short="EL-Acc",
            bologna_ascending=False,
            output_pdf=FIG_MSE_VS_ELACC_PDF,
            output_png=FIG_MSE_VS_ELACC_PNG,
        )
    )

    return written


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {INPUT_PARQUET.resolve()}"
        )

    if not EXAM_METRICS_PARQUET.exists():
        raise FileNotFoundError(
            f"Exam metrics parquet not found: {EXAM_METRICS_PARQUET.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for stale_output in STALE_OUTPUTS:
        if stale_output.exists():
            try:
                stale_output.unlink()
            except PermissionError:
                print(
                    f"Warning: could not remove locked stale output: {stale_output.resolve()}"
                )

    print("=" * 100)
    print("CREATE FIGURE 2: ITEM-LEVEL VS EXAM-LEVEL METRICS")
    print("=" * 100)
    print(f"Item input parquet:  {INPUT_PARQUET.resolve()}")
    print(f"Exam metrics parquet:{EXAM_METRICS_PARQUET.resolve()}")
    print(f"Output folder:       {OUTPUT_DIR.resolve()}")
    print("")

    df = read_parquet(INPUT_PARQUET)
    item_df, item_stats = build_gold_item_df_from_original_input(df)
    item_metrics_df = compute_item_metrics(item_df)

    exam_df = read_parquet(EXAM_METRICS_PARQUET)
    exam_summary_df, exam_stats = compute_exam_summary(exam_df)

    plot_df = merge_item_and_exam_metrics(item_metrics_df, exam_summary_df)

    if plot_df.empty:
        raise ValueError("No model metrics available for plotting.")

    plot_df.to_csv(DATA_CSV, index=False, encoding="utf-8")

    write_sanity_check(
        plot_df=plot_df,
        item_stats=item_stats,
        exam_stats=exam_stats,
    )

    written_plot_paths = save_plots(plot_df)

    print("Saved:")
    for path in written_plot_paths:
        print(f"  {path.resolve()}")
    print(f"  {DATA_CSV.resolve()}")
    print(f"  {SANITY_CHECK_TXT.resolve()}")


if __name__ == "__main__":
    main()