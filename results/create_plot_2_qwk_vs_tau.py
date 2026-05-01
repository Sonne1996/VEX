#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Figure 2 style item-level plots for VEX.

Input:
    dataset/additional/vex_metric_dataset/merged_model_predictions.parquet

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

Metrics:
    - item_mse: mean squared error over gold item labels
    - item_qwk: quadratic weighted kappa over gold item labels
    - item_tau_b: Kendall's tau-b over gold item labels

Outputs:
    results/figures_plot_2/figure_2_qwk_vs_tau.pdf
    results/figures_plot_2/figure_2_qwk_vs_tau.png
    results/figures_plot_2/figure_2_scatter_item_qwk_vs_item_tau.pdf
    results/figures_plot_2/figure_2_rank_shift_item_qwk_vs_item_tau.pdf
    results/figures_plot_2/figure_2_qwk_vs_tau_data.csv
    results/figures_plot_2/figure_2_sanity_check.txt
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


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
VEX_METRIC_DIR = PROJECT_ROOT / "vex_metric"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VEX_METRIC_DIR))

import vex_config as cfg


# =========================================================
# CONFIG
# =========================================================

INPUT_PARQUET = Path(cfg.INPUT_PARQUET)
OUTPUT_DIR = SCRIPT_PATH.parent / "figures_plot_2"

MODEL_COLUMNS = list(cfg.MODEL_COLUMNS)

QUESTION_ID_COL = "question_id"
STUDENT_ID_COL = "member_id"
ANSWER_ID_COL = "answer_id"
INPUT_HUMAN_GRADE_COL = "grade"
HUMAN_GRADE_COL = "human_grade"
LABEL_TYPE_COL = "label_type"
GOLD_LABEL_VALUE = "gold"

DATA_CSV = OUTPUT_DIR / "figure_2_qwk_vs_tau_data.csv"
SANITY_CHECK_TXT = OUTPUT_DIR / "figure_2_sanity_check.txt"
COMBINED_PDF = OUTPUT_DIR / "figure_2_qwk_vs_tau.pdf"
COMBINED_PNG = OUTPUT_DIR / "figure_2_qwk_vs_tau.png"
SCATTER_PDF = OUTPUT_DIR / "figure_2_scatter_item_qwk_vs_item_tau.pdf"
SCATTER_PNG = OUTPUT_DIR / "figure_2_scatter_item_qwk_vs_item_tau.png"
RANK_SHIFT_PDF = OUTPUT_DIR / "figure_2_rank_shift_item_qwk_vs_item_tau.pdf"
RANK_SHIFT_PNG = OUTPUT_DIR / "figure_2_rank_shift_item_qwk_vs_item_tau.png"

STALE_OUTPUTS = [
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
}


FAMILY_MARKERS = {
    "LLM": "o",
    "Transformer": "s",
    "Prior": "D",
    "TF-IDF": "^",
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

    metrics_df["qwk_rank"] = metrics_df["item_qwk"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    metrics_df["tau_rank"] = metrics_df["item_tau_b"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    metrics_df["mse_rank"] = metrics_df["item_mse"].rank(
        ascending=True,
        method="min",
        na_option="bottom",
    )

    return metrics_df.sort_values(
        ["family", "qwk_rank", "tau_rank", "model"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def write_sanity_check(
    plot_df: pd.DataFrame,
    item_stats: dict[str, Any],
) -> None:
    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("FIGURE 2 SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Input parquet: {INPUT_PARQUET.resolve()}")
    lines.append(
        "Item dataframe logic: evaluate_dataframe.py::_build_item_df_from_original_input, "
        "plus label_type == 'gold' filter when available."
    )
    lines.append("")
    lines.append("Input/filter counts:")
    for key, value in item_stats.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Metrics:")
    lines.append("  item_mse  = mean((human_grade - model_prediction)^2) over gold items")
    lines.append("  item_qwk  = quadratic weighted kappa over gold items")
    lines.append("  item_tau_b = Kendall's tau-b over gold items")
    lines.append("")
    lines.append("Per-model values used in the plot:")

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
        "mse_rank",
        "qwk_rank",
        "tau_rank",
    ]

    table = plot_df[table_cols].copy()
    for col in ["item_mse", "item_qwk", "item_tau_b"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(6)

    lines.append(table.to_string(index=False))
    lines.append("")
    lines.append("Sorted by item_qwk descending:")
    by_qwk = table.sort_values(["qwk_rank", "tau_rank", "model"])
    lines.append(by_qwk.to_string(index=False))
    lines.append("")
    lines.append("Sorted by item_tau_b descending:")
    by_tau = table.sort_values(["tau_rank", "qwk_rank", "model"])
    lines.append(by_tau.to_string(index=False))

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


def plot_scatter(plot_df: pd.DataFrame, ax: plt.Axes) -> None:
    scatter_df = plot_df.dropna(subset=["item_qwk", "item_tau_b"]).copy()

    for family, family_df in scatter_df.groupby("family", sort=False):
        ax.scatter(
            family_df["item_qwk"],
            family_df["item_tau_b"],
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
            xy=(row["item_qwk"], row["item_tau_b"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=7,
        )

    x_min, x_max = finite_limits(scatter_df["item_qwk"])
    y_min, y_max = finite_limits(scatter_df["item_tau_b"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Item-level QWK")
    ax.set_ylabel("Item-level Kendall's tau-b")
    ax.set_title("(a) Item-Level QWK vs. Item-Level Tau")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_slope_chart(plot_df: pd.DataFrame, ax: plt.Axes) -> None:
    ranked = plot_df.sort_values("qwk_rank", ascending=True).copy()

    qwk_order = plot_df.sort_values(
        ["item_qwk", "model"],
        ascending=[False, True],
        na_position="last",
    )
    tau_order = plot_df.sort_values(
        ["item_tau_b", "model"],
        ascending=[False, True],
        na_position="last",
    )
    qwk_plot_rank = {
        model_col: rank
        for rank, model_col in enumerate(qwk_order["model_col"], start=1)
    }
    tau_plot_rank = {
        model_col: rank
        for rank, model_col in enumerate(tau_order["model_col"], start=1)
    }
    ranked["qwk_plot_rank"] = ranked["model_col"].map(qwk_plot_rank)
    ranked["tau_plot_rank"] = ranked["model_col"].map(tau_plot_rank)

    max_rank = len(ranked)

    left_x = 0.25
    right_x = 0.4

    ax.set_xlim(0.05, 0.62)
    ax.set_ylim(max_rank + 0.75, 0.25)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels(["Item-QWK", "Item-tau-b"])
    ax.set_yticks(np.arange(1, max_rank + 1, 2))
    ax.set_ylabel("Model rank (1 = best)")
    ax.set_title("(b) Rank Shift Between Item Metrics")

    for _, row in ranked.iterrows():
        color = FAMILY_COLORS.get(row["family"], "#666666")
        ax.plot(
            [left_x, right_x],
            [row["qwk_plot_rank"], row["tau_plot_rank"]],
            color=color,
            linewidth=1.3,
            alpha=0.8,
        )
        ax.scatter(
            [left_x, right_x],
            [row["qwk_plot_rank"], row["tau_plot_rank"]],
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=35,
            zorder=3,
        )
        ax.text(
            left_x - 0.03,
            row["qwk_plot_rank"],
            str(row["model"]),
            ha="right",
            va="center",
            fontsize=7,
        )
        ax.text(
            right_x + 0.03,
            row["tau_plot_rank"],
            str(row["model"]),
            ha="left",
            va="center",
            fontsize=7,
        )

    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def save_plots(plot_df: pd.DataFrame) -> list[Path]:
    written: list[Path] = []

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10.4, 5.6),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
    )
    plot_scatter(plot_df, axes[0])
    plot_slope_chart(plot_df, axes[1])
    fig.tight_layout(w_pad=2.0)
    fig.savefig(COMBINED_PDF, bbox_inches="tight")
    fig.savefig(COMBINED_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.extend([COMBINED_PDF, COMBINED_PNG])

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    plot_scatter(plot_df, ax)
    fig.tight_layout()
    fig.savefig(SCATTER_PDF, bbox_inches="tight")
    fig.savefig(SCATTER_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.extend([SCATTER_PDF, SCATTER_PNG])

    fig, ax = plt.subplots(figsize=(4.9, 5.6))
    plot_slope_chart(plot_df, ax)
    fig.tight_layout()
    fig.savefig(RANK_SHIFT_PDF, bbox_inches="tight")
    fig.savefig(RANK_SHIFT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    written.extend([RANK_SHIFT_PDF, RANK_SHIFT_PNG])

    return written


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {INPUT_PARQUET.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_output in STALE_OUTPUTS:
        if stale_output.exists():
            stale_output.unlink()

    print("=" * 100)
    print("CREATE FIGURE 2: ITEM-QWK VS ITEM-LEVEL TAU")
    print("=" * 100)
    print(f"Input parquet: {INPUT_PARQUET.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("")

    df = read_parquet(INPUT_PARQUET)
    item_df, item_stats = build_gold_item_df_from_original_input(df)
    plot_df = compute_item_metrics(item_df)

    if plot_df.empty:
        raise ValueError("No model metrics available for plotting.")

    plot_df.to_csv(DATA_CSV, index=False, encoding="utf-8")
    write_sanity_check(plot_df=plot_df, item_stats=item_stats)
    written_plot_paths = save_plots(plot_df)

    print("Saved:")
    for path in written_plot_paths:
        print(f"  {path.resolve()}")
    print(f"  {DATA_CSV.resolve()}")
    print(f"  {SANITY_CHECK_TXT.resolve()}")


if __name__ == "__main__":
    main()
