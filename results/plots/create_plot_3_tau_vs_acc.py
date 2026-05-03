#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Figure 3 style sensitivity plot for exam-level metrics.

Input:
    vex_metric/vex_test_env/2_metrics/exam_level_precomputed_metrics.parquet

Plot:
    One figure with two panels:
    - Absolute Linear scale: EL-tau-b, EL-Acc Linear Abs, EL-QWK Linear Abs
    - Distrobution scale: EL-tau-b, EL-Acc Distrobution, EL-QWK Distrobution

    The bold lines show the mean across models after first averaging each model
    over all virtual exams for a given N. Thin background lines show individual
    model trends.

Outputs:
    results/figures_plot_3/figure_3_tau_vs_acc.pdf
    results/figures_plot_3/figure_3_tau_vs_acc.png
    results/figures_plot_3/figure_3_tau_vs_acc_data.csv
    results/figures_plot_3/figure_3_sanity_check.txt
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


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
VEX_METRIC_DIR = PROJECT_ROOT / "vex_metric"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VEX_METRIC_DIR))

import vex_config as cfg
from vex_plot_metrics import load_or_compute_exam_metrics


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
# CONFIG
# =========================================================

INPUT_PARQUET = (
    Path(cfg.TEST_ENV_FOLDER)
    / cfg.TEST_ENV_METRICS_FOLDER
    / "exam_level_precomputed_metrics.parquet"
)
OUTPUT_DIR = SCRIPT_PATH.parent / "figures_plot_3"

MODEL_COLUMNS = list(cfg.MODEL_COLUMNS)

TEST_SIZE_COL = "test_size"
TEST_ID_COL = "test_id"
MODEL_COL = "model_col"

TAU_COL = "el_tau_b"
ACC_COL = "el_acc_linear_abs"
QWK_COL = "el_qwk_linear_abs"
LEGACY_DISTROBUTION_TOKEN = "bolo" + "gna"
BOL_ACC_COL = "el_acc_distrobution"
BOL_QWK_COL = "el_qwk_distrobution"

EXPECTED_FIGURE_TEST_SIZES = [5, 10, 15, 20, 21]

PLOT_PDF = OUTPUT_DIR / "figure_3_tau_vs_acc.pdf"
PLOT_PNG = OUTPUT_DIR / "figure_3_tau_vs_acc.png"
DATA_CSV = OUTPUT_DIR / "figure_3_tau_vs_acc_data.csv"
MODEL_SUMMARY_CSV = OUTPUT_DIR / "figure_3_tau_vs_acc_model_summary.csv"
SANITY_CHECK_TXT = OUTPUT_DIR / "figure_3_sanity_check.txt"

# =========================================================
# MANUAL PLOT LAYOUT OPTIONS
# =========================================================

# Use these values to move the legend ("Tabelle" in your screenshot) by hand.
#
# Matplotlib anchor quick guide:
#   loc="upper center", bbox_to_anchor=(0.5, -0.24)
#       -> centered below the panel.
#   loc="lower right", bbox_to_anchor=(0.99, 0.02)
#       -> inside the bottom-right corner.
#   loc="center left", bbox_to_anchor=(1.02, 0.5)
#       -> outside to the right of the panel.
#
# The coordinates in bbox_to_anchor are relative to the panel axes:
#   x=0 left, x=1 right, y=0 bottom, y=1 top.
# Negative y values move the legend below the plot.
FIGURE_SIZE = (11.2, 4.8)
FIGURE_LAYOUT_RECT = (0.0, 0.18, 1.0, 0.95)

LEGEND_LAYOUT = {
    "Absolute Linear": {
        "loc": "upper center",
        "bbox_to_anchor": (0.9, -0.24),
        "ncol": 1,
        "frameon": False,
    },
    "Distrobution": {
        "loc": "upper center",
        "bbox_to_anchor": (0.9, -0.24),
        "ncol": 1,
        "frameon": False,
    },
}

METRIC_SPECS = [
    {
        "scale": "Absolute Linear",
        "metric": "EL-tau-b",
        "summary_col": "el_tau_b_mean",
        "source_col": TAU_COL,
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "scale": "Absolute Linear",
        "metric": "EL-Acc Linear Abs",
        "summary_col": "el_acc_linear_abs_mean",
        "source_col": ACC_COL,
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
    },
    {
        "scale": "Absolute Linear",
        "metric": "EL-QWK Linear Abs",
        "summary_col": "el_qwk_linear_abs_mean",
        "source_col": QWK_COL,
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
    },
    {
        "scale": "Distrobution",
        "metric": "EL-tau-b",
        "summary_col": "el_tau_b_mean",
        "source_col": TAU_COL,
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "scale": "Distrobution",
        "metric": "EL-Acc Distrobution",
        "summary_col": "el_acc_distrobution_mean",
        "source_col": BOL_ACC_COL,
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
    },
    {
        "scale": "Distrobution",
        "metric": "EL-QWK Distrobution",
        "summary_col": "el_qwk_distrobution_mean",
        "source_col": BOL_QWK_COL,
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
    },
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
# IO AND METRICS
# =========================================================

def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Pandas cannot read the exam-level parquet file. Install pyarrow "
            "or fastparquet, e.g. `pip install pyarrow`."
        ) from exc


def validate_input(df: pd.DataFrame) -> None:
    required = [
        MODEL_COL,
        TEST_ID_COL,
        TEST_SIZE_COL,
        "n_students",
        "students_raw",
        "students_complete",
        TAU_COL,
        ACC_COL,
        QWK_COL,
        BOL_ACC_COL,
        BOL_QWK_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in exam-level metrics parquet: {missing}")

    missing_models = sorted(set(MODEL_COLUMNS) - set(df[MODEL_COL].dropna().unique()))
    if missing_models:
        raise ValueError(f"Missing model rows in exam-level metrics parquet: {missing_models}")


def normalize_input_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Accept older precomputed metric columns and the shared plot cache."""
    result = raw_df.copy()
    legacy_acc_col = f"el_acc_{LEGACY_DISTROBUTION_TOKEN}"
    legacy_qwk_col = f"el_qwk_{LEGACY_DISTROBUTION_TOKEN}"
    rename_map: dict[str, str] = {}

    if legacy_acc_col in result.columns and BOL_ACC_COL not in result.columns:
        rename_map[legacy_acc_col] = BOL_ACC_COL
    if legacy_qwk_col in result.columns and BOL_QWK_COL not in result.columns:
        rename_map[legacy_qwk_col] = BOL_QWK_COL

    if rename_map:
        result = result.rename(columns=rename_map)

    if "students_complete" not in result.columns and "n_students" in result.columns:
        result["students_complete"] = result["n_students"]

    return result


def compute_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df[TEST_SIZE_COL] = pd.to_numeric(work_df[TEST_SIZE_COL], errors="coerce")
    work_df[TAU_COL] = pd.to_numeric(work_df[TAU_COL], errors="coerce")
    work_df[ACC_COL] = pd.to_numeric(work_df[ACC_COL], errors="coerce")
    work_df[QWK_COL] = pd.to_numeric(work_df[QWK_COL], errors="coerce")
    work_df[BOL_ACC_COL] = pd.to_numeric(work_df[BOL_ACC_COL], errors="coerce")
    work_df[BOL_QWK_COL] = pd.to_numeric(work_df[BOL_QWK_COL], errors="coerce")

    summary = (
        work_df.groupby([TEST_SIZE_COL, MODEL_COL], sort=True)
        .agg(
            model_virtual_exams=(TEST_ID_COL, "nunique"),
            rows=(TEST_ID_COL, "size"),
            n_students_mean=("n_students", "mean"),
            students_raw_sum=("students_raw", "sum"),
            students_complete_sum=("students_complete", "sum"),
            el_tau_b_mean=(TAU_COL, "mean"),
            el_tau_b_std=(TAU_COL, "std"),
            el_tau_b_missing=(TAU_COL, lambda s: int(s.isna().sum())),
            el_acc_linear_abs_mean=(ACC_COL, "mean"),
            el_acc_linear_abs_std=(ACC_COL, "std"),
            el_acc_linear_abs_missing=(ACC_COL, lambda s: int(s.isna().sum())),
            el_qwk_linear_abs_mean=(QWK_COL, "mean"),
            el_qwk_linear_abs_std=(QWK_COL, "std"),
            el_qwk_linear_abs_missing=(QWK_COL, lambda s: int(s.isna().sum())),
            el_acc_distrobution_mean=(BOL_ACC_COL, "mean"),
            el_acc_distrobution_std=(BOL_ACC_COL, "std"),
            el_acc_distrobution_missing=(BOL_ACC_COL, lambda s: int(s.isna().sum())),
            el_qwk_distrobution_mean=(BOL_QWK_COL, "mean"),
            el_qwk_distrobution_std=(BOL_QWK_COL, "std"),
            el_qwk_distrobution_missing=(BOL_QWK_COL, lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    summary[TEST_SIZE_COL] = summary[TEST_SIZE_COL].astype(int)
    summary["model"] = summary[MODEL_COL].map(display_name)
    summary["family"] = summary[MODEL_COL].map(model_family)

    return summary.sort_values(
        [TEST_SIZE_COL, "family", "el_tau_b_mean", "model"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def compute_plot_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for test_size, size_df in model_summary.groupby(TEST_SIZE_COL, sort=True):
        for spec in METRIC_SPECS:
            metric_col = str(spec["summary_col"])
            values = pd.to_numeric(size_df[metric_col], errors="coerce").dropna()
            rows.append(
                {
                    TEST_SIZE_COL: int(test_size),
                    "scale": spec["scale"],
                    "metric": spec["metric"],
                    "metric_col": metric_col,
                    "model_mean": float(values.mean()) if not values.empty else np.nan,
                    "model_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "models_with_value": int(values.count()),
                    "models_missing": int(len(size_df) - values.count()),
                }
            )

    return pd.DataFrame(rows)


def write_sanity_check(
    raw_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    plot_summary: pd.DataFrame,
) -> None:
    present_sizes = sorted(
        pd.to_numeric(raw_df[TEST_SIZE_COL], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    missing_expected_sizes = [
        size for size in EXPECTED_FIGURE_TEST_SIZES if size not in present_sizes
    ]

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("FIGURE 3 SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Input parquet: {INPUT_PARQUET.resolve()}")
    lines.append(f"Raw rows: {len(raw_df)}")
    lines.append(f"Models: {raw_df[MODEL_COL].nunique()}")
    lines.append(f"Test sizes present: {present_sizes}")
    lines.append(f"Caption test sizes expected: {EXPECTED_FIGURE_TEST_SIZES}")
    lines.append(f"Caption test sizes missing from data: {missing_expected_sizes}")
    lines.append("")
    lines.append("Metric definitions:")
    lines.append("  EL-tau-b = el_tau_b from evaluate_dataframe.py")
    lines.append("  EL-Acc Linear Abs = el_acc_linear_abs from evaluate_dataframe.py")
    lines.append("  EL-QWK Linear Abs = el_qwk_linear_abs from evaluate_dataframe.py")
    lines.append("  EL-Acc Distrobution = el_acc_distrobution from evaluate_dataframe.py")
    lines.append("  EL-QWK Distrobution = el_qwk_distrobution from evaluate_dataframe.py")
    lines.append("")
    lines.append("Aggregation used for plot:")
    lines.append("  1. Average each model over all virtual exams for each test_size.")
    lines.append("  2. Average those model-level means across models for the bold curves.")
    lines.append("  3. Thin background lines show individual model-level means.")
    lines.append("  4. Tau is the same exam-level ranking metric in both scale panels.")
    lines.append("  5. Each panel y-axis is zoomed to its plotted mean curves.")
    lines.append("")
    lines.append("Raw row counts by test_size:")
    counts = (
        raw_df.groupby(TEST_SIZE_COL)
        .agg(
            rows=(TEST_ID_COL, "size"),
            virtual_exams=(TEST_ID_COL, "nunique"),
            models=(MODEL_COL, "nunique"),
            n_students_mean=("n_students", "mean"),
            students_raw_sum=("students_raw", "sum"),
            students_complete_sum=("students_complete", "sum"),
        )
        .reset_index()
    )
    lines.append(counts.to_string(index=False))
    lines.append("")
    lines.append("Bold-curve values used in plot:")
    plot_table = plot_summary.copy()
    for col in ["model_mean", "model_std"]:
        plot_table[col] = pd.to_numeric(plot_table[col], errors="coerce").round(6)
    lines.append(plot_table.to_string(index=False))
    lines.append("")
    lines.append("Per-model values used before cross-model aggregation:")
    table_cols = [
        TEST_SIZE_COL,
        MODEL_COL,
        "model",
        "family",
        "model_virtual_exams",
        "n_students_mean",
        "el_tau_b_mean",
        "el_tau_b_std",
        "el_tau_b_missing",
        "el_acc_linear_abs_mean",
        "el_acc_linear_abs_std",
        "el_acc_linear_abs_missing",
        "el_qwk_linear_abs_mean",
        "el_qwk_linear_abs_std",
        "el_qwk_linear_abs_missing",
        "el_acc_distrobution_mean",
        "el_acc_distrobution_std",
        "el_acc_distrobution_missing",
        "el_qwk_distrobution_mean",
        "el_qwk_distrobution_std",
        "el_qwk_distrobution_missing",
    ]
    model_table = model_summary[table_cols].copy()
    for col in [
        "n_students_mean",
        "el_tau_b_mean",
        "el_tau_b_std",
        "el_acc_linear_abs_mean",
        "el_acc_linear_abs_std",
        "el_qwk_linear_abs_mean",
        "el_qwk_linear_abs_std",
        "el_acc_distrobution_mean",
        "el_acc_distrobution_std",
        "el_qwk_distrobution_mean",
        "el_qwk_distrobution_std",
    ]:
        model_table[col] = pd.to_numeric(model_table[col], errors="coerce").round(6)

    for test_size, size_df in model_table.groupby(TEST_SIZE_COL, sort=True):
        lines.append("")
        lines.append(f"[test_size={int(test_size)}]")
        lines.append(size_df.to_string(index=False))

    if len(present_sizes) >= 2:
        first_size = present_sizes[0]
        last_size = present_sizes[-1]
        pivot = model_summary.pivot(
            index=[MODEL_COL, "model", "family"],
            columns=TEST_SIZE_COL,
            values=[
                "el_tau_b_mean",
                "el_acc_linear_abs_mean",
                "el_qwk_linear_abs_mean",
                "el_acc_distrobution_mean",
                "el_qwk_distrobution_mean",
            ],
        )
        deltas: list[dict[str, Any]] = []
        for idx, row in pivot.iterrows():
            model_col, model, family = idx
            tau_delta = np.nan
            acc_delta = np.nan
            if ("el_tau_b_mean", first_size) in pivot.columns and (
                "el_tau_b_mean",
                last_size,
            ) in pivot.columns:
                tau_delta = row[("el_tau_b_mean", last_size)] - row[
                    ("el_tau_b_mean", first_size)
                ]
            if ("el_acc_linear_abs_mean", first_size) in pivot.columns and (
                "el_acc_linear_abs_mean",
                last_size,
            ) in pivot.columns:
                acc_delta = row[("el_acc_linear_abs_mean", last_size)] - row[
                    ("el_acc_linear_abs_mean", first_size)
                ]
            qwk_delta = np.nan
            if ("el_qwk_linear_abs_mean", first_size) in pivot.columns and (
                "el_qwk_linear_abs_mean",
                last_size,
            ) in pivot.columns:
                qwk_delta = row[("el_qwk_linear_abs_mean", last_size)] - row[
                    ("el_qwk_linear_abs_mean", first_size)
                ]
            bol_acc_delta = np.nan
            if ("el_acc_distrobution_mean", first_size) in pivot.columns and (
                "el_acc_distrobution_mean",
                last_size,
            ) in pivot.columns:
                bol_acc_delta = row[("el_acc_distrobution_mean", last_size)] - row[
                    ("el_acc_distrobution_mean", first_size)
                ]
            bol_qwk_delta = np.nan
            if ("el_qwk_distrobution_mean", first_size) in pivot.columns and (
                "el_qwk_distrobution_mean",
                last_size,
            ) in pivot.columns:
                bol_qwk_delta = row[("el_qwk_distrobution_mean", last_size)] - row[
                    ("el_qwk_distrobution_mean", first_size)
                ]
            deltas.append(
                {
                    MODEL_COL: model_col,
                    "model": model,
                    "family": family,
                    f"delta_tau_{first_size}_to_{last_size}": tau_delta,
                    f"delta_acc_abs_{first_size}_to_{last_size}": acc_delta,
                    f"delta_qwk_abs_{first_size}_to_{last_size}": qwk_delta,
                    f"delta_acc_distrobution_{first_size}_to_{last_size}": bol_acc_delta,
                    f"delta_qwk_distrobution_{first_size}_to_{last_size}": bol_qwk_delta,
                }
            )

        delta_df = pd.DataFrame(deltas)
        for col in delta_df.columns:
            if col.startswith("delta_"):
                delta_df[col] = pd.to_numeric(delta_df[col], errors="coerce").round(6)
        lines.append("")
        lines.append(f"Per-model deltas from N={first_size} to N={last_size}:")
        lines.append(delta_df.to_string(index=False))

    SANITY_CHECK_TXT.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# PLOTTING
# =========================================================

def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)


def plot_background_model_lines(
    ax: plt.Axes,
    model_summary: pd.DataFrame,
    spec: dict[str, Any],
) -> None:
    for _model_col, model_df in model_summary.groupby(MODEL_COL, sort=False):
        model_df = model_df.sort_values(TEST_SIZE_COL)
        values = pd.to_numeric(model_df[str(spec["summary_col"])], errors="coerce")
        if values.notna().sum() < 2:
            continue
        ax.plot(
            model_df[TEST_SIZE_COL],
            values,
            color=str(spec["color"]),
            linestyle=str(spec["linestyle"]),
            linewidth=0.8,
            alpha=0.14,
            zorder=1,
        )


def plot_metric_curve(
    ax: plt.Axes,
    plot_summary: pd.DataFrame,
    spec: dict[str, Any],
) -> None:
    metric_df = plot_summary[
        (plot_summary["scale"] == spec["scale"])
        & (plot_summary["metric"] == spec["metric"])
    ].sort_values(TEST_SIZE_COL)

    ax.plot(
        metric_df[TEST_SIZE_COL],
        metric_df["model_mean"],
        color=str(spec["color"]),
        marker=str(spec["marker"]),
        linestyle=str(spec["linestyle"]),
        linewidth=2.0,
        markersize=6,
        label=str(spec["metric"]),
        zorder=3,
    )


def zoomed_y_limits(plot_summary: pd.DataFrame, scale: str) -> tuple[float, float]:
    values = pd.to_numeric(
        plot_summary.loc[plot_summary["scale"] == scale, "model_mean"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return 0.0, 1.0

    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    if span == 0:
        span = max(abs(upper), 1.0) * 0.1

    pad = max(span * 0.18, 0.015)
    return max(0.0, lower - pad), min(1.0, upper + pad)


def plot_scale_panel(
    ax: plt.Axes,
    model_summary: pd.DataFrame,
    plot_summary: pd.DataFrame,
    scale: str,
    title: str,
) -> None:
    specs = [spec for spec in METRIC_SPECS if spec["scale"] == scale]
    present_sizes = sorted(model_summary[TEST_SIZE_COL].dropna().astype(int).unique())

    for spec in specs:
        plot_background_model_lines(
            ax=ax,
            model_summary=model_summary,
            spec=spec,
        )

    for spec in specs:
        plot_metric_curve(
            ax=ax,
            plot_summary=plot_summary,
            spec=spec,
        )

    ax.set_xticks(present_sizes)
    ax.set_xlabel("Virtual exam size N")
    ax.set_ylabel("Exam-level performance")
    ax.set_title(title)
    ax.set_ylim(*zoomed_y_limits(plot_summary, scale))
    style_axes(ax)
    legend_layout = LEGEND_LAYOUT.get(scale, {})
    ax.legend(
        frameon=bool(legend_layout.get("frameon", False)),
        loc=str(legend_layout.get("loc", "upper center")),
        bbox_to_anchor=legend_layout.get("bbox_to_anchor", (0.5, -0.24)),
        ncol=int(legend_layout.get("ncol", 1)),
        borderaxespad=0.0,
    )


def save_plot(model_summary: pd.DataFrame, plot_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGURE_SIZE, sharey=False)

    plot_scale_panel(
        ax=axes[0],
        model_summary=model_summary,
        plot_summary=plot_summary,
        scale="Absolute Linear",
        title="(a) Absolute Linear Scale",
    )
    plot_scale_panel(
        ax=axes[1],
        model_summary=model_summary,
        plot_summary=plot_summary,
        scale="Distrobution",
        title="(b) Distrobution Scale",
    )

    fig.suptitle(
        "Sensitivity of Exam-Level Performance to Virtual Exam Size",
        fontsize=12,
        y=1.02,
    )

    fig.tight_layout(rect=FIGURE_LAYOUT_RECT)
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CREATE FIGURE 3: EXAM-LEVEL TAU/ACC VS VIRTUAL EXAM SIZE")
    print("=" * 100)
    print(f"Input parquet/cache: {INPUT_PARQUET.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("")

    if INPUT_PARQUET.exists():
        raw_df = read_parquet(INPUT_PARQUET)
    else:
        print("Exam-level parquet not found; using shared plot metric cache.")
        raw_df = load_or_compute_exam_metrics()
    raw_df = normalize_input_columns(raw_df)
    validate_input(raw_df)

    model_summary = compute_model_summary(raw_df)
    plot_summary = compute_plot_summary(model_summary)

    model_summary.to_csv(MODEL_SUMMARY_CSV, index=False, encoding="utf-8")
    plot_summary.to_csv(DATA_CSV, index=False, encoding="utf-8")
    write_sanity_check(
        raw_df=raw_df,
        model_summary=model_summary,
        plot_summary=plot_summary,
    )
    save_plot(model_summary=model_summary, plot_summary=plot_summary)

    print("Saved:")
    print(f"  {PLOT_PDF.resolve()}")
    print(f"  {PLOT_PNG.resolve()}")
    print(f"  {DATA_CSV.resolve()}")
    print(f"  {MODEL_SUMMARY_CSV.resolve()}")
    print(f"  {SANITY_CHECK_TXT.resolve()}")


if __name__ == "__main__":
    main()
