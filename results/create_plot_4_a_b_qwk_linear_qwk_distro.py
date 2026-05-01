#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Figure 4: EL-QWK under absolute linear vs. distribution-based grading.

Input:
    vex_metric/vex_test_env/2_metrics/exam_level_precomputed_metrics.parquet

Figure:
    (a) Absolute Linear Scale
        Uses el_qwk_linear_abs. This is the Swiss-style absolute scale:
        normalized score -> grade = 1 + 5 * score, rounded in the metric code.

    (b) Distribution-Based Scale
        Uses el_qwk_bologna. This is the Bologna/rank-based scale with the
        configured passing distribution.

Each point is one model. The two panels use the same model order, sorted by
absolute-linear EL-QWK, so differences between panels show whether degradation
is more likely caused by poor absolute calibration or poor relative ranking.

Outputs:
    results/figures_plot_4/figure_4_el_qwk_linear_vs_bologna.pdf
    results/figures_plot_4/figure_4_el_qwk_linear_vs_bologna.png
    results/figures_plot_4/el_qwk_linear.pdf
    results/figures_plot_4/el_qwk_distribution.pdf
    results/figures_plot_4/figure_4_el_qwk_linear_vs_bologna_data.csv
    results/figures_plot_4/figure_4_sanity_check.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

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


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
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
OUTPUT_DIR = SCRIPT_PATH.parent / "figures_plot_4"

MODEL_COLUMNS = list(cfg.MODEL_COLUMNS)

MODEL_COL = "model_col"
TEST_ID_COL = "test_id"
TEST_SIZE_COL = "test_size"

LINEAR_COL = "el_qwk_linear_abs"
BOLOGNA_COL = "el_qwk_bologna"

COMBINED_PDF = OUTPUT_DIR / "figure_4_el_qwk_linear_vs_bologna.pdf"
COMBINED_PNG = OUTPUT_DIR / "figure_4_el_qwk_linear_vs_bologna.png"
LINEAR_PDF = OUTPUT_DIR / "el_qwk_linear.pdf"
LINEAR_PNG = OUTPUT_DIR / "el_qwk_linear.png"
DISTRIBUTION_PDF = OUTPUT_DIR / "el_qwk_distribution.pdf"
DISTRIBUTION_PNG = OUTPUT_DIR / "el_qwk_distribution.png"
DATA_CSV = OUTPUT_DIR / "figure_4_el_qwk_linear_vs_bologna_data.csv"
BY_SIZE_CSV = OUTPUT_DIR / "figure_4_el_qwk_linear_vs_bologna_by_test_size.csv"
SANITY_CHECK_TXT = OUTPUT_DIR / "figure_4_sanity_check.txt"

STALE_OUTPUTS = [
    OUTPUT_DIR / "figure_4_el_qwk_granularity.pdf",
    OUTPUT_DIR / "figure_4_el_qwk_granularity.png",
    OUTPUT_DIR / "figure_4_el_qwk_granularity_data.csv",
    OUTPUT_DIR / "figure_4_el_qwk_granularity_model_summary.csv",
    OUTPUT_DIR / "figure_4_el_qwk_granularity_per_exam.csv",
]

FAMILY_COLORS = {
    "LLM": "#1f77b4",
    "Transformer": "#2ca02c",
    "Prior": "#9467bd",
    "TF-IDF": "#ff7f0e",
}


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
# IO AND SUMMARY
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
        LINEAR_COL,
        BOLOGNA_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in exam-level metrics parquet: {missing}")

    missing_models = sorted(set(MODEL_COLUMNS) - set(df[MODEL_COL].dropna().unique()))
    if missing_models:
        raise ValueError(f"Missing model rows in exam-level metrics parquet: {missing_models}")


def compute_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df[LINEAR_COL] = pd.to_numeric(work_df[LINEAR_COL], errors="coerce")
    work_df[BOLOGNA_COL] = pd.to_numeric(work_df[BOLOGNA_COL], errors="coerce")

    summary = (
        work_df.groupby(MODEL_COL, sort=False)
        .agg(
            exam_rows=(TEST_ID_COL, "size"),
            virtual_exams=(TEST_ID_COL, "nunique"),
            test_sizes=(TEST_SIZE_COL, lambda s: ",".join(map(str, sorted(set(s))))),
            n_students_mean=("n_students", "mean"),
            linear_qwk_mean=(LINEAR_COL, "mean"),
            linear_qwk_std=(LINEAR_COL, "std"),
            linear_missing=(LINEAR_COL, lambda s: int(s.isna().sum())),
            bologna_qwk_mean=(BOLOGNA_COL, "mean"),
            bologna_qwk_std=(BOLOGNA_COL, "std"),
            bologna_missing=(BOLOGNA_COL, lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    summary["model"] = summary[MODEL_COL].map(display_name)
    summary["family"] = summary[MODEL_COL].map(model_family)
    summary["bologna_minus_linear"] = (
        summary["bologna_qwk_mean"] - summary["linear_qwk_mean"]
    )

    return summary.sort_values(
        ["linear_qwk_mean", "bologna_qwk_mean", "model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def compute_by_test_size(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df[LINEAR_COL] = pd.to_numeric(work_df[LINEAR_COL], errors="coerce")
    work_df[BOLOGNA_COL] = pd.to_numeric(work_df[BOLOGNA_COL], errors="coerce")

    by_size = (
        work_df.groupby([TEST_SIZE_COL, MODEL_COL], sort=True)
        .agg(
            exam_rows=(TEST_ID_COL, "size"),
            virtual_exams=(TEST_ID_COL, "nunique"),
            n_students_mean=("n_students", "mean"),
            linear_qwk_mean=(LINEAR_COL, "mean"),
            linear_qwk_std=(LINEAR_COL, "std"),
            bologna_qwk_mean=(BOLOGNA_COL, "mean"),
            bologna_qwk_std=(BOLOGNA_COL, "std"),
        )
        .reset_index()
    )
    by_size["model"] = by_size[MODEL_COL].map(display_name)
    by_size["family"] = by_size[MODEL_COL].map(model_family)
    by_size["bologna_minus_linear"] = (
        by_size["bologna_qwk_mean"] - by_size["linear_qwk_mean"]
    )
    return by_size


def write_sanity_check(
    raw_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    by_size: pd.DataFrame,
) -> None:
    present_sizes = sorted(
        pd.to_numeric(raw_df[TEST_SIZE_COL], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("FIGURE 4 SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Input parquet: {INPUT_PARQUET.resolve()}")
    lines.append(f"Raw rows: {len(raw_df)}")
    lines.append(f"Models: {raw_df[MODEL_COL].nunique()}")
    lines.append(f"Test sizes present: {present_sizes}")
    lines.append("")
    lines.append("Values used:")
    lines.append(f"  Left panel  (Absolute Linear Scale): {LINEAR_COL}")
    lines.append(f"  Right panel (Distribution-Based/Bologna Scale): {BOLOGNA_COL}")
    lines.append("")
    lines.append("Metric meaning:")
    lines.append("  el_qwk_linear_abs is computed by evaluate_dataframe.py on absolute")
    lines.append("  Swiss-style grades: grade = 1 + 5 * normalized exam score.")
    lines.append("  el_qwk_bologna is computed by evaluate_dataframe.py on Bologna labels")
    lines.append("  assigned by the configured passing distribution.")
    lines.append("")
    lines.append("Aggregation:")
    lines.append("  1. Use precomputed exam-level QWK rows per model/test_id/test_size.")
    lines.append("  2. Average over all virtual exam rows for each model.")
    lines.append("  3. Sort models by Absolute Linear EL-QWK and reuse that order in both panels.")
    lines.append("")
    lines.append("Raw row counts by test_size:")
    counts = (
        raw_df.groupby(TEST_SIZE_COL)
        .agg(
            rows=(TEST_ID_COL, "size"),
            virtual_exams=(TEST_ID_COL, "nunique"),
            models=(MODEL_COL, "nunique"),
            n_students_mean=("n_students", "mean"),
        )
        .reset_index()
    )
    lines.append(counts.to_string(index=False))
    lines.append("")
    lines.append("Per-model values used in both panels:")
    table = model_summary.copy()
    for col in [
        "n_students_mean",
        "linear_qwk_mean",
        "linear_qwk_std",
        "bologna_qwk_mean",
        "bologna_qwk_std",
        "bologna_minus_linear",
    ]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(6)
    lines.append(table.to_string(index=False))
    lines.append("")
    lines.append("Per-model values split by test_size:")
    by_size_table = by_size.copy()
    for col in [
        "n_students_mean",
        "linear_qwk_mean",
        "linear_qwk_std",
        "bologna_qwk_mean",
        "bologna_qwk_std",
        "bologna_minus_linear",
    ]:
        by_size_table[col] = pd.to_numeric(by_size_table[col], errors="coerce").round(6)

    for test_size, size_df in by_size_table.groupby(TEST_SIZE_COL, sort=True):
        lines.append("")
        lines.append(f"[test_size={int(test_size)}]")
        lines.append(
            size_df.sort_values("linear_qwk_mean", ascending=False).to_string(index=False)
        )

    SANITY_CHECK_TXT.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# PLOTTING
# =========================================================

def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)


def plot_panel(
    ax: plt.Axes,
    model_summary: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    title: str,
    show_y_labels: bool,
) -> None:
    plot_df = model_summary.copy()
    plot_df["y_pos"] = np.arange(len(plot_df))
    colors = [FAMILY_COLORS.get(family, "#666666") for family in plot_df["family"]]

    ax.errorbar(
        plot_df[metric_mean_col],
        plot_df["y_pos"],
        xerr=plot_df[metric_std_col].fillna(0.0),
        fmt="none",
        ecolor="#bdbdbd",
        elinewidth=0.8,
        capsize=2,
        alpha=0.8,
        zorder=1,
    )
    ax.scatter(
        plot_df[metric_mean_col],
        plot_df["y_pos"],
        s=48,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )

    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(len(plot_df) - 0.5, -0.5)
    ax.set_xlabel("Exam-level QWK")
    ax.set_title(title)

    if show_y_labels:
        ax.set_yticks(plot_df["y_pos"])
        ax.set_yticklabels(plot_df["model"])
    else:
        ax.set_yticks(plot_df["y_pos"])
        ax.tick_params(axis="y", labelleft=False, length=0)

    style_axes(ax)


def add_family_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=6,
            label=family,
        )
        for family, color in FAMILY_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
    )


def save_plots(model_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10.2, 6.0),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    plot_panel(
        ax=axes[0],
        model_summary=model_summary,
        metric_mean_col="linear_qwk_mean",
        metric_std_col="linear_qwk_std",
        title="(a) Absolute Linear Scale",
        show_y_labels=True,
    )
    plot_panel(
        ax=axes[1],
        model_summary=model_summary,
        metric_mean_col="bologna_qwk_mean",
        metric_std_col="bologna_qwk_std",
        title="(b) Distribution-Based Scale",
        show_y_labels=False,
    )
    add_family_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.8)
    fig.savefig(COMBINED_PDF, bbox_inches="tight")
    fig.savefig(COMBINED_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 6.0))
    plot_panel(
        ax=ax,
        model_summary=model_summary,
        metric_mean_col="linear_qwk_mean",
        metric_std_col="linear_qwk_std",
        title="(a) Absolute Linear Scale",
        show_y_labels=True,
    )
    fig.tight_layout()
    fig.savefig(LINEAR_PDF, bbox_inches="tight")
    fig.savefig(LINEAR_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 6.0))
    plot_panel(
        ax=ax,
        model_summary=model_summary,
        metric_mean_col="bologna_qwk_mean",
        metric_std_col="bologna_qwk_std",
        title="(b) Distribution-Based Scale",
        show_y_labels=True,
    )
    fig.tight_layout()
    fig.savefig(DISTRIBUTION_PDF, bbox_inches="tight")
    fig.savefig(DISTRIBUTION_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Exam-level metrics parquet not found: {INPUT_PARQUET.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in STALE_OUTPUTS:
        if stale_path.exists():
            try:
                stale_path.unlink()
            except PermissionError:
                print(f"Warning: could not remove locked stale file: {stale_path}")

    print("=" * 100)
    print("CREATE FIGURE 4: LINEAR VS BOLOGNA EL-QWK")
    print("=" * 100)
    print(f"Input parquet: {INPUT_PARQUET.resolve()}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("")

    raw_df = read_parquet(INPUT_PARQUET)
    validate_input(raw_df)

    model_summary = compute_model_summary(raw_df)
    by_size = compute_by_test_size(raw_df)

    model_summary.to_csv(DATA_CSV, index=False, encoding="utf-8")
    by_size.to_csv(BY_SIZE_CSV, index=False, encoding="utf-8")
    write_sanity_check(
        raw_df=raw_df,
        model_summary=model_summary,
        by_size=by_size,
    )
    save_plots(model_summary=model_summary)

    print("Saved:")
    for path in [
        COMBINED_PDF,
        COMBINED_PNG,
        LINEAR_PDF,
        LINEAR_PNG,
        DISTRIBUTION_PDF,
        DISTRIBUTION_PNG,
        DATA_CSV,
        BY_SIZE_CSV,
        SANITY_CHECK_TXT,
    ]:
        print(f"  {path.resolve()}")


if __name__ == "__main__":
    main()
