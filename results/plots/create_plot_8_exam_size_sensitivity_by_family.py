#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create an exam-size sensitivity plot by model family.

Input:
    vex_metric/vex_test_env/4_dataframe/df_env_q*.parquet

Outputs:
    results/plots/figures_plot_8_exam_size_by_family/
        figure_8_exam_size_sensitivity_by_family.pdf
        figure_8_exam_size_sensitivity_by_family.png
        figure_8_exam_size_sensitivity_by_family_data.csv
        figure_8_exam_size_sensitivity_by_family_sanity_check.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vex_plot_metrics import (
    EXAM_METRICS_CACHE,
    FAMILY_COLORS,
    PROJECT_ROOT,
    load_or_compute_exam_metrics,
    summarize_exam_metrics,
    write_lines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_8_exam_size_by_family"

PLOT_PDF = OUTPUT_DIR / "figure_8_exam_size_sensitivity_by_family.pdf"
PLOT_PNG = OUTPUT_DIR / "figure_8_exam_size_sensitivity_by_family.png"
DATA_CSV = OUTPUT_DIR / "figure_8_exam_size_sensitivity_by_family_data.csv"
SANITY_TXT = OUTPUT_DIR / "figure_8_exam_size_sensitivity_by_family_sanity_check.txt"

# Set to True if dataframe_env is regenerated and the metric cache should be rebuilt.
FORCE_RECOMPUTE_METRICS = False


def family_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "el_tau_b_mean",
        "el_qwk_linear_abs_mean",
        "el_qwk_distrobution_mean",
        "el_acc_linear_abs_mean",
        "el_acc_distrobution_mean",
    ]
    return (
        model_summary.groupby(["family", "test_size"], sort=True)
        .agg(
            n_models=("model_col", "nunique"),
            **{col: (col, "mean") for col in metric_cols},
        )
        .reset_index()
    )


def plot_panel(
    ax: plt.Axes,
    model_summary: pd.DataFrame,
    fam_summary: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
) -> None:
    for model_col, model_df in model_summary.groupby("model_col", sort=False):
        family = str(model_df["family"].iloc[0])
        ax.plot(
            model_df["test_size"],
            model_df[metric_col],
            color=FAMILY_COLORS.get(family, "#666666"),
            linewidth=0.8,
            alpha=0.22,
        )

    for family, family_df in fam_summary.groupby("family", sort=False):
        ax.plot(
            family_df["test_size"],
            family_df[metric_col],
            color=FAMILY_COLORS.get(family, "#666666"),
            marker="o",
            linewidth=2.5,
            label=family,
        )

    ax.set_title(title)
    ax.set_xlabel("Virtual exam size N")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(model_summary["test_size"].dropna().astype(int).unique()))
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_plot(model_summary: pd.DataFrame, fam_summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.0), sharex=True)
    axes_flat = axes.ravel()

    plot_panel(
        axes_flat[0],
        model_summary,
        fam_summary,
        "el_qwk_linear_abs_mean",
        "(a) Absolute Linear EL-QWK",
        "EL-QWK",
    )
    plot_panel(
        axes_flat[1],
        model_summary,
        fam_summary,
        "el_qwk_distrobution_mean",
        "(b) Distrobution EL-QWK",
        "EL-QWK",
    )
    plot_panel(
        axes_flat[2],
        model_summary,
        fam_summary,
        "el_acc_linear_abs_mean",
        "(c) Absolute Linear EL-Acc",
        "EL-Acc",
    )
    plot_panel(
        axes_flat[3],
        model_summary,
        fam_summary,
        "el_acc_distrobution_mean",
        "(d) Distrobution EL-Acc",
        "EL-Acc",
    )
    plot_panel(
        axes_flat[4],
        model_summary,
        fam_summary,
        "el_tau_b_mean",
        r"(e) Ranking Stability EL-$\tau_b$",
        r"EL-$\tau_b$",
    )
    axes_flat[5].axis("off")

    handles, labels = axes_flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Exam Size Sensitivity by Model Family", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sanity(exam_df: pd.DataFrame, model_summary: pd.DataFrame, fam_summary: pd.DataFrame) -> None:
    lines = [
        "=" * 100,
        "FIGURE 8 EXAM SIZE SENSITIVITY BY MODEL FAMILY SANITY CHECK",
        "=" * 100,
        f"Project root:        {PROJECT_ROOT}",
        f"Metric cache:        {EXAM_METRICS_CACHE}",
        f"Exam metric rows:    {len(exam_df)}",
        f"Model summary rows:  {len(model_summary)}",
        f"Family summary rows: {len(fam_summary)}",
        "",
        "Rows by test_size:",
        exam_df.groupby("test_size").size().to_string(),
        "",
        "Metric source counts:",
        exam_df["metric_source"].value_counts(dropna=False).to_string()
        if "metric_source" in exam_df.columns
        else "metric_source column not present",
        "",
        "Family summary:",
        fam_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "Written files:",
        f"  {PLOT_PDF}",
        f"  {PLOT_PNG}",
        f"  {DATA_CSV}",
        f"  {SANITY_TXT}",
    ]
    write_lines(SANITY_TXT, lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exam_df = load_or_compute_exam_metrics(force=FORCE_RECOMPUTE_METRICS)
    model_summary = summarize_exam_metrics(exam_df)
    fam_summary = family_summary(model_summary)
    fam_summary.to_csv(DATA_CSV, index=False, encoding="utf-8")
    save_plot(model_summary, fam_summary)
    write_sanity(exam_df, model_summary, fam_summary)

    print("Saved:")
    print(f"  {PLOT_PDF}")
    print(f"  {PLOT_PNG}")
    print(f"  {DATA_CSV}")
    print(f"  {SANITY_TXT}")


if __name__ == "__main__":
    main()
