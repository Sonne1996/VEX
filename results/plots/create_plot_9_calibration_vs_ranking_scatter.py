#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create calibration-vs-ranking scatter plots.

The x-axis uses absolute linear EL-QWK. The y-axis uses distribution-based
EL-QWK. Points above the diagonal indicate stronger relative ranking than
absolute calibration.
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
    FAMILY_MARKERS,
    load_or_compute_exam_metrics,
    summarize_exam_metrics,
    write_lines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_9_calibration_vs_ranking"

PLOT_PDF = OUTPUT_DIR / "figure_9_calibration_vs_ranking_scatter.pdf"
PLOT_PNG = OUTPUT_DIR / "figure_9_calibration_vs_ranking_scatter.png"
DATA_CSV = OUTPUT_DIR / "figure_9_calibration_vs_ranking_scatter_data.csv"
SANITY_TXT = OUTPUT_DIR / "figure_9_calibration_vs_ranking_scatter_sanity_check.txt"

# Set to True if dataframe_env is regenerated and the metric cache should be rebuilt.
FORCE_RECOMPUTE_METRICS = False


def finite_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0, 1.0
    low = float(finite.min())
    high = float(finite.max())
    pad = max((high - low) * 0.12, 0.03)
    return max(0.0, low - pad), min(1.0, high + pad)


def save_plot(plot_df: pd.DataFrame) -> None:
    test_sizes = sorted(plot_df["test_size"].dropna().astype(int).unique())
    ncols = min(3, len(test_sizes))
    nrows = int(np.ceil(len(test_sizes) / ncols))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    x_col = "el_qwk_linear_abs_mean"
    y_col = "el_qwk_distrobution_mean"

    x_lim = finite_limits(plot_df[x_col])
    y_lim = finite_limits(plot_df[y_col])
    low = min(x_lim[0], y_lim[0])
    high = max(x_lim[1], y_lim[1])

    for ax, test_size in zip(axes_flat, test_sizes, strict=False):
        q_df = plot_df[plot_df["test_size"].astype(int) == int(test_size)].copy()

        for family, family_df in q_df.groupby("family", sort=False):
            ax.scatter(
                family_df[x_col],
                family_df[y_col],
                s=58,
                marker=FAMILY_MARKERS.get(family, "o"),
                color=FAMILY_COLORS.get(family, "#666666"),
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                label=family,
            )

        for _, row in q_df.iterrows():
            ax.annotate(
                str(row["model"]),
                xy=(row[x_col], row[y_col]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.5,
            )

        ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(f"q{test_size}")
        ax.set_xlabel("EL-QWK Absolute Linear")
        ax.set_ylabel("EL-QWK Distrobution")
        ax.grid(True, alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(test_sizes) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Calibration vs. Ranking: Absolute Linear and Distrobution EL-QWK", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sanity(exam_df: pd.DataFrame, plot_df: pd.DataFrame) -> None:
    work = plot_df.copy()
    work["distrobution_minus_linear"] = (
        work["el_qwk_distrobution_mean"] - work["el_qwk_linear_abs_mean"]
    )
    lines = [
        "=" * 100,
        "FIGURE 9 CALIBRATION VS RANKING SCATTER SANITY CHECK",
        "=" * 100,
        f"Metric cache:     {EXAM_METRICS_CACHE}",
        f"Exam metric rows: {len(exam_df)}",
        "",
        "Metric source counts:",
        exam_df["metric_source"].value_counts(dropna=False).to_string()
        if "metric_source" in exam_df.columns
        else "metric_source column not present",
        "",
        "Scatter data:",
        work[
            [
                "test_size",
                "model",
                "family",
                "el_qwk_linear_abs_mean",
                "el_qwk_distrobution_mean",
                "distrobution_minus_linear",
            ]
        ].sort_values(["test_size", "el_qwk_linear_abs_mean"], ascending=[True, False]).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
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
    plot_df = summarize_exam_metrics(exam_df)
    plot_df.to_csv(DATA_CSV, index=False, encoding="utf-8")
    save_plot(plot_df)
    write_sanity(exam_df, plot_df)

    print("Saved:")
    print(f"  {PLOT_PDF}")
    print(f"  {PLOT_PNG}")
    print(f"  {DATA_CSV}")
    print(f"  {SANITY_TXT}")


if __name__ == "__main__":
    main()
