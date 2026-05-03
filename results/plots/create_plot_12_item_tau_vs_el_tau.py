#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse item-level tau and exam-level tau.

Outputs:
    results/plots/figures_plot_12_tau_analysis/
        figure_12_q*_item_tau_vs_el_tau.pdf
        figure_12_q*_item_tau_vs_el_tau.png
        figure_12_q*_item_tau_vs_el_tau_data.csv
        figure_12_q*_item_tau_vs_el_tau_sanity_check.txt
        figure_12_item_tau_vs_el_tau_all_q_data.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vex_plot_metrics import (
    FAMILY_COLORS,
    FAMILY_MARKERS,
    MODEL_COLUMNS,
    PROJECT_ROOT,
    display_name,
    load_or_compute_exam_metrics,
    model_family,
    summarize_exam_metrics,
    tau_b_safe,
    write_lines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PARQUET = (
    PROJECT_ROOT / "dataset" / "additional" / "vex_metric_dataset" / "merged_model_predictions.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_12_tau_analysis"

TARGET_TEST_SIZES = [5, 10, 15, 20, 21]
COMBINED_DATA_CSV = OUTPUT_DIR / "figure_12_item_tau_vs_el_tau_all_q_data.csv"
GRADE_COL = "grade"


def figure_12_paths(test_size: int) -> dict[str, Path]:
    q = f"q{int(test_size)}"
    return {
        "plot_pdf": OUTPUT_DIR / f"figure_12_{q}_item_tau_vs_el_tau.pdf",
        "plot_png": OUTPUT_DIR / f"figure_12_{q}_item_tau_vs_el_tau.png",
        "data_csv": OUTPUT_DIR / f"figure_12_{q}_item_tau_vs_el_tau_data.csv",
        "sanity_txt": OUTPUT_DIR / f"figure_12_{q}_item_tau_vs_el_tau_sanity_check.txt",
    }


def compute_item_tau(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true_full = pd.to_numeric(df[GRADE_COL], errors="coerce")

    for model_col in MODEL_COLUMNS:
        y_pred_full = pd.to_numeric(df[model_col], errors="coerce")
        valid = y_true_full.notna() & y_pred_full.notna()
        y_true = y_true_full[valid].to_numpy(dtype=float)
        y_pred = y_pred_full[valid].to_numpy(dtype=float)
        rows.append(
            {
                "model_col": model_col,
                "model": display_name(model_col),
                "family": model_family(model_col),
                "item_n": int(valid.sum()),
                "item_tau_b": tau_b_safe(y_true, y_pred),
            }
        )

    return pd.DataFrame(rows)


def build_plot_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_df = pd.read_parquet(INPUT_PARQUET)
    item_tau = compute_item_tau(item_df)

    exam_df = load_or_compute_exam_metrics()
    exam_summary = summarize_exam_metrics(exam_df)

    plot_df = item_tau.merge(
        exam_summary[
            [
                "model_col",
                "test_size",
                "exam_instances",
                "el_tau_b_mean",
                "el_tau_b_std",
            ]
        ],
        on="model_col",
        how="left",
    )
    plot_df["tau_gain_el_minus_item"] = plot_df["el_tau_b_mean"] - plot_df["item_tau_b"]
    return plot_df, item_tau, exam_summary


def save_plot(plot_df: pd.DataFrame, test_size: int, paths: dict[str, Path]) -> None:
    q_df = plot_df[pd.to_numeric(plot_df["test_size"], errors="coerce") == int(test_size)].copy()
    if q_df.empty:
        raise ValueError(f"No plot data available for q{test_size}.")
    q_df = q_df.sort_values("el_tau_b_mean", ascending=True)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    for family, family_df in q_df.groupby("family", sort=False):
        axes[0].scatter(
            family_df["item_tau_b"],
            family_df["el_tau_b_mean"],
            s=62,
            marker=FAMILY_MARKERS.get(family, "o"),
            color=FAMILY_COLORS.get(family, "#666666"),
            edgecolor="black",
            linewidth=0.45,
            label=family,
            alpha=0.9,
        )
    for _, row in q_df.iterrows():
        axes[0].annotate(
            str(row["model"]),
            xy=(row["item_tau_b"], row["el_tau_b_mean"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    low = max(0.0, min(q_df["item_tau_b"].min(), q_df["el_tau_b_mean"].min()) - 0.05)
    high = min(1.0, max(q_df["item_tau_b"].max(), q_df["el_tau_b_mean"].max()) + 0.05)
    axes[0].plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(low, high)
    axes[0].set_ylim(low, high)
    axes[0].set_title(f"(a) Item Tau vs. EL-Tau, q{test_size}")
    axes[0].set_xlabel("Item-level tau-b")
    axes[0].set_ylabel("Exam-level tau-b")

    y = np.arange(len(q_df))
    axes[1].barh(
        y,
        q_df["tau_gain_el_minus_item"],
        color=np.where(q_df["tau_gain_el_minus_item"] >= 0, "#1f77b4", "#d62728"),
        edgecolor="#333333",
        linewidth=0.5,
    )
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(q_df["model"])
    axes[1].set_title(f"(b) EL-Tau minus Item-Tau, q{test_size}")
    axes[1].set_xlabel("Tau difference")

    for model_col, model_df in plot_df.groupby("model_col", sort=False):
        family = str(model_df["family"].iloc[0])
        axes[2].plot(
            model_df.sort_values("test_size")["test_size"],
            model_df.sort_values("test_size")["el_tau_b_mean"],
            color=FAMILY_COLORS.get(family, "#666666"),
            alpha=0.35,
            linewidth=1.0,
        )
    family_curve = (
        plot_df.groupby(["family", "test_size"], sort=True)["el_tau_b_mean"]
        .mean()
        .reset_index()
    )
    for family, family_df in family_curve.groupby("family", sort=False):
        axes[2].plot(
            family_df["test_size"],
            family_df["el_tau_b_mean"],
            color=FAMILY_COLORS.get(family, "#666666"),
            marker="o",
            linewidth=2.4,
            label=family,
        )
    axes[2].set_title("(c) EL-Tau by Exam Size")
    axes[2].set_xlabel("Virtual exam size N")
    axes[2].set_ylabel("Exam-level tau-b")
    axes[2].set_xticks(sorted(plot_df["test_size"].dropna().astype(int).unique()))

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="lower right")
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle("Item-Level and Exam-Level Tau Analysis", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(paths["plot_pdf"], bbox_inches="tight")
    fig.savefig(paths["plot_png"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sanity(
    plot_df: pd.DataFrame,
    item_tau: pd.DataFrame,
    exam_summary: pd.DataFrame,
    test_size: int,
    paths: dict[str, Path],
) -> None:
    q_df = plot_df[pd.to_numeric(plot_df["test_size"], errors="coerce") == int(test_size)]
    lines = [
        "=" * 100,
        f"FIGURE 12 Q{test_size} ITEM-LEVEL TAU VS EXAM-LEVEL TAU SANITY CHECK",
        "=" * 100,
        f"Input parquet:      {INPUT_PARQUET}",
        f"Target test_size:   {test_size}",
        f"Item tau rows:      {len(item_tau)}",
        f"Exam summary rows:  {len(exam_summary)}",
        "",
        f"q{test_size} data:",
        q_df.sort_values("el_tau_b_mean", ascending=False).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Written files:",
        f"  {paths['plot_pdf']}",
        f"  {paths['plot_png']}",
        f"  {paths['data_csv']}",
        f"  {paths['sanity_txt']}",
    ]
    write_lines(paths["sanity_txt"], lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df, item_tau, exam_summary = build_plot_data()
    plot_df.to_csv(COMBINED_DATA_CSV, index=False, encoding="utf-8")

    written: list[Path] = [COMBINED_DATA_CSV]
    for test_size in TARGET_TEST_SIZES:
        paths = figure_12_paths(test_size)
        q_df = plot_df[
            pd.to_numeric(plot_df["test_size"], errors="coerce") == int(test_size)
        ].copy()
        if q_df.empty:
            raise ValueError(f"No item/exam tau rows found for q{test_size}.")
        q_df.to_csv(paths["data_csv"], index=False, encoding="utf-8")
        save_plot(plot_df, test_size=test_size, paths=paths)
        write_sanity(
            plot_df=plot_df,
            item_tau=item_tau,
            exam_summary=exam_summary,
            test_size=test_size,
            paths=paths,
        )
        written.extend(
            [
                paths["plot_pdf"],
                paths["plot_png"],
                paths["data_csv"],
                paths["sanity_txt"],
            ]
        )

    print("Saved:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
