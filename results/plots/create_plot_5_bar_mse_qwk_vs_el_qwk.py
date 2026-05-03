#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create model bars comparing item-level metrics with exam-level QWK.

Input:
    dataset/additional/vex_metric_dataset/merged_model_predictions.parquet
    results/plots/_metric_cache/exam_level_metrics_cache.csv

Outputs:
    results/plots/figures_plot_5_item_vs_el_bars/
        figure_5_q*_item_mse_qwk_vs_el_qwk_bars.pdf
        figure_5_q*_item_mse_qwk_vs_el_qwk_bars.png
        figure_5_q*_item_mse_qwk_vs_el_qwk_bars_data.csv
        figure_5_q*_item_mse_qwk_vs_el_qwk_bars_sanity_check.txt
        figure_5_item_mse_qwk_vs_el_qwk_bars_all_q_data.csv
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
    MODEL_COLUMNS,
    PROJECT_ROOT,
    display_name,
    load_or_compute_exam_metrics,
    model_family,
    qwk_safe,
    summarize_exam_metrics,
    write_lines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PARQUET = (
    PROJECT_ROOT / "dataset" / "additional" / "vex_metric_dataset" / "merged_model_predictions.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_5_item_vs_el_bars"

TARGET_TEST_SIZES = [5, 10, 15, 20, 21]
COMBINED_DATA_CSV = OUTPUT_DIR / "figure_5_item_mse_qwk_vs_el_qwk_bars_all_q_data.csv"
GRADE_COL = "grade"


def figure_5_paths(test_size: int) -> dict[str, Path]:
    q = f"q{int(test_size)}"
    return {
        "plot_pdf": OUTPUT_DIR / f"figure_5_{q}_item_mse_qwk_vs_el_qwk_bars.pdf",
        "plot_png": OUTPUT_DIR / f"figure_5_{q}_item_mse_qwk_vs_el_qwk_bars.png",
        "data_csv": OUTPUT_DIR / f"figure_5_{q}_item_mse_qwk_vs_el_qwk_bars_data.csv",
        "sanity_txt": OUTPUT_DIR / f"figure_5_{q}_item_mse_qwk_vs_el_qwk_bars_sanity_check.txt",
    }


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def compute_item_metrics(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, [GRADE_COL, *MODEL_COLUMNS])
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
                "item_missing": int((~valid).sum()),
                "item_mse": float(np.mean((y_true - y_pred) ** 2)) if len(y_true) else np.nan,
                "item_qwk": qwk_safe(y_true, y_pred),
            }
        )

    return pd.DataFrame(rows)


def build_item_and_exam_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    item_df = pd.read_parquet(INPUT_PARQUET)
    item_metrics = compute_item_metrics(item_df)

    exam_df = load_or_compute_exam_metrics()
    exam_summary = summarize_exam_metrics(exam_df)
    return item_metrics, exam_summary


def build_plot_data_for_size(
    item_metrics: pd.DataFrame,
    exam_summary: pd.DataFrame,
    test_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_summary = exam_summary[
        pd.to_numeric(exam_summary["test_size"], errors="coerce") == int(test_size)
    ].copy()
    if q_summary.empty:
        raise ValueError(f"No exam-level summary rows found for q{test_size}.")

    plot_df = item_metrics.merge(
        q_summary[
            [
                "model_col",
                "test_size",
                "exam_instances",
                "el_qwk_linear_abs_mean",
                "el_qwk_distrobution_mean",
                "el_tau_b_mean",
            ]
        ],
        on="model_col",
        how="left",
    )
    plot_df["item_one_minus_mse"] = 1.0 - plot_df["item_mse"]
    plot_df = plot_df.sort_values("el_qwk_linear_abs_mean", ascending=True, na_position="first")
    return plot_df, q_summary


def save_plot(plot_df: pd.DataFrame, test_size: int, paths: dict[str, Path]) -> None:
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

    plot_df = plot_df.sort_values(
        "el_qwk_linear_abs_mean",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    n_models = len(plot_df)
    ncols = 4
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13.8, 2.55 * nrows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    x_positions = np.array([0.00, 0.34, 1.05, 1.39])
    x_centers = [0.17, 1.22]
    bar_width = 0.28

    legend_handles = None
    legend_labels = None

    for ax, (_, row) in zip(axes_flat, plot_df.iterrows(), strict=False):
        values = [
            row["item_one_minus_mse"],
            row["el_qwk_linear_abs_mean"],
            row["item_qwk"],
            row["el_qwk_linear_abs_mean"],
        ]
        labels = [
            "1 - Item-MSE",
            f"EL-QWK q{test_size}",
            "Item-QWK",
            f"EL-QWK q{test_size}",
        ]
        colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#1f77b4"]
        edges = ["#5b2a00", "#16344f", "#1e4f1e", "#16344f"]

        for x_pos, value, label, color, edge in zip(
            x_positions,
            values,
            labels,
            colors,
            edges,
            strict=False,
        ):
            ax.bar(
                x_pos,
                value,
                width=bar_width,
                color=color,
                edgecolor=edge,
                linewidth=0.55,
                label=label,
            )

        ax.set_title(str(row["model"]), fontsize=8.5, pad=4)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(["1-MSE vs\nEL-QWK", "QWK vs\nEL-QWK"], fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b0b0b0")
        ax.spines["bottom"].set_color("#b0b0b0")

        if legend_handles is None:
            handles, labels_seen = ax.get_legend_handles_labels()
            unique: dict[str, object] = {}
            for handle, label in zip(handles, labels_seen, strict=False):
                unique.setdefault(label, handle)
            legend_labels = list(unique.keys())
            legend_handles = list(unique.values())

    for ax in axes_flat[n_models:]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel("Metric value")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"Per-Model Item Metrics Compared with Exam-Level QWK, q{test_size}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(paths["plot_pdf"], bbox_inches="tight")
    fig.savefig(paths["plot_png"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sanity(
    plot_df: pd.DataFrame,
    item_metrics: pd.DataFrame,
    q_summary: pd.DataFrame,
    test_size: int,
    paths: dict[str, Path],
) -> None:
    lines = [
        "=" * 100,
        f"FIGURE 5 Q{test_size} ITEM METRICS VS EXAM-LEVEL QWK SANITY CHECK",
        "=" * 100,
        f"Input parquet:      {INPUT_PARQUET}",
        f"Target test_size:   {test_size}",
        f"Item metric rows:   {len(item_metrics)}",
        f"Exam summary rows:  {len(q_summary)}",
        "",
        "Definitions:",
        "  Item-MSE = mean squared error over gold item-level rows.",
        "  1 - Item-MSE is plotted so that larger bars consistently indicate better performance.",
        "  Item-QWK = quadratic weighted kappa over gold item-level rows.",
        "  EL-QWK = mean absolute linear exam-level QWK over virtual exams.",
        "",
        "Plot data:",
        plot_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
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
    item_metrics, exam_summary = build_item_and_exam_summaries()

    all_plot_dfs: list[pd.DataFrame] = []
    written: list[Path] = []

    for test_size in TARGET_TEST_SIZES:
        paths = figure_5_paths(test_size)
        plot_df, q_summary = build_plot_data_for_size(
            item_metrics=item_metrics,
            exam_summary=exam_summary,
            test_size=test_size,
        )
        plot_df.to_csv(paths["data_csv"], index=False, encoding="utf-8")
        save_plot(plot_df, test_size=test_size, paths=paths)
        write_sanity(
            plot_df=plot_df,
            item_metrics=item_metrics,
            q_summary=q_summary,
            test_size=test_size,
            paths=paths,
        )
        all_plot_dfs.append(plot_df)
        written.extend(
            [
                paths["plot_pdf"],
                paths["plot_png"],
                paths["data_csv"],
                paths["sanity_txt"],
            ]
        )

    combined_df = pd.concat(all_plot_dfs, ignore_index=True)
    combined_df.to_csv(COMBINED_DATA_CSV, index=False, encoding="utf-8")
    written.append(COMBINED_DATA_CSV)

    print("Saved:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
