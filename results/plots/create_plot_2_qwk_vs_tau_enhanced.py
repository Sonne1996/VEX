#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create an enhanced Figure 2 variant with rank-shift plots only.

This script intentionally does not overwrite create_plot_2_qwk_vs_tau.py.
It reuses the same item-level and exam-level metric preparation, but removes
the scatter panels and focuses on model rankings.

Outputs:
    results/plots/figures_plot_2_enhanced_rankings/
        figure_2_enhanced_q*_item_qwk_vs_el_qwk_model_ranking.pdf/png
        figure_2_enhanced_q*_item_mse_vs_el_qwk_model_ranking.pdf/png
        figure_2_enhanced_q*_item_mse_vs_el_acc_model_ranking.pdf/png
        figure_2_enhanced_q*_ranking_data.csv
        figure_2_enhanced_sanity_check.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import create_plot_2_qwk_vs_tau as base


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_2_enhanced_rankings"

TARGET_TEST_SIZES = [5, 10, 15, 20, 21]

FAMILY_COLORS = base.FAMILY_COLORS
MODEL_COL = base.MODEL_COL
TEST_SIZE_COL = base.TEST_SIZE_COL


def figure_paths(test_size: int) -> dict[str, Path]:
    q = f"q{int(test_size)}"
    return {
        "item_qwk_el_qwk_pdf": OUTPUT_DIR
        / f"figure_2_enhanced_{q}_item_qwk_vs_el_qwk_model_ranking.pdf",
        "item_qwk_el_qwk_png": OUTPUT_DIR
        / f"figure_2_enhanced_{q}_item_qwk_vs_el_qwk_model_ranking.png",
        "item_mse_el_qwk_pdf": OUTPUT_DIR
        / f"figure_2_enhanced_{q}_item_mse_vs_el_qwk_model_ranking.pdf",
        "item_mse_el_qwk_png": OUTPUT_DIR
        / f"figure_2_enhanced_{q}_item_mse_vs_el_qwk_model_ranking.png",
        "item_mse_el_acc_pdf": OUTPUT_DIR
        / f"figure_2_enhanced_{q}_item_mse_vs_el_acc_model_ranking.pdf",
        "item_mse_el_acc_png": OUTPUT_DIR
        / f"figure_2_enhanced_{q}_item_mse_vs_el_acc_model_ranking.png",
        "data_csv": OUTPUT_DIR / f"figure_2_enhanced_{q}_ranking_data.csv",
    }


def load_source_metrics() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not base.INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input parquet not found: {base.INPUT_PARQUET.resolve()}")

    item_source = base.read_parquet(base.INPUT_PARQUET)
    item_df, item_stats = base.build_gold_item_df_from_original_input(item_source)
    item_metrics_df = base.compute_item_metrics(item_df)

    if base.EXAM_METRICS_PARQUET.exists():
        exam_df = base.read_parquet(base.EXAM_METRICS_PARQUET)
    else:
        exam_df = base.load_or_compute_exam_metrics()

    exam_summary_df, exam_stats = base.compute_exam_summary(exam_df)
    return item_metrics_df, exam_summary_df, item_stats, exam_stats


def plot_df_for_size(
    item_metrics_df: pd.DataFrame,
    exam_summary_df: pd.DataFrame,
    test_size: int,
) -> pd.DataFrame:
    exam_summary_size_df = exam_summary_df[
        pd.to_numeric(exam_summary_df[TEST_SIZE_COL], errors="coerce") == int(test_size)
    ].copy()
    if exam_summary_size_df.empty:
        raise ValueError(f"No exam-level rows found for q{test_size}.")

    return base.merge_item_and_exam_metrics(
        item_metrics_df=item_metrics_df,
        exam_summary_df=exam_summary_size_df,
    )


def metric_rank_map(
    df: pd.DataFrame,
    metric_col: str,
    ascending: bool,
) -> tuple[dict[str, int], dict[str, float]]:
    ranked = df.dropna(subset=[metric_col]).sort_values(
        [metric_col, "model"],
        ascending=[ascending, True],
    )
    rank_map = {
        str(model_col): int(rank)
        for rank, model_col in enumerate(ranked[MODEL_COL], start=1)
    }
    value_map = {
        str(row[MODEL_COL]): float(row[metric_col])
        for _, row in ranked.iterrows()
    }
    return rank_map, value_map


def fmt_value(value: float, lower_is_better: bool) -> str:
    prefix = "" if not lower_is_better else "MSE="
    return f"{prefix}{value:.3f}"


def add_rank_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    left_label: str,
    right_label: str,
    left_ascending: bool,
    right_ascending: bool,
    title: str,
) -> pd.DataFrame:
    left_ranks, left_values = metric_rank_map(plot_df, left_col, left_ascending)
    right_ranks, right_values = metric_rank_map(plot_df, right_col, right_ascending)

    ranked = plot_df[
        plot_df[MODEL_COL].astype(str).isin(left_ranks)
        & plot_df[MODEL_COL].astype(str).isin(right_ranks)
    ].copy()
    ranked["left_rank"] = ranked[MODEL_COL].astype(str).map(left_ranks)
    ranked["right_rank"] = ranked[MODEL_COL].astype(str).map(right_ranks)
    ranked["left_value"] = ranked[MODEL_COL].astype(str).map(left_values)
    ranked["right_value"] = ranked[MODEL_COL].astype(str).map(right_values)
    ranked = ranked.sort_values(["left_rank", "right_rank", "model"])

    max_rank = int(max(ranked["left_rank"].max(), ranked["right_rank"].max()))
    left_x = 0.22
    right_x = 0.78

    for _, row in ranked.iterrows():
        color = FAMILY_COLORS.get(str(row["family"]), "#666666")
        ax.plot(
            [left_x, right_x],
            [row["left_rank"], row["right_rank"]],
            color=color,
            linewidth=1.45,
            alpha=0.82,
            zorder=1,
        )
        ax.scatter(
            [left_x, right_x],
            [row["left_rank"], row["right_rank"]],
            color=color,
            edgecolor="black",
            linewidth=0.45,
            s=32,
            zorder=2,
        )

        left_text = (
            f"{int(row['left_rank']):02d}. {row['model']} "
            f"({fmt_value(float(row['left_value']), left_ascending)})"
        )
        right_text = (
            f"{int(row['right_rank']):02d}. {row['model']} "
            f"({fmt_value(float(row['right_value']), right_ascending)})"
        )
        ax.text(left_x - 0.025, row["left_rank"], left_text, ha="right", va="center", fontsize=7)
        ax.text(right_x + 0.025, row["right_rank"], right_text, ha="left", va="center", fontsize=7)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(max_rank + 0.8, 0.2)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels([left_label, right_label])
    ax.set_yticks(range(1, max_rank + 1))
    ax.set_ylabel("Model rank (1 = best)")
    ax.set_title(title)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return ranked


def save_rank_figure(
    plot_df: pd.DataFrame,
    *,
    test_size: int,
    item_col: str,
    item_label: str,
    item_ascending: bool,
    exam_linear_col: str,
    exam_distrobution_col: str,
    exam_label: str,
    exam_ascending: bool,
    output_pdf: Path,
    output_png: Path,
) -> pd.DataFrame:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.2), sharey=False)

    linear_ranked = add_rank_panel(
        axes[0],
        plot_df,
        left_col=item_col,
        right_col=exam_linear_col,
        left_label=f"{item_label}\nitem-level ranking",
        right_label=f"{exam_label} Linear\nexam-level ranking",
        left_ascending=item_ascending,
        right_ascending=exam_ascending,
        title=f"(a) Absolute Linear Scale, q{test_size}",
    )
    linear_ranked["scale"] = "linear_abs"

    dist_ranked = add_rank_panel(
        axes[1],
        plot_df,
        left_col=item_col,
        right_col=exam_distrobution_col,
        left_label=f"{item_label}\nitem-level ranking",
        right_label=f"{exam_label} Distrobution\nexam-level ranking",
        left_ascending=item_ascending,
        right_ascending=exam_ascending,
        title=f"(b) Distrobution Scale, q{test_size}",
    )
    dist_ranked["scale"] = "distrobution"

    handles = []
    labels = []
    for family, color in FAMILY_COLORS.items():
        if family in set(plot_df["family"]):
            handles.append(plt.Line2D([0], [0], color=color, marker="o", linewidth=1.4))
            labels.append(family)

    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"Model Ranking Shift from Item-Level {item_label} to Exam-Level {exam_label}",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.035,
        "Numbers in parentheses are the metric values used for ranking.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.94], w_pad=3.6)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    combined = pd.concat([linear_ranked, dist_ranked], ignore_index=True)
    combined["comparison_item_col"] = item_col
    combined["comparison_exam_label"] = exam_label
    return combined


def write_sanity(
    all_rank_data: pd.DataFrame,
    item_stats: dict[str, Any],
    exam_stats: dict[str, Any],
    written_paths: list[Path],
) -> None:
    sanity_path = OUTPUT_DIR / "figure_2_enhanced_sanity_check.txt"
    lines = [
        "=" * 100,
        "FIGURE 2 ENHANCED RANKING-ONLY SANITY CHECK",
        "=" * 100,
        f"Item input parquet:       {base.INPUT_PARQUET.resolve()}",
        f"Exam metrics parquet:     {base.EXAM_METRICS_PARQUET.resolve()}",
        f"Target test sizes:        {TARGET_TEST_SIZES}",
        "",
        "Item stats:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in item_stats.items())
    lines.append("")
    lines.append("Exam stats:")
    lines.extend(f"  {key}: {value}" for key, value in exam_stats.items())
    lines.append("")
    lines.append("Rank data preview:")
    lines.append(
        all_rank_data[
            [
                "test_size",
                "scale",
                "comparison_item_col",
                "comparison_exam_label",
                "model",
                "family",
                "left_rank",
                "left_value",
                "right_rank",
                "right_value",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    lines.append("")
    lines.append("Written files:")
    for path in written_paths:
        lines.append(f"  {path}")
    lines.append(f"  {sanity_path}")
    sanity_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    item_metrics_df, exam_summary_df, item_stats, exam_stats = load_source_metrics()

    all_rank_frames: list[pd.DataFrame] = []
    written_paths: list[Path] = []

    for test_size in TARGET_TEST_SIZES:
        paths = figure_paths(test_size)
        plot_df = plot_df_for_size(item_metrics_df, exam_summary_df, test_size)
        plot_df.to_csv(paths["data_csv"], index=False, encoding="utf-8")
        written_paths.append(paths["data_csv"])

        comparisons = [
            {
                "item_col": "item_qwk",
                "item_label": "QWK",
                "item_ascending": False,
                "exam_linear_col": "el_qwk_linear",
                "exam_distrobution_col": "el_qwk_distrobution",
                "exam_label": "EL-QWK",
                "exam_ascending": False,
                "pdf": paths["item_qwk_el_qwk_pdf"],
                "png": paths["item_qwk_el_qwk_png"],
            },
            {
                "item_col": "item_mse",
                "item_label": "MSE",
                "item_ascending": True,
                "exam_linear_col": "el_qwk_linear",
                "exam_distrobution_col": "el_qwk_distrobution",
                "exam_label": "EL-QWK",
                "exam_ascending": False,
                "pdf": paths["item_mse_el_qwk_pdf"],
                "png": paths["item_mse_el_qwk_png"],
            },
            {
                "item_col": "item_mse",
                "item_label": "MSE",
                "item_ascending": True,
                "exam_linear_col": "el_acc_linear",
                "exam_distrobution_col": "el_acc_distrobution",
                "exam_label": "EL-Acc",
                "exam_ascending": False,
                "pdf": paths["item_mse_el_acc_pdf"],
                "png": paths["item_mse_el_acc_png"],
            },
        ]

        for comparison in comparisons:
            rank_df = save_rank_figure(
                plot_df,
                test_size=test_size,
                item_col=str(comparison["item_col"]),
                item_label=str(comparison["item_label"]),
                item_ascending=bool(comparison["item_ascending"]),
                exam_linear_col=str(comparison["exam_linear_col"]),
                exam_distrobution_col=str(comparison["exam_distrobution_col"]),
                exam_label=str(comparison["exam_label"]),
                exam_ascending=bool(comparison["exam_ascending"]),
                output_pdf=Path(comparison["pdf"]),
                output_png=Path(comparison["png"]),
            )
            rank_df["test_size"] = int(test_size)
            all_rank_frames.append(rank_df)
            written_paths.extend([Path(comparison["pdf"]), Path(comparison["png"])])

    all_rank_data = pd.concat(all_rank_frames, ignore_index=True)
    combined_csv = OUTPUT_DIR / "figure_2_enhanced_all_q_ranking_data.csv"
    all_rank_data.to_csv(combined_csv, index=False, encoding="utf-8")
    written_paths.append(combined_csv)
    write_sanity(all_rank_data, item_stats, exam_stats, written_paths)

    print("Saved:")
    for path in written_paths:
        print(f"  {path}")
    print(f"  {OUTPUT_DIR / 'figure_2_enhanced_sanity_check.txt'}")


if __name__ == "__main__":
    main()
