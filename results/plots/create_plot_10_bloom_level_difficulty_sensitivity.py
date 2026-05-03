#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create Bloom-level question difficulty and model sensitivity plots.

Input:
    dataset/additional/vex_metric_dataset/merged_model_predictions.parquet

Outputs:
    results/plots/figures_plot_10_bloom_level_sensitivity/
        figure_10_bloom_level_difficulty_sensitivity.pdf
        figure_10_bloom_level_difficulty_sensitivity.png
        figure_10_bloom_level_difficulty_sensitivity_data.csv
        figure_10_bloom_level_difficulty_sensitivity_sanity_check.txt
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
    BLOOM_LABELS,
    BLOOM_ORDER,
    FAMILY_COLORS,
    MODEL_COLUMNS,
    PROJECT_ROOT,
    display_name,
    model_family,
    qwk_safe,
    tau_b_safe,
    write_lines,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PARQUET = (
    PROJECT_ROOT / "dataset" / "additional" / "vex_metric_dataset" / "merged_model_predictions.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_10_bloom_level_sensitivity"

PLOT_PDF = OUTPUT_DIR / "figure_10_bloom_level_difficulty_sensitivity.pdf"
PLOT_PNG = OUTPUT_DIR / "figure_10_bloom_level_difficulty_sensitivity.png"
DATA_CSV = OUTPUT_DIR / "figure_10_bloom_level_difficulty_sensitivity_data.csv"
SANITY_TXT = OUTPUT_DIR / "figure_10_bloom_level_difficulty_sensitivity_sanity_check.txt"

GRADE_COL = "grade"
BLOOM_COL = "bloom_level"
QUESTION_COL = "question_id"
STUDENT_COL = "member_id"


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def compute_bloom_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(df, [GRADE_COL, BLOOM_COL, QUESTION_COL, STUDENT_COL, *MODEL_COLUMNS])
    work = df[[GRADE_COL, BLOOM_COL, QUESTION_COL, STUDENT_COL, *MODEL_COLUMNS]].copy()
    work[GRADE_COL] = pd.to_numeric(work[GRADE_COL], errors="coerce")
    work[BLOOM_COL] = work[BLOOM_COL].astype("string").str.lower().str.strip()
    work = work[work[BLOOM_COL].isin(BLOOM_ORDER)].copy()

    difficulty = (
        work.groupby(BLOOM_COL, sort=False)
        .agg(
            responses=(GRADE_COL, "size"),
            unique_questions=(QUESTION_COL, "nunique"),
            unique_students=(STUDENT_COL, "nunique"),
            human_mean_score=(GRADE_COL, "mean"),
            human_median_score=(GRADE_COL, "median"),
        )
        .reset_index()
    )
    difficulty["bloom_label"] = difficulty[BLOOM_COL].map(BLOOM_LABELS)
    difficulty["difficulty"] = 1.0 - difficulty["human_mean_score"]

    model_rows: list[dict[str, Any]] = []
    for model_col in MODEL_COLUMNS:
        for bloom, bloom_df in work.groupby(BLOOM_COL, sort=False):
            subset = bloom_df[[GRADE_COL, model_col]].copy()
            subset[model_col] = pd.to_numeric(subset[model_col], errors="coerce")
            subset = subset.dropna(subset=[GRADE_COL, model_col])
            y_true = subset[GRADE_COL].to_numpy(dtype=float)
            y_pred = subset[model_col].to_numpy(dtype=float)

            if len(subset) == 0:
                mae = mse = qwk = tau = np.nan
            else:
                mae = float(np.mean(np.abs(y_true - y_pred)))
                mse = float(np.mean((y_true - y_pred) ** 2))
                qwk = qwk_safe(y_true, y_pred)
                tau = tau_b_safe(y_true, y_pred)

            model_rows.append(
                {
                    "model_col": model_col,
                    "model": display_name(model_col),
                    "family": model_family(model_col),
                    "bloom_level": bloom,
                    "bloom_label": BLOOM_LABELS.get(str(bloom), str(bloom)),
                    "n": int(len(subset)),
                    "mae": mae,
                    "mse": mse,
                    "qwk": qwk,
                    "tau_b": tau,
                }
            )

    model_metrics = pd.DataFrame(model_rows)
    family_metrics = (
        model_metrics.groupby(["family", "bloom_level", "bloom_label"], sort=False)
        .agg(
            n_models=("model_col", "nunique"),
            n_items_mean=("n", "mean"),
            mae_mean=("mae", "mean"),
            mse_mean=("mse", "mean"),
            qwk_mean=("qwk", "mean"),
            tau_b_mean=("tau_b", "mean"),
        )
        .reset_index()
    )

    bloom_order_map = {value: idx for idx, value in enumerate(BLOOM_ORDER)}
    for out_df in (difficulty, model_metrics, family_metrics):
        out_df["bloom_order"] = out_df["bloom_level"].map(bloom_order_map)
        out_df.sort_values("bloom_order", inplace=True)

    return difficulty, model_metrics, family_metrics


def plot_family_lines(
    ax: plt.Axes,
    family_metrics: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
) -> None:
    for family, family_df in family_metrics.groupby("family", sort=False):
        family_df = family_df.sort_values("bloom_order")
        ax.plot(
            family_df["bloom_label"],
            family_df[metric_col],
            marker="o",
            linewidth=2.0,
            color=FAMILY_COLORS.get(family, "#666666"),
            label=family,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_plot(difficulty: pd.DataFrame, family_metrics: pd.DataFrame) -> None:
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

    difficulty = difficulty.sort_values("bloom_order")
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))

    bars = axes[0].bar(
        difficulty["bloom_label"],
        difficulty["human_mean_score"],
        color="#4c78a8",
        edgecolor="#263645",
        linewidth=0.8,
    )
    for bar, responses in zip(bars, difficulty["responses"], strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"n={int(responses)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("(a) Human Score by Bloom Level")
    axes[0].set_ylabel("Mean gold score")
    axes[0].grid(True, axis="y", alpha=0.28)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    plot_family_lines(
        axes[1],
        family_metrics,
        "mae_mean",
        "(b) Model Error by Bloom Level",
        "Mean item-level MAE",
    )
    plot_family_lines(
        axes[2],
        family_metrics,
        "qwk_mean",
        "(c) Ordinal Agreement by Bloom Level",
        "Mean item-level QWK",
    )

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Question Difficulty Sensitivity by Bloom Level", fontsize=13)
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sanity(
    df: pd.DataFrame,
    difficulty: pd.DataFrame,
    model_metrics: pd.DataFrame,
    family_metrics: pd.DataFrame,
) -> None:
    lines = [
        "=" * 100,
        "FIGURE 10 BLOOM LEVEL DIFFICULTY SENSITIVITY SANITY CHECK",
        "=" * 100,
        f"Input parquet:       {INPUT_PARQUET}",
        f"Source rows:         {len(df)}",
        f"Model metric rows:   {len(model_metrics)}",
        f"Family metric rows:  {len(family_metrics)}",
        "",
        "Difficulty table:",
        difficulty.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "",
        "Family metrics:",
        family_metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"),
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
    df = pd.read_parquet(INPUT_PARQUET)
    difficulty, model_metrics, family_metrics = compute_bloom_metrics(df)

    combined = pd.concat(
        [
            difficulty.assign(table="difficulty"),
            model_metrics.assign(table="model_metrics"),
            family_metrics.assign(table="family_metrics"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(DATA_CSV, index=False, encoding="utf-8")

    save_plot(difficulty, family_metrics)
    write_sanity(df, difficulty, model_metrics, family_metrics)

    print("Saved:")
    print(f"  {PLOT_PDF}")
    print(f"  {PLOT_PNG}")
    print(f"  {DATA_CSV}")
    print(f"  {SANITY_TXT}")


if __name__ == "__main__":
    main()
