#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create dataset/data-card overview plots for the VEX v1.0 release.

Panels:
    (a) answer length by label
    (b) label distribution by Bloom level
    (c) question-level label distribution heatmap
    (d) student performance distribution
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vex_plot_metrics import BLOOM_LABELS, BLOOM_ORDER, PROJECT_ROOT, write_lines


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PARQUET = PROJECT_ROOT / "dataset" / "vex" / "v1_0_release" / "v1_0_stable.parquet"
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_11_dataset_datacard"

PLOT_PDF = OUTPUT_DIR / "figure_11_dataset_datacard_overview.pdf"
PLOT_PNG = OUTPUT_DIR / "figure_11_dataset_datacard_overview.png"
DATA_CSV = OUTPUT_DIR / "figure_11_dataset_datacard_overview_data.csv"
SANITY_TXT = OUTPUT_DIR / "figure_11_dataset_datacard_overview_sanity_check.txt"

GRADE_COL = "grade"
ANSWER_COL = "answer"
BLOOM_COL = "bloom_level"
QUESTION_COL = "question_id"
STUDENT_COL = "member_id"

GRADE_ORDER = [0.0, 0.25, 0.5, 0.75, 1.0]
GRADE_LABELS = ["0.00", "0.25", "0.50", "0.75", "1.00"]


def token_count(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    return len(text.split()) if text else 0


def prepare_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df[[GRADE_COL, ANSWER_COL, BLOOM_COL, QUESTION_COL, STUDENT_COL]].copy()
    work[GRADE_COL] = pd.to_numeric(work[GRADE_COL], errors="coerce")
    work["answer_tokens"] = work[ANSWER_COL].map(token_count)
    work[BLOOM_COL] = work[BLOOM_COL].astype("string").str.lower().str.strip()

    length_by_label = (
        work.groupby(GRADE_COL)
        .agg(
            responses=(GRADE_COL, "size"),
            mean_tokens=("answer_tokens", "mean"),
            median_tokens=("answer_tokens", "median"),
            p75_tokens=("answer_tokens", lambda s: float(np.percentile(s, 75))),
        )
        .reindex(GRADE_ORDER)
        .reset_index()
    )

    bloom_label_counts = (
        work.groupby([BLOOM_COL, GRADE_COL])
        .size()
        .rename("count")
        .reset_index()
    )
    bloom_totals = bloom_label_counts.groupby(BLOOM_COL)["count"].transform("sum")
    bloom_label_counts["percent"] = bloom_label_counts["count"] / bloom_totals * 100.0
    bloom_label_counts["bloom_order"] = bloom_label_counts[BLOOM_COL].map(
        {value: idx for idx, value in enumerate(BLOOM_ORDER)}
    )
    bloom_label_counts["bloom_label"] = bloom_label_counts[BLOOM_COL].map(BLOOM_LABELS)

    question_label_counts = (
        work.groupby([QUESTION_COL, GRADE_COL])
        .size()
        .rename("count")
        .reset_index()
    )
    question_totals = question_label_counts.groupby(QUESTION_COL)["count"].transform("sum")
    question_label_counts["percent"] = question_label_counts["count"] / question_totals * 100.0
    question_difficulty = (
        work.groupby(QUESTION_COL)
        .agg(
            human_mean_score=(GRADE_COL, "mean"),
            responses=(GRADE_COL, "size"),
            bloom_level=(BLOOM_COL, "first"),
        )
        .reset_index()
        .sort_values("human_mean_score")
    )
    question_difficulty["question_order"] = np.arange(len(question_difficulty))
    question_label_counts = question_label_counts.merge(
        question_difficulty[[QUESTION_COL, "question_order", "human_mean_score", "bloom_level"]],
        on=QUESTION_COL,
        how="left",
    )

    student_perf = (
        work.groupby(STUDENT_COL)
        .agg(
            mean_score=(GRADE_COL, "mean"),
            responses=(GRADE_COL, "size"),
            median_score=(GRADE_COL, "median"),
        )
        .reset_index()
    )

    return {
        "work": work,
        "length_by_label": length_by_label,
        "bloom_label_counts": bloom_label_counts,
        "question_label_counts": question_label_counts,
        "question_difficulty": question_difficulty,
        "student_perf": student_perf,
    }


def save_plot(data: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4))

    length_df = data["length_by_label"]
    x = np.arange(len(GRADE_ORDER))
    axes[0, 0].bar(
        x,
        length_df["mean_tokens"],
        color="#4c78a8",
        edgecolor="#263645",
        linewidth=0.7,
        label="Mean",
    )
    axes[0, 0].plot(
        x,
        length_df["median_tokens"],
        color="#f58518",
        marker="o",
        linewidth=2,
        label="Median",
    )
    axes[0, 0].set_title("(a) Answer Length by Label")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(GRADE_LABELS)
    axes[0, 0].set_xlabel("Gold label")
    axes[0, 0].set_ylabel("Whitespace tokens")
    axes[0, 0].legend(frameon=False)

    bloom_df = data["bloom_label_counts"].copy()
    pivot = (
        bloom_df.pivot_table(
            index="bloom_label",
            columns=GRADE_COL,
            values="percent",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex([BLOOM_LABELS[x] for x in BLOOM_ORDER])
        .reindex(columns=GRADE_ORDER)
    )
    bottom = np.zeros(len(pivot))
    colors = ["#7f7f7f", "#bcbd22", "#ffbf79", "#98df8a", "#2ca02c"]
    for grade, color, label in zip(GRADE_ORDER, colors, GRADE_LABELS, strict=False):
        values = pivot[grade].to_numpy()
        axes[0, 1].bar(
            pivot.index,
            values,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        bottom += values
    axes[0, 1].set_title("(b) Label Distribution by Bloom Level")
    axes[0, 1].set_ylabel("Share of responses (%)")
    axes[0, 1].legend(title="Label", frameon=False, ncol=5, loc="upper center")

    q_counts = data["question_label_counts"]
    heat = (
        q_counts.pivot_table(
            index="question_order",
            columns=GRADE_COL,
            values="percent",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=GRADE_ORDER)
        .sort_index()
    )
    image = axes[1, 0].imshow(heat.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    axes[1, 0].set_title("(c) Question-Level Label Distribution")
    axes[1, 0].set_xlabel("Gold label")
    axes[1, 0].set_ylabel("Questions ordered by mean score")
    axes[1, 0].set_xticks(np.arange(len(GRADE_ORDER)))
    axes[1, 0].set_xticklabels(GRADE_LABELS)
    axes[1, 0].set_yticks([0, len(heat) - 1])
    axes[1, 0].set_yticklabels(["Hardest", "Easiest"])
    cbar = fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Share (%)")

    student_df = data["student_perf"]
    axes[1, 1].hist(
        student_df["mean_score"],
        bins=np.linspace(0, 1, 16),
        color="#9467bd",
        edgecolor="#3d2457",
        linewidth=0.7,
    )
    axes[1, 1].axvline(
        student_df["mean_score"].mean(),
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean {student_df['mean_score'].mean():.2f}",
    )
    axes[1, 1].set_title("(d) Student Performance Distribution")
    axes[1, 1].set_xlabel("Mean gold score per student")
    axes[1, 1].set_ylabel("Students")
    axes[1, 1].legend(frameon=False)

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("VEX Dataset Overview for Data Card", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_data_csv(data: dict[str, pd.DataFrame]) -> None:
    frames: list[pd.DataFrame] = []
    for name in [
        "length_by_label",
        "bloom_label_counts",
        "question_label_counts",
        "question_difficulty",
        "student_perf",
    ]:
        frames.append(data[name].assign(table=name))
    pd.concat(frames, ignore_index=True, sort=False).to_csv(DATA_CSV, index=False, encoding="utf-8")


def write_sanity(df: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    lines = [
        "=" * 100,
        "FIGURE 11 DATASET DATA-CARD OVERVIEW SANITY CHECK",
        "=" * 100,
        f"Input parquet: {INPUT_PARQUET}",
        f"Rows:          {len(df)}",
        "",
        "Answer length by label:",
        data["length_by_label"].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        "Bloom-level label counts:",
        data["bloom_label_counts"].sort_values(["bloom_order", GRADE_COL]).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        ),
        "",
        "Question difficulty:",
        data["question_difficulty"].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        "Student performance summary:",
        data["student_perf"]["mean_score"].describe().to_string(),
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
    data = prepare_data(df)
    write_data_csv(data)
    save_plot(data)
    write_sanity(df, data)

    print("Saved:")
    print(f"  {PLOT_PDF}")
    print(f"  {PLOT_PNG}")
    print(f"  {DATA_CSV}")
    print(f"  {SANITY_TXT}")


if __name__ == "__main__":
    main()
