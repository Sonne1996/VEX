#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the relationship between audited LLM-gold status and answer length.

Input:
    dataset/additional/audit_dataset/audit_dataset.parquet

Outputs:
    results/plots/figures_plot_7_llm_gold_vs_answer_length/
        figure_7_llm_gold_vs_answer_length.pdf
        figure_7_llm_gold_vs_answer_length.png
        figure_7_llm_gold_vs_answer_length_bins.csv
        figure_7_llm_gold_vs_answer_length_sanity_check.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_PARQUET = (
    PROJECT_ROOT / "dataset" / "additional" / "audit_dataset" / "audit_dataset.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_7_llm_gold_vs_answer_length"

PLOT_PDF = OUTPUT_DIR / "figure_7_llm_gold_vs_answer_length.pdf"
PLOT_PNG = OUTPUT_DIR / "figure_7_llm_gold_vs_answer_length.png"
BINS_CSV = OUTPUT_DIR / "figure_7_llm_gold_vs_answer_length_bins.csv"
SANITY_TXT = OUTPUT_DIR / "figure_7_llm_gold_vs_answer_length_sanity_check.txt"

ANSWER_COL = "answer"
LLM_GOLD_COL = "gold_is_llm"

LENGTH_BINS = [0, 1, 3, 5, 10, 20, 40, 80, 160, np.inf]
LENGTH_LABELS = ["0", "1-2", "3-4", "5-9", "10-19", "20-39", "40-79", "80-159", "160+"]


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in input parquet:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def whitespace_token_count(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return len(text.split())


def normalize_yes_no(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "Yes"
    if text in {"no", "n", "false", "0"}:
        return "No"
    return "Unknown"


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[ANSWER_COL, LLM_GOLD_COL]].copy()
    work["answer_tokens"] = work[ANSWER_COL].map(whitespace_token_count)
    work["llm_gold_status"] = work[LLM_GOLD_COL].map(normalize_yes_no)
    work["is_llm_gold"] = work["llm_gold_status"].eq("Yes").astype(int)
    work["length_bin"] = pd.cut(
        work["answer_tokens"],
        bins=LENGTH_BINS,
        labels=LENGTH_LABELS,
        right=False,
        include_lowest=True,
    )

    grouped = (
        work.groupby("length_bin", observed=False)
        .agg(
            responses=("answer_tokens", "size"),
            llm_gold_count=("is_llm_gold", "sum"),
            mean_tokens=("answer_tokens", "mean"),
            median_tokens=("answer_tokens", "median"),
        )
        .reset_index()
    )
    grouped["llm_gold_rate_percent"] = np.where(
        grouped["responses"] > 0,
        grouped["llm_gold_count"] / grouped["responses"] * 100.0,
        np.nan,
    )
    grouped["non_llm_gold_count"] = grouped["responses"] - grouped["llm_gold_count"]

    return work, grouped


def save_plot(work: pd.DataFrame, grouped: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    x = np.arange(len(grouped))

    axes[0].bar(
        x,
        grouped["non_llm_gold_count"],
        label="gold_is_llm = No",
        color="#4c78a8",
        edgecolor="#263645",
        linewidth=0.7,
    )
    axes[0].bar(
        x,
        grouped["llm_gold_count"],
        bottom=grouped["non_llm_gold_count"],
        label="gold_is_llm = Yes",
        color="#f58518",
        edgecolor="#52310c",
        linewidth=0.7,
    )
    axes[0].set_title("(a) Answer Length Distribution")
    axes[0].set_ylabel("Responses")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grouped["length_bin"].astype(str), rotation=30, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        x,
        grouped["llm_gold_rate_percent"],
        color="#f58518",
        marker="o",
        linewidth=2.0,
    )
    for idx, row in grouped.iterrows():
        if pd.notna(row["llm_gold_rate_percent"]):
            axes[1].text(
                idx,
                float(row["llm_gold_rate_percent"]) + 0.8,
                f"{float(row['llm_gold_rate_percent']):.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[1].set_title("(b) LLM-Gold Rate by Answer Length")
    axes[1].set_ylabel("gold_is_llm = Yes (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grouped["length_bin"].astype(str), rotation=30, ha="right")
    axes[1].set_ylim(0, max(10.0, float(grouped["llm_gold_rate_percent"].max()) + 5.0))
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_xlabel("Answer length (whitespace tokens)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    total_yes = int(work["is_llm_gold"].sum())
    total = int(len(work))
    fig.suptitle(
        f"Audited LLM-Gold Status by Answer Length (Yes: {total_yes}/{total})",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sanity(work: pd.DataFrame, grouped: pd.DataFrame) -> None:
    status_counts = work["llm_gold_status"].value_counts(dropna=False)
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("AUDIT DATASET LLM-GOLD VS ANSWER LENGTH SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Input parquet: {INPUT_PARQUET}")
    lines.append(f"Rows:          {len(work)}")
    lines.append("")
    lines.append("gold_is_llm values after normalization:")
    lines.append(status_counts.to_string())
    lines.append("")
    lines.append("Answer length summary:")
    lines.append(work["answer_tokens"].describe().to_string())
    lines.append("")
    lines.append("Binned LLM-gold rate:")
    lines.append(
        grouped[
            [
                "length_bin",
                "responses",
                "llm_gold_count",
                "non_llm_gold_count",
                "llm_gold_rate_percent",
                "mean_tokens",
                "median_tokens",
            ]
        ].to_string(
            index=False,
            formatters={
                "llm_gold_rate_percent": lambda value: (
                    "nan" if pd.isna(value) else f"{float(value):.2f}"
                ),
                "mean_tokens": lambda value: "nan" if pd.isna(value) else f"{float(value):.2f}",
                "median_tokens": lambda value: (
                    "nan" if pd.isna(value) else f"{float(value):.2f}"
                ),
            },
        )
    )
    lines.append("")
    lines.append("Written files:")
    lines.append(f"  {PLOT_PDF}")
    lines.append(f"  {PLOT_PNG}")
    lines.append(f"  {BINS_CSV}")
    lines.append(f"  {SANITY_TXT}")

    SANITY_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PARQUET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PARQUET)
    require_columns(df, [ANSWER_COL, LLM_GOLD_COL])

    work, grouped = prepare_data(df)
    grouped.to_csv(BINS_CSV, index=False, encoding="utf-8")

    save_plot(work, grouped)
    write_sanity(work, grouped)

    print("Saved:")
    print(f"  {PLOT_PDF}")
    print(f"  {PLOT_PNG}")
    print(f"  {BINS_CSV}")
    print(f"  {SANITY_TXT}")


if __name__ == "__main__":
    main()
