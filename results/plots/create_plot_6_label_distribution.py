#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create dataset label-distribution plots for the VEX v1.0 release.

Input:
    dataset/vex/v1_0_release/v1_0_stable.parquet

Outputs:
    results/plots/figures_plot_6_label_distribution/
        figure_6_label_distribution.pdf
        figure_6_label_distribution.png
        figure_6_label_distribution_matrices.pdf
        figure_6_label_distribution_matrices.png
        figure_6_label_distribution_counts.csv
        figure_6_label_distribution_matrices.csv
        figure_6_label_distribution_sanity_check.txt
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

INPUT_PARQUET = PROJECT_ROOT / "dataset" / "vex" / "v1_0_release" / "v1_0_stable.parquet"
OUTPUT_DIR = SCRIPT_DIR / "figures_plot_6_label_distribution"

PLOT_PDF = OUTPUT_DIR / "figure_6_label_distribution.pdf"
PLOT_PNG = OUTPUT_DIR / "figure_6_label_distribution.png"
MATRIX_PDF = OUTPUT_DIR / "figure_6_label_distribution_matrices.pdf"
MATRIX_PNG = OUTPUT_DIR / "figure_6_label_distribution_matrices.png"
COUNTS_CSV = OUTPUT_DIR / "figure_6_label_distribution_counts.csv"
MATRIX_CSV = OUTPUT_DIR / "figure_6_label_distribution_matrices.csv"
SANITY_TXT = OUTPUT_DIR / "figure_6_label_distribution_sanity_check.txt"

GRADE_COL = "grade"
LABEL_TYPE_COL = "label_type"

GRADE_ORDER = [0.0, 0.25, 0.5, 0.75, 1.0]
GRADE_LABELS = {
    0.0: "Incorrect\n0.00",
    0.25: "Mostly incorrect\n0.25",
    0.5: "Partially correct\n0.50",
    0.75: "Mostly correct\n0.75",
    1.0: "Correct\n1.00",
}


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in input parquet:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def grade_counts(df: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    counts = (
        pd.to_numeric(df[GRADE_COL], errors="coerce")
        .value_counts(dropna=False)
        .rename_axis("grade")
        .reset_index(name="count")
    )
    total = int(counts["count"].sum())
    counts["scope"] = scope_name
    counts["percent"] = np.where(total > 0, counts["count"] / total * 100.0, np.nan)

    ordered = pd.DataFrame({"grade": GRADE_ORDER})
    result = ordered.merge(counts, on="grade", how="left")
    result["scope"] = result["scope"].fillna(scope_name)
    result["count"] = result["count"].fillna(0).astype(int)
    result["percent"] = result["percent"].fillna(0.0)
    result["grade_label"] = result["grade"].map(GRADE_LABELS)
    result["scope_total"] = total
    return result[["scope", "scope_total", "grade", "grade_label", "count", "percent"]]


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle], values: list[float]) -> None:
    for bar, value in zip(bars, values, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_matrix_df(counts_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ["full", "gold", "silver"]:
        scope_df = counts_df[counts_df["scope"] == scope].sort_values("grade")
        for _, row in scope_df.iterrows():
            rows.append(
                {
                    "scope": scope,
                    "grade": row["grade"],
                    "grade_label": str(row["grade_label"]).replace("\n", " "),
                    "count": int(row["count"]),
                    "percent": float(row["percent"]),
                }
            )
    return pd.DataFrame(rows)


def matrix_values(matrix_df: pd.DataFrame, scope: str, value_col: str) -> np.ndarray:
    values = (
        matrix_df[matrix_df["scope"] == scope]
        .sort_values("grade")[value_col]
        .to_numpy(dtype=float)
    )
    return values.reshape(1, -1)


def annotate_matrix(
    ax: plt.Axes,
    values: np.ndarray,
    counts: np.ndarray,
    percent: np.ndarray,
) -> None:
    for col_idx in range(values.shape[1]):
        ax.text(
            col_idx,
            0,
            f"{int(counts[0, col_idx])}\n{percent[0, col_idx]:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if values[0, col_idx] >= values.max() * 0.55 else "black",
        )


def save_matrix_plot(matrix_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.4, 4.9),
        constrained_layout=True,
    )
    grade_labels = [
        GRADE_LABELS[grade].replace("\n", "\n") for grade in GRADE_ORDER
    ]
    scopes = [
        ("full", "(a) Full Dataset 100%"),
        ("silver", "(b) Silver/Train 90%"),
        ("gold", "(c) Gold/Test 10%"),
    ]
    max_percent = float(matrix_df["percent"].max())

    for ax, (scope, title) in zip(axes, scopes, strict=False):
        percent = matrix_values(matrix_df, scope, "percent")
        counts = matrix_values(matrix_df, scope, "count")
        image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=max_percent, aspect="auto")
        annotate_matrix(ax, percent, counts, percent)
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks(np.arange(len(GRADE_ORDER)))
        ax.set_xticklabels(grade_labels, fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Share of responses (%)")
    fig.suptitle("Label Distribution Matrices", fontsize=13)
    fig.savefig(MATRIX_PDF, bbox_inches="tight")
    fig.savefig(MATRIX_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plot(counts_df: pd.DataFrame) -> None:
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

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), sharey=True)
    x = np.arange(len(GRADE_ORDER))

    full = counts_df[counts_df["scope"] == "full"].sort_values("grade")
    bars = axes[0].bar(
        x,
        full["percent"],
        color="#4c78a8",
        edgecolor="#263645",
        linewidth=0.8,
    )
    annotate_bars(axes[0], list(bars), full["percent"].tolist())
    axes[0].set_title("(a) Full VEX v1.0 Dataset")
    axes[0].set_ylabel("Share of responses (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(full["grade_label"], rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.28)

    gold = counts_df[counts_df["scope"] == "gold"].sort_values("grade")
    silver = counts_df[counts_df["scope"] == "silver"].sort_values("grade")
    width = 0.38
    bars_gold = axes[1].bar(
        x - width / 2,
        gold["percent"],
        width=width,
        label="Gold/test 10%",
        color="#f58518",
        edgecolor="#52310c",
        linewidth=0.8,
    )
    bars_silver = axes[1].bar(
        x + width / 2,
        silver["percent"],
        width=width,
        label="Silver/train 90%",
        color="#54a24b",
        edgecolor="#264b22",
        linewidth=0.8,
    )
    annotate_bars(axes[1], list(bars_gold), gold["percent"].tolist())
    annotate_bars(axes[1], list(bars_silver), silver["percent"].tolist())
    axes[1].set_title("(b) Gold vs. Silver Label Distribution")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gold["grade_label"], rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.28)
    axes[1].legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.set_ylim(0, max(65.0, counts_df["percent"].max() + 8.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Label Distribution in the VEX v1.0 Release", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_sanity(
    df: pd.DataFrame,
    counts_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("VEX V1.0 LABEL DISTRIBUTION SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Input parquet: {INPUT_PARQUET}")
    lines.append(f"Rows:          {len(df)}")
    lines.append("")
    lines.append("Rows by label_type:")
    lines.append(df[LABEL_TYPE_COL].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("Grade distribution by scope:")
    for scope in ["full", "gold", "silver"]:
        scope_df = counts_df[counts_df["scope"] == scope].sort_values("grade")
        lines.append("")
        lines.append(f"[{scope}] n={int(scope_df['scope_total'].iloc[0])}")
        lines.append(
            scope_df[["grade", "count", "percent"]].to_string(
                index=False,
                formatters={"percent": lambda value: f"{float(value):.2f}"},
            )
        )
    lines.append("")
    lines.append("Matrix table:")
    lines.append(
        matrix_df.to_string(
            index=False,
            formatters={"percent": lambda value: f"{float(value):.2f}"},
        )
    )
    lines.append("")
    lines.append("Written files:")
    lines.append(f"  {PLOT_PDF}")
    lines.append(f"  {PLOT_PNG}")
    lines.append(f"  {MATRIX_PDF}")
    lines.append(f"  {MATRIX_PNG}")
    lines.append(f"  {COUNTS_CSV}")
    lines.append(f"  {MATRIX_CSV}")
    lines.append(f"  {SANITY_TXT}")

    SANITY_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PARQUET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PARQUET)
    require_columns(df, [GRADE_COL, LABEL_TYPE_COL])

    label_type = df[LABEL_TYPE_COL].astype("string").str.lower()
    full_counts = grade_counts(df, "full")
    gold_counts = grade_counts(df[label_type == "gold"], "gold")
    silver_counts = grade_counts(df[label_type == "silver"], "silver")

    counts_df = pd.concat([full_counts, gold_counts, silver_counts], ignore_index=True)
    counts_df.to_csv(COUNTS_CSV, index=False, encoding="utf-8")
    matrix_df = build_matrix_df(counts_df)
    matrix_df.to_csv(MATRIX_CSV, index=False, encoding="utf-8")

    save_plot(counts_df)
    save_matrix_plot(matrix_df)
    write_sanity(df, counts_df, matrix_df)

    print("Saved:")
    print(f"  {PLOT_PDF}")
    print(f"  {PLOT_PNG}")
    print(f"  {MATRIX_PDF}")
    print(f"  {MATRIX_PNG}")
    print(f"  {COUNTS_CSV}")
    print(f"  {MATRIX_CSV}")
    print(f"  {SANITY_TXT}")


if __name__ == "__main__":
    main()
