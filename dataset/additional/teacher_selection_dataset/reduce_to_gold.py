#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

INPUT_PATH = Path("gold_with_all_models.parquet")
OUTPUT_PATH = Path("gold_with_all_models.parquet")

LABEL_TYPE_COL = "label_type"
SPLIT_COL = "split"

LABEL_TYPE_VALUE = "gold"
SPLIT_VALUE = "test"

CREATE_BACKUP = True


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH.resolve()}")

    df = pd.read_parquet(INPUT_PATH)

    required_cols = [LABEL_TYPE_COL, SPLIT_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(
            f"Missing required column(s): {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    print("=" * 80)
    print("FILTER PARQUET")
    print("=" * 80)
    print(f"Input path:  {INPUT_PATH.resolve()}")
    print(f"Output path: {OUTPUT_PATH.resolve()}")
    print(f"Original rows: {len(df):,}")

    filtered_df = df[
        (df[LABEL_TYPE_COL] == LABEL_TYPE_VALUE)
        & (df[SPLIT_COL] == SPLIT_VALUE)
    ].copy()

    print(f"Filtered rows: {len(filtered_df):,}")
    print()
    print("Filter:")
    print(f"  {LABEL_TYPE_COL} == {LABEL_TYPE_VALUE!r}")
    print(f"  {SPLIT_COL} == {SPLIT_VALUE!r}")

    if filtered_df.empty:
        raise ValueError(
            "Filtered dataframe is empty. No file was written. "
            "Check whether label_type='gold' and split='test' exist in the input file."
        )

    if CREATE_BACKUP and INPUT_PATH.resolve() == OUTPUT_PATH.resolve():
        backup_path = INPUT_PATH.with_suffix(".backup_before_filter.parquet")
        df.to_parquet(backup_path, index=False)
        print()
        print(f"Backup written to: {backup_path.resolve()}")

    filtered_df.to_parquet(OUTPUT_PATH, index=False)

    print()
    print(f"Filtered parquet written to: {OUTPUT_PATH.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()