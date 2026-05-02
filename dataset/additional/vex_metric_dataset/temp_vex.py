#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd


PARQUET_PATH = Path(__file__).resolve().parent / "merged_model_predictions.parquet"

COLUMNS_TO_DROP = [
    "human_grade_1",
    "human_grade_2",
    "grade_bert_base",
    "grade_mdeberta_base",
]


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)
    existing_drop_cols = [col for col in COLUMNS_TO_DROP if col in df.columns]

    df = df.drop(columns=existing_drop_cols)
    df.to_parquet(PARQUET_PATH, index=False)

    human_cols = [col for col in df.columns if "human" in col.lower()]

    print(f"Saved: {PARQUET_PATH}")
    print(f"Dropped columns: {existing_drop_cols}")
    print(f"Remaining human columns: {human_cols}")


if __name__ == "__main__":
    main()
