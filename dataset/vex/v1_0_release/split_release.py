#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

INPUT = Path("v1_0_stable.parquet")
OUT_DIR = INPUT.parent

SPLIT_COL = "split"

df = pd.read_parquet(INPUT)

if SPLIT_COL not in df.columns:
    raise ValueError(f"Missing required split column: {SPLIT_COL}")

for split_name in ["train", "test"]:
    part = df[df[SPLIT_COL] == split_name].copy()
    if part.empty:
        raise ValueError(f"No rows found for split={split_name!r}")

    output_path = OUT_DIR / f"{split_name}.parquet"
    part.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} with {len(part)} rows")