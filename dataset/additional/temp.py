#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge audited human-grade columns into merged_model_predictions.parquet.

Input 1:
    dataset/additional/audit_dataset/audit_dataset.parquet

Input 2 / output:
    dataset/additional/vex_metric_dataset/merged_model_predictions.parquet

The script:
    1. Checks whether audit_dataset.grade is the numeric representation of
       audit_dataset.gold_label_after_human_audit.
    2. Maps audit_dataset["human_grade 1"] and ["human_grade 2"] to numeric
       values.
    3. Writes human_grade_1 / human_grade_2 as trace columns.
    4. Writes human_expert_one_gold / human_expert_two for VEX evaluation:
       human_grade 1 is fake gold, human_grade 2 is evaluated against it.
    5. Merges those mapped columns into merged_model_predictions.parquet by
       answer_id.
    6. Overwrites merged_model_predictions.parquet and writes a sanity-check txt.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

AUDIT_PARQUET = (
    PROJECT_ROOT
    / "dataset"
    / "additional"
    / "audit_dataset"
    / "audit_dataset.parquet"
)

PREDICTIONS_PARQUET = (
    PROJECT_ROOT
    / "dataset"
    / "additional"
    / "vex_metric_dataset"
    / "merged_model_predictions.parquet"
)

OUTPUT_PARQUET = PREDICTIONS_PARQUET

SANITY_CHECK_TXT = (
    PROJECT_ROOT
    / "dataset"
    / "additional"
    / "vex_metric_dataset"
    / "merged_model_predictions_human_grades_sanity_check.txt"
)

MERGE_KEY = "answer_id"

HUMAN_GRADE_1_SOURCE = "human_grade 1"
HUMAN_GRADE_2_SOURCE = "human_grade 2"
HUMAN_GRADE_1_OUTPUT = "human_grade_1"
HUMAN_GRADE_2_OUTPUT = "human_grade_2"
HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT = "human_expert_one_gold"
HUMAN_EXPERT_TWO_OUTPUT = "human_expert_two"

GRADE_LABEL_COL = "gold_label_after_human_audit"
GRADE_NUMERIC_COL = "grade"

LABEL_TO_NUMERIC = {
    "incorrect": 0.0,
    "mostly incorrect": 0.25,
    "partially correct": 0.5,
    "mostly correct": 0.75,
    "correct": 1.0,
}


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    return pd.read_parquet(path)


def require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {path}:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def assert_unique_key(df: pd.DataFrame, key: str, path: Path) -> None:
    missing = int(df[key].isna().sum())
    if missing:
        raise ValueError(f"{path} contains {missing} missing values in {key}.")

    duplicates = df[df[key].duplicated(keep=False)]
    if not duplicates.empty:
        preview = duplicates[[key]].head(20).to_string(index=False)
        raise ValueError(
            f"{path} contains duplicate {key} values.\n"
            f"Examples:\n{preview}"
        )


def map_grade_labels(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(LABEL_TO_NUMERIC)

    unknown = sorted(normalized[mapped.isna() & normalized.notna()].unique().tolist())
    if unknown:
        raise ValueError(
            "Unknown grade labels found:\n"
            + "\n".join(f"  - {value}" for value in unknown)
        )

    return mapped.astype("float64")


def build_sanity_check(
    audit_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> str:
    audit_grade_numeric = pd.to_numeric(audit_df[GRADE_NUMERIC_COL], errors="coerce")
    audit_label_numeric = map_grade_labels(audit_df[GRADE_LABEL_COL])

    equal_mask = audit_grade_numeric.eq(audit_label_numeric)
    both_missing_mask = audit_grade_numeric.isna() & audit_label_numeric.isna()
    sanity_equal_mask = equal_mask | both_missing_mask

    mismatch_df = audit_df.loc[
        ~sanity_equal_mask,
        [MERGE_KEY, GRADE_NUMERIC_COL, GRADE_LABEL_COL],
    ].copy()
    mismatch_df["mapped_gold_label_after_human_audit"] = audit_label_numeric[
        ~sanity_equal_mask
    ].to_numpy()

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("MERGED MODEL PREDICTIONS HUMAN GRADES SANITY CHECK")
    lines.append("=" * 100)
    lines.append(f"Audit parquet:       {AUDIT_PARQUET}")
    lines.append(f"Predictions parquet: {PREDICTIONS_PARQUET}")
    lines.append(f"Output parquet:      {OUTPUT_PARQUET}")
    lines.append(f"Merge key:           {MERGE_KEY}")
    lines.append("")
    lines.append("Input rows:")
    lines.append(f"  audit_dataset:              {len(audit_df)}")
    lines.append(f"  merged_model_predictions:   {len(merged_df)}")
    lines.append(f"  output:                     {len(output_df)}")
    lines.append("")
    lines.append("Human-grade column mapping:")
    for label, value in LABEL_TO_NUMERIC.items():
        lines.append(f"  {label}: {value}")
    lines.append("")
    lines.append("Audit grade sanity check:")
    lines.append(
        "  Check: grade == mapped(gold_label_after_human_audit)"
    )
    lines.append(f"  Equal rows:                 {int(sanity_equal_mask.sum())}")
    lines.append(f"  Mismatch rows:              {len(mismatch_df)}")
    lines.append("")
    lines.append("Mapped human-grade missing values in output:")
    lines.append(
        f"  {HUMAN_GRADE_1_OUTPUT}: {int(output_df[HUMAN_GRADE_1_OUTPUT].isna().sum())}"
    )
    lines.append(
        f"  {HUMAN_GRADE_2_OUTPUT}: {int(output_df[HUMAN_GRADE_2_OUTPUT].isna().sum())}"
    )
    lines.append(
        f"  {HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT}: "
        f"{int(output_df[HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT].isna().sum())}"
    )
    lines.append(
        f"  {HUMAN_EXPERT_TWO_OUTPUT}: "
        f"{int(output_df[HUMAN_EXPERT_TWO_OUTPUT].isna().sum())}"
    )
    lines.append("")
    lines.append("VEX human-expert comparison columns:")
    lines.append(
        f"  {HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT} = mapped {HUMAN_GRADE_1_SOURCE}"
    )
    lines.append(
        f"  {HUMAN_EXPERT_TWO_OUTPUT} = mapped {HUMAN_GRADE_2_SOURCE}"
    )
    lines.append(
        "  evaluate_dataframe.py treats human_expert_one_gold as fake gold "
        "only for model human_expert_two."
    )
    lines.append("")

    if mismatch_df.empty:
        lines.append("No mismatches found.")
    else:
        lines.append("Mismatch preview:")
        lines.append(mismatch_df.head(50).to_string(index=False))

    return "\n".join(lines)


def main() -> None:
    audit_df = read_parquet(AUDIT_PARQUET)
    predictions_df = read_parquet(PREDICTIONS_PARQUET)

    require_columns(
        audit_df,
        [
            MERGE_KEY,
            GRADE_NUMERIC_COL,
            GRADE_LABEL_COL,
            HUMAN_GRADE_1_SOURCE,
            HUMAN_GRADE_2_SOURCE,
        ],
        AUDIT_PARQUET,
    )
    require_columns(predictions_df, [MERGE_KEY], PREDICTIONS_PARQUET)

    assert_unique_key(audit_df, MERGE_KEY, AUDIT_PARQUET)
    assert_unique_key(predictions_df, MERGE_KEY, PREDICTIONS_PARQUET)

    audit_merge_df = audit_df[
        [MERGE_KEY, HUMAN_GRADE_1_SOURCE, HUMAN_GRADE_2_SOURCE]
    ].copy()
    audit_merge_df[HUMAN_GRADE_1_OUTPUT] = map_grade_labels(
        audit_merge_df[HUMAN_GRADE_1_SOURCE]
    )
    audit_merge_df[HUMAN_GRADE_2_OUTPUT] = map_grade_labels(
        audit_merge_df[HUMAN_GRADE_2_SOURCE]
    )
    audit_merge_df[HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT] = audit_merge_df[
        HUMAN_GRADE_1_OUTPUT
    ]
    audit_merge_df[HUMAN_EXPERT_TWO_OUTPUT] = audit_merge_df[
        HUMAN_GRADE_2_OUTPUT
    ]
    audit_merge_df = audit_merge_df[
        [
            MERGE_KEY,
            HUMAN_GRADE_1_OUTPUT,
            HUMAN_GRADE_2_OUTPUT,
            HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT,
            HUMAN_EXPERT_TWO_OUTPUT,
        ]
    ]

    output_df = predictions_df.drop(
        columns=[
            HUMAN_GRADE_1_OUTPUT,
            HUMAN_GRADE_2_OUTPUT,
            HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT,
            HUMAN_EXPERT_TWO_OUTPUT,
        ],
        errors="ignore",
    ).merge(
        audit_merge_df,
        on=MERGE_KEY,
        how="left",
        validate="one_to_one",
    )

    missing_after_merge = output_df[
        output_df[HUMAN_GRADE_1_OUTPUT].isna()
        | output_df[HUMAN_GRADE_2_OUTPUT].isna()
        | output_df[HUMAN_EXPERT_ONE_FAKE_GOLD_OUTPUT].isna()
        | output_df[HUMAN_EXPERT_TWO_OUTPUT].isna()
    ]
    if not missing_after_merge.empty:
        preview = missing_after_merge[[MERGE_KEY]].head(20).to_string(index=False)
        raise ValueError(
            "Some prediction rows did not receive both human-grade columns.\n"
            f"Rows with missing mapped human grades: {len(missing_after_merge)}\n"
            f"Examples:\n{preview}"
        )

    sanity_text = build_sanity_check(
        audit_df=audit_df,
        merged_df=predictions_df,
        output_df=output_df,
    )

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(OUTPUT_PARQUET, index=False)
    SANITY_CHECK_TXT.write_text(sanity_text, encoding="utf-8")

    print("Saved parquet:")
    print(f"  {OUTPUT_PARQUET}")
    print("Saved sanity check:")
    print(f"  {SANITY_CHECK_TXT}")


if __name__ == "__main__":
    main()
